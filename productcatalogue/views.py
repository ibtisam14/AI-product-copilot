# productcatalogue/views.py
import os
import io
import csv
import fitz  # PyMuPDF
import PyPDF2  # kept if needed
from chromadb import PersistentClient

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.parsers import MultiPartParser, FormParser

from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.shortcuts import render

from .adapters import MockAdapter, OpenAIAdapter
from .db import collection
from .utils import (
    chunk_faq_markdown,
    chunk_plain_text,
    store_product_and_embeddings,
    store_faq_chunks_and_embeddings,
    retrieve_top_k,
)
from .models import Product, FAQChunk


def home(request):
    return render(request, "index.html")


def upload_page(request):
    return render(request, "upload.html")


@method_decorator(csrf_exempt, name="dispatch")
class GetDataView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        prods = list(Product.objects.all().values())
        faqs = list(FAQChunk.objects.all().values("id", "heading", "text"))
        return Response({"products": prods, "faqs": faqs})


def get_adapter():
    api_key = getattr(settings, "OPENAI_API_KEY", "")
    if api_key:
        print("✅ Using OpenAI Adapter")
        return OpenAIAdapter(api_key)
    else:
        print("⚠️ Using Mock Adapter (no API key found)")
        return MockAdapter()


@method_decorator(csrf_exempt, name="dispatch")
class UploadIngestView(APIView):
    """
    POST files:
      - products.csv
      - faq.md or faq.pdf
    """

    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        adapter = get_adapter()
        results = {}

        try:
            # ------------------------
            # Step 0: Save uploaded files to MEDIA folder
            # ------------------------
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
            saved_files = {}  # maps original upload key -> saved absolute path
            for key, file in request.FILES.items():
                # Save file to MEDIA_ROOT
                saved_name = fs.save(file.name, file)
                file_path = os.path.join(settings.MEDIA_ROOT, saved_name)
                saved_files[key] = file_path
                print(f"📥 File saved to: {file_path}")

            # ------------------------
            # Handle PRODUCTS upload
            # ------------------------
            products_file = request.FILES.get("products.csv") or request.FILES.get("products")
            if products_file:
                print(f"📦 Detected product upload: {products_file.name}")
                # If saved to disk, read from in-memory file for CSV decoding (works either way)
                text = products_file.read().decode("utf-8")
                reader = csv.DictReader(io.StringIO(text))
                prods = []
                texts = []
                for row in reader:
                    raw_price = row.get("price", 0)
                    try:
                        price = float(raw_price) if raw_price not in (None, "", "null", "NULL") else 0.0
                    except (ValueError, TypeError):
                        price = 0.0

                    prod = {
                        "id": row.get("id") or row.get("ID") or row.get("Id"),
                        "name": row.get("name", ""),
                        "notes": row.get("notes", ""),
                        "accords": row.get("accords", ""),
                        "price": price,
                        "longevity": row.get("longevity", ""),
                        "season": row.get("season", ""),
                        "imageUrl": row.get("imageUrl", ""),
                        "popularity": float(row.get("popularity", 0) or 0),
                    }
                    prods.append(prod)
                    embed_text = (
                        f"Product name: {prod['name']}. "
                        f"Description: {prod['notes']}. "
                        f"Price: ${price:.2f}. "
                        f"Features/Accords: {prod['accords']}. "
                        f"Longevity: {prod['longevity']}. "
                        f"Recommended season: {prod['season']}."
                    )
                    texts.append((prod["id"], embed_text))

                embed_texts = [t[1] for t in texts]
                vectors = adapter.get_embeddings(embed_texts)
                store_product_and_embeddings(prods, vectors)
                results["products"] = len(prods)
                print(f"✅ Stored {len(prods)} products successfully.")

            # ------------------------
            # Handle FAQ upload (.md or .pdf)
            # ------------------------
            faq_file = None
            faq_upload_key = None
            for key, f in request.FILES.items():
                if f.name.lower().endswith(".pdf") or key.lower().endswith(".pdf"):
                    faq_file = f
                    faq_upload_key = key
                    print(f"📂 Detected PDF upload with key='{key}', filename='{f.name}'")
                    break
                elif f.name.lower().endswith(".md") or key.lower() in ["faq", "faq.md"]:
                    faq_file = f
                    faq_upload_key = key
                    print(f"📄 Detected Markdown upload with key='{key}', filename='{f.name}'")
                    break

            # Ensure Chroma persistent client and collection
            chroma_client = PersistentClient(path="chroma_db")
            chroma_collection = chroma_client.get_or_create_collection("faq_collection")

            if faq_file:
                file_name = faq_file.name.lower()
                # prefer using the saved file path (so we open file from disk)
                file_path = saved_files.get(faq_upload_key)

                # PDF branch
                if file_name.endswith(".pdf"):
                    print("🧾 Starting PDF processing (using fitz)...")

                    # Open from saved file path when possible (more reliable)
                    if file_path and os.path.exists(file_path):
                        pdf = fitz.open(file_path)
                    else:
                        # fallback to file-like object
                        pdf_bytes = faq_file.read()
                        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

                    chunks = []
                    for page_num, page in enumerate(pdf):
                        page_text = page.get_text("text") or ""
                        text = page_text.strip()
                        print(f"   → Extracted page {page_num + 1} with {len(text)} characters")
                        if text:
                            chunks.append({
                                "id": f"faq_pdf_{page_num + 1}",
                                "heading": f"Page {page_num + 1}",
                                "text": text,
                            })

                    print(f"✅ Total extracted PDF characters: {sum(len(c['text']) for c in chunks)}")
                    print(f"🧩 Created {len(chunks)} chunks from PDF")

                    texts = [c["text"] for c in chunks]
                    if texts:
                        vectors = adapter.get_embeddings(texts)
                        # vectors could be fallback vectors if API failed; still save
                        for i, chunk in enumerate(chunks):
                            try:
                                chroma_collection.add(
                                    documents=[chunk["text"]],
                                    embeddings=[vectors[i]],
                                    metadatas=[{"source": "pdf"}],
                                    ids=[chunk["id"]]
                                )
                            except Exception as e:
                                print(f"⚠️ Error saving chunk {chunk['id']} to ChromaDB: {e}")
                        # persist into your DB tables as well
                        store_faq_chunks_and_embeddings(chunks, vectors)
                        results["faq_chunks"] = len(chunks)
                        print(f"✅ Stored {len(chunks)} PDF chunks successfully.")
                    else:
                        print("⚠️ No text extracted from PDF.")

                    # close pdf if applicable
                    try:
                        pdf.close()
                    except Exception:
                        pass

                # Markdown branch
                else:
                    print("📘 Processing Markdown FAQ file...")
                    md_text = faq_file.read().decode("utf-8")
                    chunks = chunk_faq_markdown(md_text)
                    print(f"🧩 Created {len(chunks)} chunks from Markdown")
                    chunk_objs = []
                    texts = []
                    for i, (heading, chunk_text) in enumerate(chunks):
                        cid = f"faq_{i + 1}"
                        chunk_objs.append({"id": cid, "heading": heading, "text": chunk_text})
                        texts.append(chunk_text)

                    vectors = adapter.get_embeddings(texts)
                    store_faq_chunks_and_embeddings(chunk_objs, vectors)
                    results["faq_chunks"] = len(chunk_objs)
                    print(f"✅ Stored {len(chunk_objs)} FAQ chunks successfully.")

            else:
                print("⚠️ No FAQ file (.md or .pdf) found in upload request.")

            # ------------------------
            # Final response (always return something)
            # ------------------------
            if not results:
                return Response({"detail": "No files provided"}, status=status.HTTP_400_BAD_REQUEST)

            return Response(results, status=status.HTTP_200_OK)

        except Exception as e:
            # Unexpected error — return JSON error and log to console
            print(f"❌ UploadIngestView exception: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name="dispatch")
class EmbeddingsView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        adapter = get_adapter()
        texts = request.data.get("texts", [])
        if not isinstance(texts, list) or not texts:
            return Response({"error": "texts must be a non-empty list"}, status=400)
        vectors = adapter.get_embeddings(texts)
        return Response({"vectors": vectors})


@method_decorator(csrf_exempt, name="dispatch")
class ChatView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        data = request.data
        messages = data.get("messages", [])
        provided_context = data.get("context_snippets", [])
        mode = data.get("mode", "fast")

        adapter = get_adapter()

        if provided_context:
            resp = adapter.get_completion(
                messages=messages, mode=mode, context_snippets=provided_context
            )
            return Response(resp)

        if not messages or "content" not in messages[-1]:
            return Response({"error": "messages must include content"}, status=400)

        query_text = messages[-1]["content"]
        vectors = adapter.get_embeddings([query_text])
        if not vectors:
            return Response({"error": "Failed to compute query embedding"}, status=500)
        query_vec = vectors[0]

        top = retrieve_top_k(query_vec, k=8)

        if not top:
            return Response(
                {"answer": "Sorry, I couldn't find any relevant information.", "citations": []}
            )

        top3 = top[:3]
        context_snippets = [
            {"id": t["id"], "source": t["source"], "text": t["text"]} for t in top3
        ]

        resp = adapter.get_completion(
            messages=messages, mode=mode, context_snippets=context_snippets
        )

        if "citations" not in resp:
            resp["citations"] = [c["id"] for c in context_snippets]

        return Response(resp)
