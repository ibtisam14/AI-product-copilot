import json
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.parsers import MultiPartParser, FormParser
from .adapters import MockAdapter, OpenAIAdapter
from .utils import (
    chunk_faq_markdown,
    store_product_and_embeddings,
    store_faq_chunks_and_embeddings,
    retrieve_top_k,
)
from .models import Product, FAQChunk, EmbeddingVector
from django.conf import settings
from django.shortcuts import render


def home(request):
    return render(request, "index.html")


# Choose adapter: in production change to OpenAIAdapter if OPENAI_API_KEY set
def get_adapter():
    if getattr(settings, "OPENAI_API_KEY", ""):
        return OpenAIAdapter(settings.OPENAI_API_KEY)
    return MockAdapter()


@method_decorator(csrf_exempt, name="dispatch")
class UploadIngestView(APIView):
    """
    POST files:
      - products.csv
      - faq.md
    """

    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        adapter = get_adapter()
        products_file = request.FILES.get("products.csv") or request.FILES.get("products")
        faq_file = request.FILES.get("faq.md") or request.FILES.get("faq")
        results = {}

        if products_file:
            import csv, io

            text = products_file.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))
            prods = []
            texts = []
            for row in reader:
                prod = {
                    "id": row.get("id") or row.get("ID") or row.get("Id"),
                    "name": row.get("name", ""),
                    "notes": row.get("notes", ""),
                    "accords": row.get("accords", ""),
                    "price": float(row.get("price", 0) or 0),
                    "longevity": row.get("longevity", ""),
                    "season": row.get("season", ""),
                    "imageUrl": row.get("imageUrl", ""),
                    "popularity": float(row.get("popularity", 0) or 0),
                }
                prods.append(prod)
                texts.append((prod["id"], f"{prod['name']}. {prod['notes']}"))

            embed_texts = [t[1] for t in texts]
            vectors = adapter.get_embeddings(embed_texts)
            store_product_and_embeddings(prods, vectors)
            results["products"] = len(prods)

        if faq_file:
            md_text = faq_file.read().decode("utf-8")
            chunks = chunk_faq_markdown(md_text)
            chunk_objs = []
            texts = []
            for i, (heading, chunk_text) in enumerate(chunks):
                cid = f"faq_{i+1}"
                chunk_objs.append({"id": cid, "heading": heading, "text": chunk_text})
                texts.append(chunk_text)
            vectors = adapter.get_embeddings(texts)
            store_faq_chunks_and_embeddings(chunk_objs, vectors)
            results["faq_chunks"] = len(chunk_objs)

        if not results:
            return Response({"detail": "No files provided"}, status=status.HTTP_400_BAD_REQUEST)
        return Response(results)


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
            resp = adapter.get_completion(messages=messages, mode=mode, context_snippets=provided_context)
            return Response(resp)

        if not messages or "content" not in messages[-1]:
            return Response({"error": "messages must include content"}, status=400)

        query_text = messages[-1]["content"]
        vectors = adapter.get_embeddings([query_text])
        query_vec = vectors[0]

        top = retrieve_top_k(query_vec, k=8)
        top3 = top[:3]
        context_snippets = [{"id": t["id"], "source": t["source"], "text": t["text"]} for t in top3]

        resp = adapter.get_completion(messages=messages, mode=mode, context_snippets=context_snippets)

        if "citations" not in resp:
            resp["citations"] = [c["id"] for c in context_snippets]

        return Response(resp)
