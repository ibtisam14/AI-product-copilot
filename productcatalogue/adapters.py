import json
import hashlib
from django.conf import settings
import chromadb


class MockAdapter:
    """A simple mock adapter for testing or demo use."""

    def __init__(self):
        pass

    def get_embeddings(self, texts):
        """Generate deterministic fake embeddings using hashing."""
        vectors = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [((b % 128) / 127.0) * (1 if i % 2 == 0 else -1)
                   for i, b in enumerate(h[:16])]
            vectors.append(vec)
        return vectors

    def get_completion(self, messages, mode, context_snippets):
        """Generate fake completion response with citations."""
        snippet_ids = [s.get("id") for s in context_snippets]
        product_ids = [s["id"] for s in context_snippets if s.get("source") == "product"]
        answer = "Mock answer: I used provided snippets to answer."

        if 'compare' in messages[-1]["content"].lower() and len(product_ids) >= 2:
            answer = f"Mock compare: {product_ids[0]} seems stronger than {product_ids[1]}."
            citations = product_ids[:2]
        else:
            citations = snippet_ids[:3]

        return {"answer": answer, "citations": citations}


class OpenAIAdapter:
    """Adapter for OpenAI or OpenRouter embedding + chat completions."""

    def __init__(self, api_key):
        try:
            from openai import OpenAI as OpenAIClient
            base_url = getattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.client = OpenAIClient(api_key=api_key, base_url=base_url)
            self.client_type = "openai_sdk_object"
            print(f"✅ Connected to OpenAI client with base_url={base_url}")
        except Exception:
            try:
                import openai
                openai.api_key = api_key
                openai.api_base = getattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1")
                self.client = openai
                self.client_type = "openai_legacy"
                print(f"✅ Connected using legacy OpenAI client with base_url={openai.api_base}")
            except Exception:
                raise

        # 🟢 Persistent Chroma client (saved on disk)
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="faq_collection")
        print("✅ Connected to ChromaDB (persistent collection: faq_collection)")

    # ✅ FIXED FINAL VERSION — handles all response formats
    def get_embeddings(self, texts):
        """Get and store embeddings safely, compatible with OpenAI + OpenRouter."""
        try:
            if isinstance(texts, str):
                texts = [texts]

            # Call embedding API (SDK or legacy)
            if self.client_type == "openai_sdk_object":
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts
                )
            else:
                response = self.client.Embedding.create(
                    model="text-embedding-3-small",
                    input=texts
                )

            # 🧠 Handle multiple response formats
            if isinstance(response, str):
                response = json.loads(response)

            # Extract data safely
            data = None
            if hasattr(response, "data"):
                data = response.data
            elif isinstance(response, dict) and "data" in response:
                data = response["data"]
            else:
                print("⚠️ Unexpected embedding response format:", type(response))
                data = []

            embeddings = []
            for item in data:
                emb = None
                if isinstance(item, dict):
                    emb = item.get("embedding")
                elif hasattr(item, "embedding"):
                    emb = item.embedding
                if emb is not None:
                    embeddings.append([float(x) for x in emb])

            if not embeddings:
                print("⚠️ No embeddings returned — using fallback.")
                return self._get_fallback_embeddings(texts)

            # 🟢 Save embeddings to ChromaDB
            for i, text in enumerate(texts):
                try:
                    self.collection.add(
                        documents=[text],
                        embeddings=[embeddings[i]],
                        metadatas=[{"source": "pdf"}],
                        ids=[f"doc_{hash(text)}"]
                    )
                except Exception as e:
                    print(f"⚠️ Error saving to ChromaDB: {e}")

            print(f"✅ Got and saved {len(embeddings)} embeddings to ChromaDB.")
            return embeddings

        except Exception as e:
            print(f"❌ Error getting embeddings: {e}")
            return self._get_fallback_embeddings(texts)

    def _get_fallback_embeddings(self, texts):
        """Fallback deterministic pseudo-embeddings if API fails."""
        vectors = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [((b % 128) / 127.0) * (1 if i % 2 == 0 else -1)
                   for i, b in enumerate(h[:16])]
            vectors.append(vec)
        return vectors

    def get_completion(self, messages, mode, context_snippets):
        """Generate a completion using context and user question."""
        if mode == "fast":
            model_name = "gpt-3.5-turbo"
            temperature = 0.7
        else:
            model_name = "gpt-3.5-turbo"
            temperature = 0.2

        base_url = getattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        if "openrouter.ai" in base_url:
            model_name = "openai/gpt-3.5-turbo"

        system_prompt = (
            "You are a precise product/FAQ assistant. Answer the user's question using ONLY the context below.\n"
            "- If the context contains specific facts, quote them exactly.\n"
            "- If you cannot find an answer, say: \"I don't know based on the provided info.\"\n"
            "- Always end with: Sources: [list of source IDs].\n\n"
            "Context:\n"
        )

        context_text = "\n\n".join([f"[{s['id']}]: {s['text']}" for s in context_snippets])
        full_context = system_prompt + context_text

        user_query = messages[-1]["content"] if messages else "Hello"
        enhanced_messages = [
            {"role": "system", "content": full_context},
            {"role": "user", "content": user_query}
        ]

        try:
            if self.client_type == "openai_sdk_object":
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=enhanced_messages,
                    temperature=temperature,
                    max_tokens=300
                )
                answer = response.choices[0].message.content.strip()
            else:
                response = self.client.ChatCompletion.create(
                    model=model_name,
                    messages=enhanced_messages,
                    temperature=temperature,
                    max_tokens=300
                )
                answer = response["choices"][0]["message"]["content"].strip()

            citations = [s["id"] for s in context_snippets[:3]]
            return {"answer": answer, "citations": citations}

        except Exception as e:
            print(f"❌ Error getting completion: {e}")
            return self._get_fallback_completion(messages, mode, context_snippets)

    def _get_fallback_completion(self, messages, mode, context_snippets):
        """Fallback to mock completion."""
        snippet_ids = [s.get("id") for s in context_snippets]
        answer = "Fallback: I used the provided context snippets."
        citations = snippet_ids[:3]
        return {"answer": answer, "citations": citations}
