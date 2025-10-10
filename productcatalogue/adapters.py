import json
import hashlib
from django.conf import settings


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

    def get_embeddings(self, texts):
        """Robust embedding getter compatible with multiple OpenAI SDK shapes"""
        try:
            if isinstance(texts, str):
                texts = [texts]

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

            # Normalize response
            if isinstance(response, dict):
                data = response.get("data", [])
            elif isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    data = parsed.get("data", [])
                except Exception:
                    data = []
            else:
                try:
                    data = getattr(response, "data", [])
                except Exception:
                    data = []

            embeddings = []
            for item in data:
                if isinstance(item, dict):
                    emb = item.get("embedding")
                else:
                    emb = getattr(item, "embedding", None)
                if emb is not None:
                    embeddings.append([float(x) for x in list(emb)])

            if not embeddings:
                return self._get_fallback_embeddings(texts)

            return embeddings

        except Exception as e:
            print("❌ Error getting embeddings:", e)
            return self._get_fallback_embeddings(texts)

    def _get_fallback_embeddings(self, texts):
        """Fallback to deterministic pseudo-embeddings"""
        vectors = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [((b % 128) / 127.0) * (1 if i % 2 == 0 else -1)
                   for i, b in enumerate(h[:16])]
            vectors.append(vec)
        return vectors

    def get_completion(self, messages, mode, context_snippets):
        """Chat completion handler with improved prompt for price/hours/rating"""
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
            "You are a precise product assistant. Answer the user's question using ONLY the context below.\n"
            "- If the context contains a **price**, state it exactly (e.g., $99.99).\n"
            "- If asked about **battery life**, report the number of hours (e.g., 12 hours).\n"
            "- If asked about **rating**, give the numerical score (e.g., 4.8).\n"
            "- If the answer is not in the context, say: \"I don't know based on the provided info.\"\n"
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
                answer = None
                if hasattr(response, "choices"):
                    ch = getattr(response, "choices", None)
                    if ch and len(ch) > 0:
                        msg = getattr(ch[0], "message", None)
                        if isinstance(msg, dict):
                            answer = msg.get("content")
                        elif hasattr(msg, "content"):
                            answer = msg.content
                if not answer:
                    answer = "I don't know based on the provided info."
            else:
                response = self.client.ChatCompletion.create(
                    model=model_name,
                    messages=enhanced_messages,
                    temperature=temperature,
                    max_tokens=300
                )
                choices = response.get("choices", [])
                answer = choices[0]["message"]["content"].strip() if choices else "I don't know based on the provided info."

            citations = [s["id"] for s in context_snippets[:3]]
            return {
                "answer": answer.strip() if isinstance(answer, str) else str(answer),
                "citations": citations
            }

        except Exception as e:
            print(f"❌ Error getting completion: {e}")
            return self._get_fallback_completion(messages, mode, context_snippets)

    def _get_fallback_completion(self, messages, mode, context_snippets):
        """Fallback to mock completion"""
        snippet_ids = [s.get("id") for s in context_snippets]
        answer = "Fallback: I used the provided context snippets."
        citations = snippet_ids[:3]
        return {"answer": answer, "citations": citations}
