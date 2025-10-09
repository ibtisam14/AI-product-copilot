import json
import hashlib
from django.conf import settings


# ----------------------------
# 🧩 Mock Adapter (for local/demo mode)
# ----------------------------
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

        # Simple comparison logic
        if 'compare' in messages[-1]["content"].lower() and len(product_ids) >= 2:
            answer = f"Mock compare: {product_ids[0]} seems stronger than {product_ids[1]}."
            citations = product_ids[:2]
        else:
            citations = snippet_ids[:3]

        return {"answer": answer, "citations": citations}


# ----------------------------
# 🤖 OpenAI Adapter (real API mode)
# ----------------------------
class OpenAIAdapter:
    def __init__(self, api_key):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def get_embeddings(self, texts):
        """Robust embedding getter compatible with OpenAI & OpenRouter"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )

            # 🧠 Handle all possible response shapes
            if isinstance(response, str):
                # JSON string
                data = json.loads(response)
                embeddings = [item["embedding"] for item in data.get("data", [])]
            elif hasattr(response, "data"):
                # OpenAI SDK object
                embeddings = [item.embedding for item in response.data]
            elif isinstance(response, dict):
                # Already a dict from some wrapper
                embeddings = [item["embedding"] for item in response.get("data", [])]
            else:
                print(f"⚠️ Unknown response type: {type(response)}")
                embeddings = []

            # ✅ Guarantee non-empty embeddings (fallback to deterministic mock)
            if not embeddings:
                print("⚠️ Empty embedding list — using fallback vectors.")
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
        """Chat completion handler"""
        if mode == "fast":
            model_name = "gpt-3.5-turbo"
            temperature = 0.7
        else:
            model_name = "gpt-3.5-turbo"
            temperature = 0.2

        base_url = getattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1")
        if "openrouter.ai" in base_url:
            model_name = "openai/gpt-3.5-turbo"

        # Prepare system + context
        context_text = "\n\n".join([f"[{s['id']}] {s['text']}" for s in context_snippets])
        system_prompt = (
            "You are a helpful assistant. "
            "Answer using ONLY the provided context below. "
            "If the answer is not found, say 'I don't know based on the provided info.'\n\n"
            f"Context:\n{context_text}"
        )

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=full_messages,
                temperature=temperature,
                max_tokens=500
            )

            # Parse both OpenAI and JSON-string responses
            if isinstance(response, str):
                data = json.loads(response)
                answer = data["choices"][0]["message"]["content"].strip()
            else:
                answer = response.choices[0].message.content.strip()

            citations = [s["id"] for s in context_snippets[:3]]
            return {"answer": answer, "citations": citations}

        except Exception as e:
            print(f"❌ Error getting completion: {e}")
            return self._get_fallback_completion(messages, mode, context_snippets)

    def _get_fallback_completion(self, messages, mode, context_snippets):
        """Fallback to mock completion"""
        snippet_ids = [s.get("id") for s in context_snippets]
        answer = "Mock fallback: I used provided snippets."
        citations = snippet_ids[:3]
        return {"answer": answer, "citations": citations}
