import os
import json
from typing import List, Dict
from django.conf import settings


# Adapter interface: must implement get_completion(messages, mode, context_snippets) and get_embeddings(texts)
class BaseAdapter:
    def get_completion(self, messages: List[Dict], mode: str, context_snippets: List[Dict]):
        raise NotImplementedError

    def get_embeddings(self, texts: List[str]):
        raise NotImplementedError

# Deterministic mock adapter (required)
class MockAdapter(BaseAdapter):
    def __init__(self):
        pass

    def _deterministic_answer(self, prompt):
        # simple deterministic rule: if "compare" in prompt and two product names are in context snippets,
        # create a short comparative answer and cite the two product ids.
        # This ensures reproducible behavior for tests/demo.
        snippet_ids = [s.get('id') for s in prompt.get('context_snippets', [])]
        product_ids = [s['id'] for s in prompt.get('context_snippets', []) if s.get('source') == 'product']
        answer = "Mock answer: I used the provided snippets to answer."
        if 'compare' in prompt.get('messages', [{}])[-1].get('content','').lower() and len(product_ids) >= 2:
            answer = f"Mock compare: {product_ids[0]} appears stronger than {product_ids[1]} in the context provided."
            citations = product_ids[:2]
        else:
            citations = snippet_ids[:3]
        return {"answer": answer, "citations": citations}

    def get_completion(self, messages, mode, context_snippets):
        # messages: list of {"role","content"}
        prompt = {"messages": messages, "mode": mode, "context_snippets": context_snippets}
        return self._deterministic_answer(prompt)

    def get_embeddings(self, texts):
        # deterministic pseudo-embeddings: stable hashing -> small vector
        import hashlib
        import numpy as np
        vectors = []
        for t in texts:
            h = hashlib.sha256(t.encode('utf-8')).digest()
            # turn first 16 bytes into floats
            vec = [((b % 128) / 127.0) * (1 if i % 2 == 0 else -1) for i, b in enumerate(h[:16])]
            vectors.append(vec)
        return vectors

# Optional: OpenAI adapter — uses OPENAI_API_KEY from env
# Optional: OpenAI adapter — uses OPENAI_API_KEY from env
class OpenAIAdapter(BaseAdapter):
    def __init__(self, api_key: str):
        from openai import OpenAI
        base_url = getattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_embeddings(self, texts: List[str]):
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            # Extract embeddings from the response - this is the key fix
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"❌ Error getting embeddings: {str(e)}")
            # Fallback to mock embeddings if API call fails
            return self._get_fallback_embeddings(texts)

    def _get_fallback_embeddings(self, texts):
        """Fallback to deterministic mock embeddings if API fails"""
        import hashlib
        vectors = []
        for t in texts:
            h = hashlib.sha256(t.encode('utf-8')).digest()
            vec = [((b % 128) / 127.0) * (1 if i % 2 == 0 else -1) for i, b in enumerate(h[:16])]
            vectors.append(vec)
        return vectors

    def get_completion(self, messages, mode, context_snippets):
        # Set model and temperature based on mode
        if mode == "fast":
            model_name = "gpt-3.5-turbo"
            temperature = 0.7
        else:  # accurate
            model_name = "gpt-3.5-turbo"
            temperature = 0.2

        # If using OpenRouter, you might need to prefix the model name
        base_url = getattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1")
        if "openrouter.ai" in base_url:
            model_name = "openai/gpt-3.5-turbo"  # OpenRouter requires provider prefix

        # Build context
        context_text = "\n\n".join([f"[{s['id']}] {s['text']}" for s in context_snippets])

        # System message with context
        system_prompt = (
            "You are a helpful fragrance assistant. "
            "Answer using ONLY the context below. "
            "If the answer isn't in the context, say: 'I don't know based on the provided info.'\n\n"
            f"Context:\n{context_text}"
        )

        # Full message history
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            # Call OpenAI
            response = self.client.chat.completions.create(
                model=model_name,
                messages=full_messages,
                temperature=temperature,
                max_tokens=500,
            )

            answer = response.choices[0].message.content.strip()
            citations = [s["id"] for s in context_snippets[:3]]
            return {"answer": answer, "citations": citations}
        except Exception as e:
            print(f"❌ Error getting completion: {str(e)}")
            # Fallback to mock response
            return self._get_fallback_completion(messages, mode, context_snippets)

    def _get_fallback_completion(self, messages, mode, context_snippets):
        """Fallback to mock completion if API fails"""
        # Reuse your existing mock logic
        prompt = {"messages": messages, "mode": mode, "context_snippets": context_snippets}
        return self._deterministic_answer(prompt)

    def _deterministic_answer(self, prompt):
        """Mock response generator for fallback - reused from MockAdapter"""
        snippet_ids = [s.get('id') for s in prompt.get('context_snippets', [])]
        product_ids = [s['id'] for s in prompt.get('context_snippets', []) if s.get('source') == 'product']
        answer = "Mock answer: I used the provided snippets to answer."
        if 'compare' in prompt.get('messages', [{}])[-1].get('content','').lower() and len(product_ids) >= 2:
            answer = f"Mock compare: {product_ids[0]} appears stronger than {product_ids[1]} in the context provided."
            citations = product_ids[:2]
        else:
            citations = snippet_ids[:3]
        return {"answer": answer, "citations": citations}