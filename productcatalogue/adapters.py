import os
import json
from typing import List, Dict

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
class OpenAIAdapter(BaseAdapter):
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def get_embeddings(self, texts: List[str]):
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [data.embedding for data in response.data]

    def get_completion(self, messages, mode, context_snippets):
        # Pick model based on mode
        model_name = "gpt-3.5-turbo" if mode == "fast" else "gpt-4-turbo"

        # Combine FAQ/product text
        context_text = "\n\n".join([f"[{s['id']}] {s['text']}" for s in context_snippets])

        # Create the system + user message for the chat
        system_message = {
            "role": "system",
            "content": "You are a helpful product assistant. Always base your answers on the context below."
        }
        user_message = {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion:\n{messages[-1]['content']}"
        }

        # Call OpenAI Chat API
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[system_message, user_message],
            temperature=0.5,
            max_tokens=300,
        )

        answer = completion.choices[0].message.content.strip()
        citations = [s['id'] for s in context_snippets[:3]]

        return {"answer": answer, "citations": citations}