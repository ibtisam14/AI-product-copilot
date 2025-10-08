import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import EmbeddingVector, Product, FAQChunk
from django.db import transaction
from typing import List, Dict

def chunk_faq_markdown(md_text: str, approx_k=700):
    """
    Split markdown into ~approx_k char chunks by paragraphs/headers.
    Returns list of (heading, chunk_text).
    """
    lines = md_text.splitlines()
    chunks = []
    cur = []
    cur_heading = ""
    for line in lines:
        if line.strip().startswith("##"):
            # flush
            if cur:
                chunks.append((cur_heading, "\n".join(cur).strip()))
                cur = []
            cur_heading = line.strip().lstrip("#").strip()
        else:
            cur.append(line)
            if len(" ".join(cur)) > approx_k:
                chunks.append((cur_heading, "\n".join(cur).strip()))
                cur = []
    if cur:
        chunks.append((cur_heading, "\n".join(cur).strip()))
    return chunks

def store_product_and_embeddings(products: List[Dict], vectors: List[List[float]], adapter_name="server"):
    """
    products: list of dict parsed from CSV (each with id, name, notes, accords...)
    vectors aligned one-per-product chunk (for simplicity we embed `name + notes`)
    """
    from .models import Product, EmbeddingVector
    import json
    with transaction.atomic():
        for prod, vec in zip(products, vectors):
            p, _ = Product.objects.update_or_create(id=prod['id'], defaults={
                'name': prod.get('name',''),
                'notes': prod.get('notes',''),
                'accords': prod.get('accords',''),
                'price': prod.get('price') or None,
                'longevity': prod.get('longevity',''),
                'season': prod.get('season',''),
                'image_url': prod.get('imageUrl',''),
                'popularity': prod.get('popularity') or 0.0,
            })
            ev_id = f"p_{p.id}"
            EmbeddingVector.objects.update_or_create(id=ev_id, defaults={
                'source':'product',
                'source_obj_id': p.id,
                'text': f"{p.name}. {p.notes}",
                'vector': json.dumps(vec)
            })

def store_faq_chunks_and_embeddings(chunks: List[Dict], vectors: List[List[float]]):
    from .models import FAQChunk, EmbeddingVector
    import json
    with transaction.atomic():
        for chunk, vec in zip(chunks, vectors):
            fid = chunk.get('id')
            FAQChunk.objects.update_or_create(id=fid, defaults={
                'heading': chunk.get('heading',''),
                'text': chunk.get('text','')
            })
            EmbeddingVector.objects.update_or_create(id=f"f_{fid}", defaults={
                'source':'faq',
                'source_obj_id': fid,
                'text': chunk.get('text',''),
                'vector': json.dumps(vec)
            })

def load_all_vectors():
    """
    returns list of dicts: {id, source, source_obj_id, text, vector(np.array)}
    """
    import json
    items = []
    for ev in EmbeddingVector.objects.all():
        vec = np.array(json.loads(ev.vector), dtype=float)
        items.append({
            'id': ev.id,
            'source': ev.source,
            'source_obj_id': ev.source_obj_id,
            'text': ev.text,
            'vector': vec
        })
    return items

def retrieve_top_k(query_vector, k=8, alpha_bm25=0.2):
    """
    Combine cosine similarity on embeddings + TF-IDF similarity score (as BM25 approximation).
    Returns top_k items with combined score and sorted.
    """
    items = load_all_vectors()
    if not items:
        return []
    texts = [it['text'] for it in items]
    # TF-IDF vectorizer
    tfidf = TfidfVectorizer().fit(texts + [" ".join(map(str, query_vector))])
    text_vecs = tfidf.transform(texts)
    # approximate text-based similarity by transforming query into the same tf-idf space
    # (we don't have the raw query text here normally; this function expects caller to pass query_text for BM25)
    # For simplicity, compute cosine similarity on embedding vectors:
    mats = np.vstack([it['vector'] for it in items])
    qv = np.array(query_vector).reshape(1, -1)
    cos = cosine_similarity(qv, mats).flatten()  # shape (n,)
    # If text-based scoring needed, caller can compute and pass; using cos only here
    combined = cos  # placeholder for combination
    ranked_idx = np.argsort(-combined)[:k]
    results = []
    for idx in ranked_idx:
        it = items[idx]
        results.append({
            'id': it['id'],
            'source': it['source'],
            'source_obj_id': it['source_obj_id'],
            'text': it['text'],
            'score': float(combined[idx])
        })
    return results
