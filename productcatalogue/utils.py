import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from django.db import transaction
from typing import List, Dict

def chunk_faq_markdown(md_text: str, approx_k=1200):
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

def _coerce_vector_to_list(vec):
    """
    Ensure embedding vector is a plain Python list of floats (JSON serializable).
    Accepts list, tuple, numpy array.
    """
    try:
        arr = np.array(vec, dtype=float).reshape(-1)
        return [float(x) for x in arr.tolist()]
    except Exception:
        try:
            return [float(x) for x in list(vec)]
        except Exception:
            return []

def store_product_and_embeddings(products: List[Dict], vectors: List[List[float]]):
    """
    Store product info + embedding vectors in the database.
    Uses the SAME detailed text format as used during embedding generation
    to ensure consistency between embedding and retrieved context.
    """
    from .models import Product, EmbeddingVector
    with transaction.atomic():
        for prod, vec in zip(products, vectors):
            p, _ = Product.objects.update_or_create(
                id=prod['id'],
                defaults={
                    'name': prod.get('name',''),
                    'notes': prod.get('notes',''),
                    'accords': prod.get('accords',''),
                    'price': prod.get('price') or None,
                    'longevity': prod.get('longevity',''),
                    'season': prod.get('season',''),
                    'image_url': prod.get('imageUrl',''),
                    'popularity': prod.get('popularity') or 0.0,
                }
            )

            price_display = f"${p.price:.2f}" if p.price is not None and p.price > 0 else "Price not available"
            stored_text = (
                f"Product name: {p.name}. "
                f"Description: {p.notes}. "
                f"Price: {price_display}. "
                f"Features/Accords: {p.accords}. "
                f"Longevity: {p.longevity}. "
                f"Recommended season: {p.season}."
            )

            clean_vec = _coerce_vector_to_list(vec)
            ev_id = f"p_{p.id}"
            EmbeddingVector.objects.update_or_create(
                id=ev_id,
                defaults={
                    'source': 'product',
                    'source_obj_id': p.id,
                    'text': stored_text,
                    'vector': json.dumps(clean_vec)
                }
            )

def store_faq_chunks_and_embeddings(chunks: List[Dict], vectors: List[List[float]]):
    """
    Store FAQ chunks + embedding vectors in the database.
    """
    from .models import FAQChunk, EmbeddingVector
    with transaction.atomic():
        for chunk, vec in zip(chunks, vectors):
            fid = chunk.get('id')
            FAQChunk.objects.update_or_create(
                id=fid,
                defaults={
                    'heading': chunk.get('heading',''),
                    'text': chunk.get('text','')
                }
            )
            clean_vec = _coerce_vector_to_list(vec)
            EmbeddingVector.objects.update_or_create(
                id=f"f_{fid}",
                defaults={
                    'source': 'faq',
                    'source_obj_id': fid,
                    'text': chunk.get('text',''),
                    'vector': json.dumps(clean_vec)
                }
            )

def load_all_vectors():
    """
    Returns list of dicts: {id, source, source_obj_id, text, vector(np.array)}
    """
    from .models import EmbeddingVector
    items = []
    for ev in EmbeddingVector.objects.all():
        try:
            vec_list = json.loads(ev.vector)
            vec = np.array(vec_list, dtype=float)
        except Exception:
            try:
                vec = np.array(ev.vector, dtype=float)
            except Exception:
                continue
        items.append({
            'id': ev.id,
            'source': ev.source,
            'source_obj_id': ev.source_obj_id,
            'text': ev.text,
            'vector': vec
        })
    return items

def retrieve_top_k(query_vector, k=8, threshold=0.35):
    """
    Returns top_k embeddings above similarity threshold.
    """
    items = load_all_vectors()
    if not items:
        return []

    mats = np.vstack([it['vector'] for it in items])
    qv = np.array(query_vector).reshape(1, -1)
    cos = cosine_similarity(qv, mats).flatten()

    filtered_idx = [i for i, s in enumerate(cos) if s >= threshold]
    if not filtered_idx:
        return []

    sorted_idx = sorted(filtered_idx, key=lambda i: -cos[i])[:k]
    results = []
    for idx in sorted_idx:
        it = items[idx]
        results.append({
            'id': it['id'],
            'source': it['source'],
            'source_obj_id': it['source_obj_id'],
            'text': it['text'],
            'score': float(cos[idx])
        })
    return results
