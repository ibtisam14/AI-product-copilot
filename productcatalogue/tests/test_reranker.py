import pytest
from productcatalogue.adapters import MockAdapter
from productcatalogue.utils import chunk_faq_markdown, store_product_and_embeddings, store_faq_chunks_and_embeddings, retrieve_top_k
from productcatalogue.models import Product, FAQChunk, EmbeddingVector
from django.test import TestCase

class RerankerTest(TestCase):
    def setUp(self):
        self.adapter = MockAdapter()
        # create two demo products and embeddings
        prods = [
            {'id':'12','name':'Rise Again','notes':'citrus-woody longevity 8-10h','accords':'citrus,woody','price':45,'longevity':'8-10h','season':'all','imageUrl':'','popularity':1.0},
            {'id':'07','name':'Lost Words','notes':'fresh-woody longevity 6-8h','accords':'fresh,woody','price':30,'longevity':'6-8h','season':'all','imageUrl':'','popularity':0.8},
        ]
        texts = [f"{p['name']}. {p['notes']}" for p in prods]
        vectors = self.adapter.get_embeddings(texts)
        store_product_and_embeddings(prods, vectors)
        # create a faq chunk
        faq = [{'id':'02','heading':'longevity','text':'Longevity depends on skin chemistry; EDP usually lasts longer.'}]
        fvecs = self.adapter.get_embeddings([faq[0]['text']])
        store_faq_chunks_and_embeddings(faq, fvecs)

    def test_retrieve_compare(self):
        q = "Compare Rise Again and Lost Words for longevity."
        qvec = self.adapter.get_embeddings([q])[0]
        top = retrieve_top_k(qvec, k=3)
        # Expect some items (non-empty) and product ids present
        assert len(top) >= 2
        # top should include product items with ids starting with p_
        ids = [t['id'] for t in top]
        assert any(i.startswith('p_') for i in ids)

    def tearDown(self):
        Product.objects.all().delete()
        FAQChunk.objects.all().delete()
        EmbeddingVector.objects.all().delete()
