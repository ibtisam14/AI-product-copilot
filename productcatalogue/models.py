from django.db import models
import uuid
from django.utils import timezone

class Product(models.Model):
    id = models.CharField(max_length=100, primary_key=True)  # keep CSV id
    name = models.CharField(max_length=255)
    notes = models.TextField(blank=True)
    accords = models.CharField(max_length=255, blank=True)
    price = models.FloatField(null=True, blank=True)
    longevity = models.CharField(max_length=100, blank=True)
    season = models.CharField(max_length=100, blank=True)
    image_url = models.URLField(blank=True)
    popularity = models.FloatField(default=0.0)

    def __str__(self):
        return self.name

class FAQChunk(models.Model):
    id = models.CharField(max_length=100, primary_key=True)
    heading = models.CharField(max_length=255, blank=True)
    text = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.heading[:50]}"

class EmbeddingVector(models.Model):
    """
    Store embeddings for both product text chunks and faq chunks.
    vector stored as bytes (numpy) or JSON string of floats — here we'll store as text JSON.
    """
    SOURCE_CHOICES = (('product','product'), ('faq','faq'))
    id = models.CharField(max_length=120, primary_key=True)
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES)
    source_obj_id = models.CharField(max_length=100)  # product.id or faq.id
    text = models.TextField()
    vector = models.TextField() 
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.id} ({self.source})"
