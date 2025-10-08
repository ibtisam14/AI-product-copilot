from rest_framework import serializers
from .models import Product, FAQChunk, EmbeddingVector

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = '__all__'

class FAQChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = FAQChunk
        fields = '__all__'

class EmbeddingVectorSerializer(serializers.ModelSerializer):
    class Meta:
        model = EmbeddingVector
        fields = '__all__'
