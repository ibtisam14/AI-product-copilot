from django.contrib import admin
from .models import Product, FAQChunk, EmbeddingVector

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'price', 'popularity', 'longevity', 'season')
    list_filter = ('season', 'longevity')
    search_fields = ('name', 'notes', 'accords')
    list_editable = ('price', 'popularity')
    list_per_page = 20
    ordering = ('-popularity', 'name')
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'name', 'notes', 'accords')
        }),
        ('Pricing & Metrics', {
            'fields': ('price', 'popularity', 'longevity', 'season')
        }),
        ('Media', {
            'fields': ('image_url',)
        }),
    )

@admin.register(FAQChunk)
class FAQChunkAdmin(admin.ModelAdmin):
    list_display = ('id', 'heading_preview', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('heading', 'text')
    readonly_fields = ('created_at',)
    list_per_page = 20
    ordering = ('-created_at',)
    
    def heading_preview(self, obj):
        return obj.heading[:50] + "..." if len(obj.heading) > 50 else obj.heading
    heading_preview.short_description = 'Heading'

@admin.register(EmbeddingVector)
class EmbeddingVectorAdmin(admin.ModelAdmin):
    list_display = ('id', 'source', 'source_obj_id', 'created_at', 'text_preview')
    list_filter = ('source', 'created_at')
    search_fields = ('source_obj_id', 'text')
    readonly_fields = ('created_at',)
    list_per_page = 20
    ordering = ('-created_at',)
    
    def text_preview(self, obj):
        return obj.text[:100] + "..." if len(obj.text) > 100 else obj.text
    text_preview.short_description = 'Text Preview'