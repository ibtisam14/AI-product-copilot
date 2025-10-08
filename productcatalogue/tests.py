from django.urls import path
from .views import UploadIngestView, EmbeddingsView, ChatView

urlpatterns = [
    path('upload/', UploadIngestView.as_view(), name='upload'),
    path('embeddings/', EmbeddingsView.as_view(), name='embeddings'),
    path('chat/', ChatView.as_view(), name='chat'),
]
