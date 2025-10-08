from django.urls import path
from django.views.decorators.csrf import csrf_exempt  
from .views import UploadIngestView, EmbeddingsView, ChatView, home

urlpatterns = [
    path('', home, name='home'),
    # Wrap each API view with csrf_exempt:
    path('upload/', csrf_exempt(UploadIngestView.as_view()), name='upload'),
    path('embeddings/', csrf_exempt(EmbeddingsView.as_view()), name='embeddings'),
    path('chat/', csrf_exempt(ChatView.as_view()), name='chat'),
]