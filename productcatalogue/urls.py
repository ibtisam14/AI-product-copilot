from django.urls import path
from django.views.decorators.csrf import csrf_exempt  
from .views import UploadIngestView, EmbeddingsView, ChatView, home, upload_page

urlpatterns = [
    path('', home, name='home'),
    path('upload-data/', upload_page, name='upload_page'),
    path('upload/', csrf_exempt(UploadIngestView.as_view()), name='upload'),
    path('embeddings/', csrf_exempt(EmbeddingsView.as_view()), name='embeddings'),
    path('chat/', csrf_exempt(ChatView.as_view()), name='chat'),
]