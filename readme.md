AI Product FAQ Copilot

A smart, AI-powered shopping assistant that answers product-related questions using a brand's product catalog and FAQ knowledge base, providing accurate responses with citations and no hallucinations.

Features
- Product Catalog: Upload and process a product catalog from a CSV file (products.csv) with details like name, notes, price, longevity, and more.
- FAQ Knowledge Base: Upload Markdown FAQs (faq.md) for instant retrieval and querying.
- AI Chat with Citations: Ask questions (e.g., "Which perfume lasts longer?") and get answers grounded in your uploaded data, with citations to specific products or FAQ entries (e.g., p_1, f_faq_5).
- Secure AI Integration: Supports OpenAI for embeddings and completions, or a mock adapter for local development (no API key needed).
- File Upload & Processing: Upload products.csv and faq.md via a web interface to build the knowledge base.
- Django Backend: Handles file uploads, embeddings, and chat responses securely.
- Simple Frontend: HTML, CSS, and JavaScript for an intuitive user interface.

Tech Stack
- Frontend: HTML, CSS, JavaScript (vanilla, no frameworks)
- Backend: Django 5.2.3, Django REST Framework
- AI Integration: OpenAI API (optional) or Mock Adapter for local development
- Database: SQLite (default, configurable for other databases)
- Dependencies: django, djangorestframework, openai, django-cors-headers (see requirements.txt)
- Tooling: Python virtual environment, Git

Setup & Usage

1. Clone the Repository
   git clone https://github.com/ibtisam14/AI-product-copilot.git
   cd AI-product-copilot

2. Set Up a Virtual Environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
   pip install -r requirements.txt

   If requirements.txt is missing, install core dependencies:
   pip install django djangorestframework openai django-cors-headers

   Update requirements.txt:
   pip freeze > requirements.txt

4. Apply Database Migrations
   python manage.py migrate

5. Run the Django Development Server
   python manage.py runserver

   Open http://localhost:8000 in your browser to access the app.

6. Configure OpenAI (Optional)
   For real AI mode (using OpenAI for embeddings and completions):
   - Add your OpenAI API key to copilot/settings.py:
     OPENAI_API_KEY = "your-openai-api-key"
   - Get a key from OpenAI's platform.
   - Without a key, the app uses a mock adapter, returning mock responses (e.g., "Mock answer...").

7. Upload Data
   - Open http://localhost:8000 in your browser.
   - Upload:
     - products.csv (columns: id, name, notes, accords, price, longevity, season, imageUrl, popularity)
     - faq.md (use ## Heading and paragraphs for FAQ entries)
   - Example products.csv:
     id,name,notes,accords,price,longevity,season,imageUrl,popularity
     1,Perfume A,Floral notes,Floral,99.99,Long-lasting,Spring,http://example.com/image.jpg,0.8
   - Example faq.md:
     ## What is this product?
     This is a product description.

8. Use the Chat Interface
   - After uploading files, ask questions in the chat input (e.g., "Which perfume is best for summer?").
   - Responses include answers and citations (e.g., p_1, f_faq_5) based on uploaded data.

API Endpoints

The Django backend provides the following endpoints:

- POST /api/upload/: Upload products.csv and faq.md files.
  {
    "products": <number of products>,
    "faq_chunks": <number of FAQ chunks>
  }

- POST /api/chat/: Send a chat query and receive an answer with citations.
  {
    "messages": [{"role": "user", "content": "Compare Perfume A and B"}],
    "mode": "fast"
  }
  Response:
  {
    "answer": "Perfume A is longer-lasting than B.",
    "citations": ["p_1", "f_faq_5"]
  }

- POST /api/embeddings/: Generate embeddings for text inputs (optional fallback).
  {
    "texts": ["text 1", "text 2"]
  }
  Response:
  {
    "vectors": [[0.1, -0.3, ...], ...]
  }

Troubleshooting

- Browser Error (CSRF): If POST requests fail with 403 Forbidden, add CSRF token to fetch requests in index.html:
  function getCookie(name) {
      let cookieValue = null;
      if (document.cookie && document.cookie !== '') {
          const cookies = document.cookie.split(';');
          for (let i = 0; i < cookies.length; i++) {
              const cookie = cookies[i].trim();
              if (cookie.substring(0, name.length + 1) === (name + '=')) {
                  cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                  break;
              }
          }
      }
      return cookieValue;
  }
  fetch('/api/chat/', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCookie('csrftoken')
      },
      body: JSON.stringify(payload)
  });

- CORS Issues: Ensure django-cors-headers is installed and configured in settings.py:
  INSTALLED_APPS = [... , 'corsheaders']
  MIDDLEWARE = ['corsheaders.middleware.CorsMiddleware', ...]
  CORS_ALLOWED_ORIGINS = ["http://localhost:8000", "http://127.0.0.1:8000"]

- No Data Uploaded: Verify products.csv and faq.md match expected formats.
- Mock Responses: If no OPENAI_API_KEY is set, the app uses MockAdapter, returning mock answers.
- Check Django logs (python manage.py runserver) and browser console (F12) for errors.

Project Structure

AI-product-copilot/
├── copilot/              # Django project settings and URLs
├── productcatalogue/      # Django app (models, views, adapters, utils)
├── templates/             # HTML templates (index.html)
├── requirements.txt       # Python dependencies
├── manage.py             # Django management script
├── db.sqlite3            # SQLite database
├── .gitignore            # Git ignore file

Contributing

1. Fork the repository.
2. Create a feature branch (git checkout -b feature/YourFeature).
3. Commit changes (git commit -m "Add YourFeature").
4. Push to the branch (git push origin feature/YourFeature).
5. Open a pull request.

License

MIT License. See LICENSE for details.

Contact

- GitHub: ibtisam14
- Email: Your contact email (optional)