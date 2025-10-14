import chromadb

# Create a persistent Chroma database folder named "chroma_db"
client = chromadb.PersistentClient(path="chroma_db")

# Create or get the collection where chat data will be saved
collection = client.get_or_create_collection("chatbot_data")