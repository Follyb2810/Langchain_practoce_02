from langchain_ollama import OllamaEmbeddings

# 1. Create an instance of the Ollama embeddings model
#    Make sure you've pulled the embedding model locally, e.g.:
#    ollama pull nomic-embed-text
embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# 2. Embed multiple documents (expects a list of strings)
documents = ["Hi there", "Oh, Hello", "What's your name", "My friends call me folly", "Hello world"]
embeddings = embedding_model.embed_documents(documents)

# embeddings is now a list of vectors, one for each document
print("Number of vectors:", len(embeddings))       # should equal number of documents
print("Vector length:", len(embeddings[0]))        # embedding dimension (depends on model, e.g. 768 or 1024)

# 3. Embed a single query string
query = "Tell me what is AI"
embedding_query = embedding_model.embed_query(query)

# embedding_query is a single vector
print("Query vector length:", len(embedding_query))
print("First 5 dimensions of query vector:", embedding_query[:5])
