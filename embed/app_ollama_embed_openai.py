from langchain_openai import OpenAIEmbeddings

# 1. Create an instance of the embedding model (not just a reference to the class)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")  
# you can also use "text-embedding-3-large" for higher dimensionality

# 2. Embed multiple documents (expects a list of strings, not list of lists)
documents = ["Hi there", "Oh, Hello", "What's your name", "My friends call me folly", "Hello world"]
embeddings = embedding_model.embed_documents(documents)

# embeddings is now a list of vectors, one for each document
print("Number of vectors:", len(embeddings))       # should equal number of documents
print("Vector length:", len(embeddings[0]))        # embedding dimension (e.g. 1536 or 3072)

# 3. Embed a single query string
query = "Tell me what is AI"
embedding_query = embedding_model.embed_query(query)

# embedding_query is a single vector
print("Query vector length:", len(embedding_query))
print("First 5 dimensions of query vector:", embedding_query[:5])
