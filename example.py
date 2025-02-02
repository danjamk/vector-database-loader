from vector_database_loader.pinecone_vectory_db import PineconeVectorLoader, PineconeVectorQuery
from vector_database_loader.base_vector_db import EmbeddingModels

# Define your content sources and add them to the array
web_page_content_source =  {"name": "SpaceX", "type": "Website", "items": [
        "https://en.wikipedia.org/wiki/SpaceX"

    ], "chunk_size": 512}

content_sources = [web_page_content_source]

# Load into your vector database.  Be sure to add your Pinecone and OpenAI API keys to your .env file
embedding_model = EmbeddingModels.OPENAI_GPT_V3LARGE
index_name = "my-vectordb-index"
vector_db_loader = PineconeVectorLoader(index_name=index_name,
                                 embedding_model=embedding_model)
vector_db_loader.load_sources(content_sources, delete_index=True)

# Query your vector database
vector_db_query = PineconeVectorQuery(index_name=index_name,
                                embedding_model=embedding_model)
query = "What is SpaceX's most recent rocket model being tested?"
documents = vector_db_query.query(query)
print(f"Query: {query} returned {len(documents)} results")
for doc in documents:
    print(f"   {doc.metadata['title']}")
    print(f"   {doc.page_content}")