import time

from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings

from vector_database_loader.pinecone_vector_db import PineconeVectorLoader, PineconeVectorQuery


# Define your content sources and add them to the array
web_page_content_source =  {"name": "SpaceX", "type": "Website", "items": [
        "https://en.wikipedia.org/wiki/SpaceX"

    ], "chunk_size": 512}
content_sources = [web_page_content_source]


# Load into your vector database.  Be sure to add your Pinecone and OpenAI API keys to your .env file
load_dotenv(find_dotenv())
embedding_client = OpenAIEmbeddings()
index_name = "my-vectordb-index"
vector_db_loader = PineconeVectorLoader(index_name=index_name,
                                 embedding_client=embedding_client)
vector_db_loader.load_sources(content_sources, delete_index=True)


# Query your vector database
print("Waiting 30 seconds before running the query, to make sure the data is available")
time.sleep(30)  # This is needed because there is a latency in the data being available
vector_db_query = PineconeVectorQuery(index_name=index_name,
                                embedding_client=embedding_client)
query = "What is SpaceX's most recent rocket model being tested?"
documents = vector_db_query.query(query)
print(f"Query: {query} returned {len(documents)} results")
for doc in documents:
    print(f"   {doc.metadata['title']}")
    print(f"   {doc.page_content}")