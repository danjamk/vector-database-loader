# vector-database-loader
Loading content info a vector database is relatively easy to do, especially with frameworks like [LangChain](https://www.langchain.com/).
However, the process of curating the content and loading it into the database can be a bit more complex.  If you are building 
a RAG application or similar, the quality and relevance of the content is critical.  This project is meant to help with that process.

A use case for this type of project is discussed in more depth in the blog [A Cost-Effective AI Chatbot Architecture with AWS Bedrock, Lambda, and Pinecone](https://medium.com/@dan.jam.kuhn/a-cost-effective-ai-chatbot-architecture-with-aws-bedrock-lambda-and-pinecone-40935b9ec361)

## Features
- **Vector Database Support** - The framework is built to support multiple vector databases but is currently implementing support for [Pinecone](https://www.pinecone.io/).
More vector databases will be added, but if needed you can fork the project and handle your own needs by extending the base class.  
- **Embedding Support** - You can use any embedding provided by [Langchain](https://python.langchain.com/docs/integrations/text_embedding/), which includes OpenAI, AWS Bedrock, HuggingFace, Cohere and much, much more.
- **Content Curation** - The framework is built configure some common content types and sources, but again is meant to be extended a needed.
  - Sources include websites and local folders
  - Types include PDF, Word, and Web content

## Example
```python
import time

from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings

from vector_database_loader.pinecone_vectory_db import PineconeVectorLoader, PineconeVectorQuery


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
```

## Setup
Ensure you have Python 3.12 or later installed. Pyenv always great for this.
This project also using Poetry, so you will need to have that installed as well.  

1. Clone the repo
2. cd to the project directory `cd vector-database-loader`
3. Create a virtual environment `python -m venv .venv`
4. Activate the virtual environment:
    - On Windows: `.venv\Scripts\activate`
    - On macOS/Linux: `source .venv/bin/activate`
5. Run `poetry install` to install the dependencies
6. Create a .env file in the root directory.  See .evn.example for the required variables
7. Run the example: `python example.py`



## Roadmap
- Add support for more vector databases.  Shortlist is: [Milvus](https://milvus.io/) and [Weaviate](https://weaviate.io/)
https://python.langchain.com/docs/integrations/vectorstores/ 
- Add support for more sources.  Shortlist: Google Drive, AWS S3 folder.  
https://python.langchain.com/docs/integrations/document_loaders/#cloud-providers 


