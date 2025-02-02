# vector-database-loader
Loading content info a vector database is relatively easy to do, especially with frameworks like [LangChain](https://www.langchain.com/).
However, the process of curating the content and loading it into the database can be a bit more complex.  If you are building 
a RAG application or similar, the quality and relevance of the content is critical.  This project is meant to help with that process.



## Features
- **Vector Database Support** - The framework is built to support multiple vector databases but is currently implementing support for [Pinecone](https://www.pinecone.io/).
More vector databases will be added, but if needed you can fork the project and handle your own needs by extending the base class.  
- **Embedding Support** - The framework is built to support multiple embedding models but is currently implementing support for [OpenAI](https://platform.openai.com/docs/guides/embeddings) and [HuggingFace](https://huggingface.co/models?other=embeddings) local embeddings.
Again, forking the database and extending the base class will allow you to add your own embedding models.
- **Content Curation** - The framework is built configure some common content types and sources, but again is meant to be extended a needed.
 - Sources include websites and local folders
 - Types include PDF, Word, and Web content

## Example
```python
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
```

## Roadmap
- Add support for more vector databases.  Shortlist is: [Milvus](https://milvus.io/) and [Weaviate](https://weaviate.io/)
- Add support for more embedding model providers. Shortlist: [AWS Bedrock](https://aws.amazon.com/bedrock/), [Cohere](https://cohere.com/embed), [HuggingFace](https://huggingface.co/models?other=embeddings) remote models
- Simplify the process of extending embedding support by making models purely a parameter, not a definition
- Add support for more sources.  Shortlist: Google Drive, AWS S3 folder.  


## TODO
MVP:
- Do a clean test of project
- Make public
- Link to my chatbot architecture article

Post MVP
- create test cases for each type and load to vector
- publish to pypi and test - make sure it works
- Update readme to reflect new changes
- Create project download regression tester
- Create a pypi download regression tester
