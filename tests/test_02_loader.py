import unittest

from dotenv import load_dotenv, find_dotenv

from vector_database_loader.pinecone_vector_db import PineconeVectorLoader
from vector_database_loader.milvus_vector_db import MilvusVectorLoader
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

web_page_content_source =  {"name": "SpaceX", "type": "Website", "items": [
        "https://en.wikipedia.org/wiki/SpaceX"

    ], "chunk_size": 512}

load_dotenv(find_dotenv())

class PineconeLoaderTestCases(unittest.TestCase):
    def test_load_webpage_openai_embedding(self):
        #embedding_model = EmbeddingModels.OPENAI_GPT_V3LARGE
        embedding_client = OpenAIEmbeddings()

        content_sources = [web_page_content_source]
        index_name = "test-webpage-index-loader"

        # # assert that I have an embedding model with gpt in the text
        # self.assertTrue('gpt' in embedding_model['name'])

        vector_db = PineconeVectorLoader(index_name=index_name,
                                         embedding_client=embedding_client)
        doc_count = vector_db.load_sources(content_sources, delete_index=True)
        print(f"Loaded {doc_count} documents into {index_name}")
        self.assertTrue(doc_count > 0)

    def test_load_webpage_huggingface_local_embedding(self):
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        model_name = "BAAI/bge-large-en"  # 1024
        embedding_client = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        content_sources = [web_page_content_source]
        index_name = "test-webpage-index-loader"

        vector_db = PineconeVectorLoader(index_name=index_name,
                                         embedding_client=embedding_client)
        doc_count = vector_db.load_sources(content_sources, delete_index=True)
        print(f"Loaded {doc_count} documents into {index_name}")
        self.assertTrue(doc_count > 0)


class MilvusLoaderTestCases(unittest.TestCase):
    def test_load_webpage_openai_embedding(self):
        # embedding_client = OpenAIEmbeddings()
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        model_name = "BAAI/bge-large-en"  # 1024
        embedding_client = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        content_sources = [web_page_content_source]
        index_name = "test_webpage_index_loader"

        # # assert that I have an embedding model with gpt in the text
        # self.assertTrue('gpt' in embedding_model['name'])

        vector_db = MilvusVectorLoader(index_name=index_name,
                                         embedding_client=embedding_client)
        doc_count = vector_db.load_sources(content_sources, delete_index=True)
        print(f"Loaded {doc_count} documents into {index_name}")
        self.assertTrue(doc_count > 0)

        index_info = vector_db.describe_index()
        print(index_info)
        self.assertTrue(index_name is not None)



if __name__ == '__main__':
    unittest.main()
