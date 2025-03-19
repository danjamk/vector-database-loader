import unittest
import time

from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from vector_database_loader.pinecone_vector_db import PineconeVectorQuery
from vector_database_loader.milvus_vector_db import MilvusVectorQuery

load_dotenv(find_dotenv())

class TestPineconeVDBQuery(unittest.TestCase):
    def setUp(self):
        # This is needed because there is a latenecy in the data being available
        print("Waiting 30 seconds before running the tests")
        time.sleep(30)

    def test_query_webpage_index(self):
        embedding_client = OpenAIEmbeddings()
        index_name = "test-webpage-index-loader"
        vector_db = PineconeVectorQuery(index_name=index_name,
                                        embedding_client=embedding_client)
        query = "What is SpaceX's most recent rocket model being tested?"
        documents = vector_db.query(query)
        self.assertTrue(documents is not None)
        self.assertTrue(len(documents) > 0)
        print(f"Query: {query} returned {len(documents)} results")
        for doc in documents:
            print(f"   {doc.metadata['title']}")
            print(f"   {doc.page_content}")


# class TestMilvusVDBQuery(unittest.TestCase):
#
#     def test_query_webpage_index(self):
#         embedding_client = OpenAIEmbeddings()
#         index_name = "test_webpage_index_loader"
#         vector_db = MilvusVectorQuery(index_name=index_name,
#                                         embedding_client=embedding_client)
#         query = "What is SpaceX's most recent rocket model being tested?"
#         documents = vector_db.query(query)
#         self.assertTrue(documents is not None)
#         self.assertTrue(len(documents) > 0)
#         print(f"Query: {query} returned {len(documents)} results")
#         for doc in documents:
#             print(doc)
#             print(f"   {doc.metadata['title']}")
#             print(f"   {doc.page_content}")



if __name__ == '__main__':
    unittest.main()
