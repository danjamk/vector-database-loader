import unittest
import time

from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings

from vector_database_loader.pinecone_vectory_db import PineconeVectorQuery

load_dotenv(find_dotenv())

class TestVDBQuery(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
