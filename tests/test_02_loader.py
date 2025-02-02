import unittest

from dotenv import load_dotenv, find_dotenv

from vector_database_loader.pinecone_vectory_db import PineconeVectorLoader
from langchain_openai import OpenAIEmbeddings

web_page_content_source =  {"name": "SpaceX", "type": "Website", "items": [
        "https://en.wikipedia.org/wiki/SpaceX"

    ], "chunk_size": 512}

load_dotenv(find_dotenv())

class LoaderTestCases(unittest.TestCase):
    def test_load_webpage(self):
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


if __name__ == '__main__':
    unittest.main()
