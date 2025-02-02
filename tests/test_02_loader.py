import unittest
import os
from vector_database_loader.pinecone_vectory_db import PineconeVectorLoader
from vector_database_loader.base_vector_db import EmbeddingModels


web_page_content_source =  {"name": "SpaceX", "type": "Website", "items": [
        "https://en.wikipedia.org/wiki/SpaceX"

    ], "chunk_size": 512}


class LoaderTestCases(unittest.TestCase):
    def test_load_webpage(self):
        embedding_model = EmbeddingModels.OPENAI_GPT_V3LARGE
        content_sources = [web_page_content_source]
        index_name = "test-webpage-index-loader"

        # assert that I have an embedding model with gpt in the text
        self.assertTrue('gpt' in embedding_model['name'])

        vector_db = PineconeVectorLoader(index_name=index_name,
                                         embedding_model=embedding_model)
        vector_db.load_sources(content_sources, delete_index=True)


if __name__ == '__main__':
    unittest.main()
