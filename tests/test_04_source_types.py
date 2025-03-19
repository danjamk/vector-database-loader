import json
import os
import unittest
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings

from vector_database_loader.pinecone_vector_db import PineconeVectorLoader
from vector_database_loader.pinecone_vector_db import PineconeVectorQuery
os.environ["TOKENIZERS_PARALLELISM"] = "false"

full_website_w_blacklist_content_source = {
    "name": "Superlinked",
    "type": "Website",
    "location": "https://superlinked.com/vectorhub/sitemap.xml",
    "chunk_size": 512,
    "blacklist": [
        "https://superlinked.com/vectorhub/articles/*",
        "https://superlinked.com/vectorhub/tags/*"
    ]
}

full_website_w_whitelist_content_source = {
    "name": "Superlinked",
    "type": "Website",
    "location": "https://superlinked.com/vectorhub/sitemap.xml",
    "chunk_size": 512,
    "whitelist": [
        "https://superlinked.com/vectorhub/articles/*"
    ]
}


single_web_page_content_source =  {
    "name": "SpaceX",
    "type": "Website",
    "items": ["https://en.wikipedia.org/wiki/SpaceX" ],
    "chunk_size": 1024
}

local_pdf_files_content_source = {
    "name": "DanJamK",
    "type": "PDF",
    "location": "doc_folder",
    "chunk_size": 512
}

local_word_files_content_source = {
    "name": "Word Docs",
    "type": "Microsoft Word",
    "location": "doc_folder",
    "chunk_size": 512
}


web_pdf_content_source = {
    "name": "Jetson Orin Nano",
    "type": "Web PDFs",
    "location": "pdf_downloads",
    "chunk_size": 512,
    "items": [
        {
            "filename": "jetson-orin-nano-developer-kit-datasheet.pdf",
            "url": "https://files.seeedstudio.com/wiki/Jetson-Orin-Nano-DevKit/jetson-orin-nano-developer-kit-datasheet.pdf"
        }
    ]
}

google_drive_content_source = {"name": "Test Google Drive",
                  "type": "Google Drive",
                  "location": "1chQNbpiaozntsP5obMnSeBl1DgW8EBd9"
                  }

load_dotenv(find_dotenv())


class SourceTypeTestCases(unittest.TestCase):
    def test_load_source_types_pinecone(self):
        embedding_client = OpenAIEmbeddings()

        content_sources = [
            full_website_w_blacklist_content_source,
            full_website_w_whitelist_content_source,
            single_web_page_content_source,
            local_pdf_files_content_source,
            local_word_files_content_source,
            web_pdf_content_source,
            google_drive_content_source
        ]

        index_name = "test-sources-index-loader"

        vector_db_loader = PineconeVectorLoader(
            index_name=index_name,
            embedding_client=embedding_client)

        doc_count = vector_db_loader.load_sources(content_sources, delete_index=True)
        print(f"Loaded {doc_count} documents into {index_name}")
        self.assertTrue(doc_count > 0)

        index_info = vector_db_loader.describe_index()
        print(json.dumps(index_info, indent=2))

        # simple test query
        query_string = "what is a good use case for the Jetson Orin Nano dev kit?"
        vector_db_query = PineconeVectorQuery(index_name=index_name,
                                               embedding_client=embedding_client)
        query_results = vector_db_query.query(query_string)
        print(f"Query: {query_string} returned {len(query_results)} results")
        print(f"First result: {query_results[0].metadata['title']}")
        print(f"First result: {query_results[0].page_content}")

        vector_db_loader.delete_index() # clean-up




if __name__ == '__main__':
    unittest.main()