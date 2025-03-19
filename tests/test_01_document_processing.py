import unittest
import os
from dotenv import load_dotenv, find_dotenv
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vector_database_loader.document_processing_utils import (
    get_sitemap_urls,
    website_crawler,
    get_folder_documents,
    get_website_pdfs,
    blacklist_url_filter,
    url_whitelist,
    get_website_documents,
    get_google_drive_documents
)

load_dotenv(find_dotenv())

class MyTestCase(unittest.TestCase):
    def test_get_sitemap_urls(self):
        sitemap_url = "https://superlinked.com/vectorhub/sitemap.xml"
        urls = get_sitemap_urls(sitemap_url)
        self.assertTrue(len(urls) > 0)
        print(f"Found {len(urls)} urls in sitemap {sitemap_url}")
        for url in urls:
            print(f"   {url}")

    def test_website_crawler(self):
        sitemap_url = "https://superlinked.com/vectorhub/sitemap.xml"
        urls = get_sitemap_urls(sitemap_url)
        self.assertTrue(len(urls) > 0)

        docs = website_crawler(urls[:5])  # Only crawl the first 5 urls
        print(f"Found {len(docs)} documents")
        self.assertTrue(len(docs) > 0)

    def test_get_folder_documents(self):
        # Word Docs
        content_source = {"name": "Test Folder", "type": "Microsoft Word", "location": "doc_folder"}
        docs = get_folder_documents(content_source)
        print(f"Found {len(docs)} documents")
        self.assertTrue(len(docs) > 0)

        # PDFs
        content_source = {"name": "Test Folder", "type": "PDF", "location": "doc_folder"}
        docs = get_folder_documents(content_source)
        print(f"Found {len(docs)} documents")
        self.assertTrue(len(docs) > 0)

    def test_get_website_pdfs(self):
        content_source = {"name": "Test Website PDF",
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

        # If folder pdf_downloads does not exist, create it
        if not os.path.exists(content_source["location"]):
            os.makedirs(content_source["location"])


        docs = get_website_pdfs(content_source)
        print(f"Found {len(docs)} documents")
        self.assertTrue(len(docs) > 0)

    def test_blacklist_url_filter(self):
        sitemap_url = "https://superlinked.com/vectorhub/sitemap.xml"
        urls = get_sitemap_urls(sitemap_url)
        full_url_count = len(urls)

        # Blacklist a URL
        single_blacklist_url = "https://superlinked.com/vectorhub/articles/vector-indexes"
        new_url_list = blacklist_url_filter(urls, [single_blacklist_url])
        print(f"Reduced from {full_url_count} to {len(new_url_list)} with single URL filter")
        self.assertTrue(len(new_url_list) < full_url_count)

        wildcard_blacklist_url = "https://superlinked.com/vectorhub/articles/*"
        new_url_list = blacklist_url_filter(urls, [wildcard_blacklist_url])
        print(f"Reduced from {full_url_count} to {len(new_url_list)} with wildcard URL filter")
        self.assertTrue(len(new_url_list) < full_url_count)

    def test_url_whitelist(self):
        sitemap_url = "https://superlinked.com/vectorhub/sitemap.xml"
        urls = get_sitemap_urls(sitemap_url)
        full_url_count = len(urls)

        # Whitelist a URL
        single_whitelist_url = "https://superlinked.com/vectorhub/articles/vector-indexes"
        new_url_list = url_whitelist(urls, [single_whitelist_url])
        print(f"Reduced from {full_url_count} to {len(new_url_list)} with single URL filter")
        self.assertTrue(len(new_url_list) < full_url_count)

    def test_get_website_documents(self):
        content_source = {"name": "Test Website",
                          "type": "Website",
                          "location": "https://superlinked.com/vectorhub/sitemap.xml",
                          "chunk_size": 512,
                          "blacklist": [
                              "https://superlinked.com/vectorhub/articles/*",
                              "https://superlinked.com/vectorhub/tags/*"
                          ]
                          }
        docs = get_website_documents(content_source)
        print(f"Found {len(docs)} documents")
        self.assertTrue(len(docs) > 0)

    def test_get_google_drive_documents(self):
        # The location in this case is the google drive folder ID
        # You can get this from the URL when you are in the folder
        # You will need to enable the Google Drive API and create a service account
        # and then configure the local service account file path in the .env file GOOGLE_SERVICE_ACCOUNT_FILE

        content_source = {"name": "Test Google Drive",
                          "type": "Google Drive",
                          "location": "1chQNbpiaozntsP5obMnSeBl1DgW8EBd9"
                          }

        docs = get_google_drive_documents(content_source)
        full_doc_count = len(docs)
        print(f"Found {len(docs)} documents")
        self.assertTrue(len(docs) > 0)

        # Test with a blacklist
        content_source["blacklist"] = ["Medium*"]

        docs = get_google_drive_documents(content_source)
        print(f"Found {len(docs)} documents with blacklist")
        self.assertTrue(len(docs) < full_doc_count)

        # Test with a whitelist
        content_source.pop("blacklist")
        content_source["whitelist"] = ["Medium*"]

        docs = get_google_drive_documents(content_source)
        print(f"Found {len(docs)} documents with whitelist")
        self.assertTrue(len(docs) > 0)

        self.assertTrue(len(docs) < full_doc_count)






if __name__ == '__main__':
    unittest.main()
