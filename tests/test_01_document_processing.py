import unittest
from vector_database_loader.document_processing_utils import (
    get_sitemap_urls,
    website_crawler,
    get_folder_documents,
    get_website_pdfs,
    blacklist_url_filter,
    url_whitelist,
    get_website_documents,
)

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



if __name__ == '__main__':
    unittest.main()
