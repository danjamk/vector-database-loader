import re
import os
import sys
import time

from colorama import Fore, Style
import requests
from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    SeleniumURLLoader,
    PyPDFLoader
)
import xml.etree.ElementTree as ET
from langchain.text_splitter import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 512


def print_progress(task_name, current, total, item_name):
    """
    Prints progress for a given task.

    :param task_name: Name of the task being executed.
    :param current: Current progress count.
    :param total: Total count of items to process.
    :param item_name: Name of the item being processed.
    """
    progress_message = f"{task_name}: Processing {current} of {total} - {item_name}"
    sys.stdout.write('\r' + Fore.BLUE + progress_message + Style.RESET_ALL)
    sys.stdout.flush()


def extract_filename_url(content_source, filename):
    """
    Extracts the URL corresponding to a given filename from the content source.

    :param content_source: Dictionary containing items with filenames and URLs.
    :param filename: Filename to search for in the content source.
    :return: URL corresponding to the filename, if found.
    """
    for item in content_source['items']:
        if item['filename'] == filename:
            return item['url']
    return None


def cleanup_documents(docs):
    """
    Cleans up document content by removing excessive newlines, spaces, and tabs.

    :param docs: List of document objects with page_content attributes.
    :return: List of cleaned document objects.
    """
    newline_regex = re.compile(r'\n+')
    space_regex = re.compile(r' {2,}')
    tab_regex = re.compile(r'\t+')

    for doc in docs:
        cleaned_content = newline_regex.sub('\n', doc.page_content)
        cleaned_content = space_regex.sub(' ', cleaned_content)
        cleaned_content = tab_regex.sub(' ', cleaned_content)
        doc.page_content = cleaned_content

    return docs


def document_chunker(documents, chunk_size):
    """
    Splits documents into smaller chunks for processing.  This uses langchain RecursiveCharacterTextSplitter

    :param documents: List of documents to be chunked.
    :param chunk_size: Desired size of each chunk.
    :return: List of chunked documents.
    """
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    if chunk_size == 1:
        return documents

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=round(chunk_size * 0.15),
        length_function=len,
    )
    docs_chunks = text_splitter.split_documents(documents)
    print(f"Chunked {len(documents)} documents into {len(docs_chunks)} chunks. Size={chunk_size}")

    return docs_chunks


def blacklist_url_filter(urls, blacklist):
    """
    Filters URLs based on a blacklist of exact and wildcard patterns.

    :param urls: List of URLs to filter.
    :param blacklist: List of blacklisted URLs or patterns.
    :return: Filtered list of URLs.
    """
    if not blacklist or len(blacklist) == 0:
        return urls

    exact_matches = [url for url in blacklist if "*" not in url]
    wildcard_matches = [url.replace("*", ".*") for url in blacklist if "*" in url]

    # Compile
    wildcard_patterns = [re.compile(pattern) for pattern in wildcard_matches]

    filtered_urls = []
    for url in urls:
        if url in exact_matches:
            continue

        if any(pattern.match(url) for pattern in wildcard_patterns):
            continue

        filtered_urls.append(url)

    return filtered_urls


def url_whitelist(urls, whitelist):
    """
    Filters URLs based on a whitelist of exact and wildcard patterns.

    :param urls: List of URLs to filter.
    :param whitelist: List of whitelisted URLs or patterns.
    :return: Filtered list of URLs.
    """

    if not whitelist or len(whitelist) == 0:
        return urls

    exact_matches = [url for url in whitelist if "*" not in url]
    wildcard_matches = [url.replace("*", ".*") for url in whitelist if "*" in url]

    # Compile regex patterns for wildcard matches
    wildcard_patterns = [re.compile(pattern) for pattern in wildcard_matches]

    # Filter URLs
    filtered_urls = []
    for url in urls:
        # Check against exact matches
        if url in exact_matches:
            filtered_urls.append(url)
            continue

        # Check against wildcard patterns
        if any(pattern.match(url) for pattern in wildcard_patterns):
            filtered_urls.append(url)
            continue

    return filtered_urls


def get_sitemap_urls(sitemap_url):
    """
    Extracts URLs from a given sitemap XML file.

    :param sitemap_url: URL of the sitemap.
    :return: List of extracted URLs.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/xml;q=0.9, */*;q=0.8'
    }
    sitemap_response = requests.get(sitemap_url, headers=headers, timeout=60)
    sitemap_response.raise_for_status()
    root = ET.fromstring(sitemap_response.content)
    urls = []

    if root.tag.endswith('sitemapindex'):
        for sitemap in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap"):
            loc = sitemap.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
            urls.extend(get_sitemap_urls(loc))
    else:
        for url in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
            loc = url.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
            urls.append(loc)

    return urls


def get_folder_documents(content_source):
    """
    Loads documents from a specified folder based on the content type.
    NOTE: If the documents are PDF's, a single PDF may be logged as multiple documents if it has multiple pages.

    :param content_source: Dictionary specifying the folder location, document type, and processing options.
    :return: List of document chunks ready for processing.
    """
    search_expression = None
    recursive = True

    directory = content_source['location']
    if content_source['type'] == 'Microsoft Word':
        search_expression = "*.docx"
        loader_class = Docx2txtLoader
    elif content_source['type'] == 'PDF':
        search_expression = "*.pdf"
        loader_class = PyPDFLoader
    else:
        raise ValueError(f"ERROR: Cannot handle loading documents of type {content_source['type']}")

    if 'recursive' in content_source:
        recursive = content_source['recursive']

    print(f"Reading {content_source['type']} documents from {content_source['location']}")
    loader = DirectoryLoader(path=directory, glob=search_expression, loader_cls=loader_class, recursive=recursive)
    folder_docs = cleanup_documents(loader.load())

    print(f"Documents loaded. Count={len(folder_docs)}")

    filtered_docs = folder_docs
    if 'whitelist' in content_source:
        filtered_docs = url_whitelist(filtered_docs, content_source['whitelist'])
    if 'blacklist' in content_source:
        filtered_docs = blacklist_url_filter(filtered_docs, content_source['blacklist'])

    print(f"Found {len(folder_docs)} documents. Reduced to {len(filtered_docs)}.")

    for doc in filtered_docs:
        print(f"   {doc.metadata['source']}")

    return document_chunker(filtered_docs, content_source.get('chunk_size', None))


def website_crawler(url_list, headless=True):
    """
    Crawls the given list of URLs using Selenium and loads their content.

    Args:
        url_list (list): A list of URLs to crawl.
        headless (bool, optional): Whether to run the Selenium browser in headless mode. Defaults to True.

    Returns:
        list: A list of loaded documents retrieved from the crawled URLs.
    """
    crawler = SeleniumURLLoader(urls=url_list, headless=headless,
                               arguments=["enable-features=NetworkServiceInProcess"])
    docs = crawler.load()

    ready_to_use_docs = []

    for doc in docs:
        ready_to_use_docs.append(doc)
    return ready_to_use_docs


def get_folder_documents(content_source):
    """
    Loads and processes documents from a specified directory based on the content type and filtering criteria.

    Args:
        content_source (dict): A dictionary containing the following keys:
            - 'location' (str): The directory path where documents are stored.
            - 'type' (str): The type of documents ('Microsoft Word' or 'PDF').
            - 'recursive' (bool, optional): Whether to search recursively. Defaults to True.
            - 'whitelist' (list, optional): A list of allowed URLs or document sources.
            - 'blacklist' (list, optional): A list of disallowed URLs or document sources.
            - 'chunk_size' (int, optional): The size of document chunks to return. If 0, returns full documents.

    Returns:
        list: A list of processed and optionally chunked documents.

    Raises:
        ValueError: If the document type is not supported.
    """
    search_expression = None
    recursive = True

    directory = content_source['location']
    if content_source['type'] == 'Microsoft Word':
        search_expression = "*.docx"
        loader_class = Docx2txtLoader

    elif content_source['type'] == 'PDF':
        # NOTE: The PDF parser will break the document up into pages, so one doc could translate into many
        search_expression = "*.pdf"
        loader_class = PyPDFLoader

    else:
        raise (f"ERROR: Cannot handle loading documents of type {content_source['type']}")

    if 'recursive' in content_source:
        recursive = content_source['recursive']

    print(f"Reading {content_source['type']} documents from {content_source['location']}")
    loader = DirectoryLoader(path=directory, glob=search_expression, loader_cls=loader_class, recursive=recursive)
    folder_docs = cleanup_documents(loader.load())

    print(f"Documents loaded. Count={len(folder_docs)}")

    filtered_docs = folder_docs
    if 'whitelist' in content_source:
        filtered_docs = url_whitelist(filtered_docs, content_source['whitelist'])

    if 'blacklist' in content_source:
        filtered_docs = blacklist_url_filter(filtered_docs, content_source['blacklist'])

    print(
        f"Found {len(folder_docs)} documents. Reduced to {len(filtered_docs)}.")

    for doc in filtered_docs:
        print(f"   {doc.metadata['source']}")

    if 'chunk_size' in content_source and content_source['chunk_size'] == 0:
        return filtered_docs
    else:
        doc_chunks = document_chunker(filtered_docs,
                                      content_source['chunk_size'] if 'chunk_size' in content_source else None)
        return doc_chunks


def download_pdf(url, filename):
    """
    Downloads a PDF from a given URL and saves it to a specified filename.

    :param url: The URL of the PDF file.
    :param filename: The destination filename to save the PDF.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded to {filename}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: downloading: {e}")


def get_website_documents(content_source, headless=True):
    """
    Extracts documents from a website based on the content source configuration.

    :param content_source: Dictionary defining the website source and filtering criteria.
    :param headless: Boolean indicating whether to run the browser in headless mode.
    :return: List of processed website documents.
    """
    if content_source['type'] != 'Website':
        raise (f"ERROR: Cannot handle loading documents of type {content_source['type']}")

    site_urls = None
    filtered_urls = None
    if 'location' in content_source:
        print(f"Scraping URLs from site map {content_source['location']}")
        site_urls = get_sitemap_urls(content_source['location'])

        filtered_urls = site_urls
        if 'whitelist' in content_source:
            filtered_urls = url_whitelist(filtered_urls, content_source['whitelist'])

        if 'blacklist' in content_source:
            filtered_urls = blacklist_url_filter(filtered_urls, content_source['blacklist'])

    else:
        site_urls = filtered_urls = content_source['items']

    print(
        f"Found {len(site_urls)} URLs. Reduced to {len(filtered_urls)}.")

    for url in filtered_urls:
        print(f"   {url}")

    website_documents = cleanup_documents(website_crawler(filtered_urls, headless))

    if 'chunk_size' in content_source and content_source['chunk_size'] == 0:
        return website_documents
    else:
        webpage_chunks = document_chunker(website_documents,
                                          content_source['chunk_size'] if 'chunk_size' in content_source else None)
        return webpage_chunks


def get_website_pdfs(content_source, delete_existing_files=True):
    """
    Downloads and processes PDFs from a website.

    :param content_source: Dictionary specifying the website source and PDF location.
    :param delete_existing_files: Boolean indicating whether to clear existing files before download.
    :return: List of processed PDF documents.
    """
    if delete_existing_files:
        folder = content_source['location']
        print(f"Cleaning (deleting files from) folder {folder}")
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    # Ensure the file is deleted
                    while os.path.exists(file_path):
                        time.sleep(0.1)
            except Exception as e:
                print(f"Error deleting file {file_path}")

    for item in content_source['items']:
        download_pdf(item['url'], f"{content_source['location']}/{item['filename']}")

    content_source_alt = {
        "name": "Website PDF Files",
        "type": "PDF",
        "location": content_source['location'],
        "chunk_size": 512
    }
    docs = get_folder_documents(content_source_alt)

    new_doc_array = []
    for doc in docs:
        filename = doc.metadata['source'].split('/')[-1]
        new_source = extract_filename_url(content_source, filename)
        if new_source:
            doc.metadata['source'] = new_source
            doc.metadata['filename'] = filename
        else:
            print(f"Unable to find source URL for {filename}")
        new_doc_array.append(doc)

    return new_doc_array