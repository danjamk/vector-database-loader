from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from vector_database_loader.document_processing_utils import (
    get_website_documents,
    get_folder_documents,
    get_website_pdfs,
    print_progress
)


class BaseVectorLoader:
    """
    Base class for loading documents into a vector database.
    """

    def __init__(self, index_name, embedding_client):
        """
        Initializes the BaseVectorLoader.

        :param index_name: The name of the index.
        :param embedding_client: The LangChain embedding client to be used.
        """
        self.index_name = index_name
        self.embedding_client = embedding_client
        load_dotenv(find_dotenv())

    def load_sources(self, content, delete_index=False):
        """
        Loads multiple sources into the vector database index.

        :param content: A list of content sources.
        :param delete_index: Boolean flag to determine if the existing index should be deleted before loading.
        :return: The total number of documents loaded.
        """
        document_count = 0
        source_count = 0
        print(f"Going to load {len(content)} data sources into {self.index_name} index")

        for content_source in content:
            print(f"Processing content for {content_source['name']} ")
            print_progress("Load Source", source_count + 1, len(content), content_source['name'])

            if content_source['type'] == 'Website':
                content_docs = get_website_documents(content_source)
            elif content_source['type'] in ['Microsoft Word', 'PDF']:
                content_docs = get_folder_documents(content_source)
            elif content_source['type'] == 'Web PDFs':
                content_docs = get_website_pdfs(content_source)
            else:
                raise ValueError(f"ERROR: Cannot handle loading document type {content_source['type']}")

            document_count += len(content_docs)
            source_count += 1
            print(
                f"Loading {len(content_docs)} document chunks from {content_source['name']} into VDB index {self.index_name}")
            self.load_documents(content_docs, delete_index=delete_index)
            delete_index = False

        print(f"Done! Loaded {document_count} documents from {source_count} sources into index: {self.index_name}")
        return document_count

    def load_documents(self, document_set, delete_index=False):
        """
        Loads a set of documents into the index in batches.

        :param document_set: The list of document embeddings to be loaded.
        :param delete_index: Whether to delete the existing index before loading.
        """
        index_exists = self.index_exists()
        if index_exists and delete_index:
            self.delete_index()
            self.create_index()
        elif not index_exists:
            self.create_index()

        batch_size = 500
        total_batches = len(document_set) // batch_size + (1 if len(document_set) % batch_size > 0 else 0)
        print(f"Now loading {len(document_set)} document chunks in {total_batches} batches of {batch_size}")

        for batch_num in range(total_batches):
            start_index = batch_num * batch_size
            end_index = start_index + batch_size
            document_subset = document_set[start_index:end_index]
            self.load_document_batch(document_subset)
            print(f"Loaded batch {batch_num + 1} of {total_batches}")

    def load_document_batch(self, document_set):
        """
        Load a batch of documents. To be implemented in subclasses.
        """
        pass

    def index_exists(self, index_name=None):
        raise NotImplementedError

    def create_index(self, index_name=None, embedding_client=None):
        raise NotImplementedError

    def delete_index(self, index_name=None):
        raise NotImplementedError

    def get_vector_dimension_size(self):
        """
        Get the dimension size of the vector embeddings.

        :return: The dimension size of the vector embeddings.
        """
        embedding_vector = self.embedding_client.embed_query(
            "Some string to determine embedding dimensional size to create the index")
        dimension_size = len(embedding_vector)
        return dimension_size

    def describe_index(self, index_name=None):
        raise NotImplementedError


class BaseVectorQuery:
    """
    Base class for querying a vector database.
    """

    def __init__(self, index_name, embedding_client):
        """
        Initializes the BaseVectorQuery class.

        :param index_name: The name of the index.
        :param embedding_client: The LangChain embedding client to be used.
        """
        self.index_name = index_name
        self.embedding_client = embedding_client
        load_dotenv(find_dotenv())
        self.vdb_client = self.get_client()

    def get_client(self):
        raise NotImplementedError

    def query(self, query, num_results=4):
        """
        Performs a similarity search on the vector database.

        :param query: The search query.
        :param num_results: Number of top results to return.
        :return: Query results.
        """
        query_results = self.vdb_client.similarity_search(query, k=num_results)
        return query_results
