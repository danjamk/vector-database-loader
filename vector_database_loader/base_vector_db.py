from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from vector_database_loader.document_processing_utils import (
    get_website_documents,
    get_folder_documents,
    get_website_pdfs,
    print_progress
)


class EmbeddingModels:
    """
    A class containing predefined embedding model configurations.
    """
    OPENAI_GPT_DEFAULT = {"name": "openai-gpt-default", "provider": "openai", "model": "default"}
    OPENAI_GPT_V3LARGE = {"name": "openai-gpt-v3-large", "provider": "openai", "model": "text-embedding-3-large"}
    HUGFACE_DEFAULT = {"name": "hugging-face-default", "provider": "hugging-face", "model": "default"}
    HUGFACE_BGE_LARGE = {"name": "hugging-face-bge-large", "provider": "hugging-face", "model": "BAAI/bge-large-en"}
    HUGFACE_MPNET_BASE = {"name": "hugging-face-mpnet-base", "provider": "hugging-face",
                          "model": "sentence-transformers/all-mpnet-base-v2"}
    HUGFACE_MINILM = {"name": "hugging-face-minilm", "provider": "hugging-face",
                      "model": "sentence-transformers/all-MiniLM-L6-v2"}
    # Additional models can be added here


def get_embedding(model=None):
    """
    Returns an embedding model instance based on the provided configuration.

    :param model: A dictionary containing model details (name, provider, model identifier). Defaults to OpenAI GPT-3 Large.
    :return: An embedding model instance.
    """
    embeddings = None
    if model is None:
        model = EmbeddingModels.OPENAI_GPT_V3LARGE

    if model['provider'] == 'openai':
        load_dotenv(find_dotenv())
        if model['name'] == EmbeddingModels.OPENAI_GPT_DEFAULT['name']:
            embeddings = OpenAIEmbeddings()
        else:
            embeddings = OpenAIEmbeddings(model=model['model'])

    elif model['provider'] == 'hugging-face':
        if model['name'] == EmbeddingModels.HUGFACE_DEFAULT['name']:
            embeddings = HuggingFaceEmbeddings()
        else:
            embeddings = HuggingFaceEmbeddings(model_name=model['model'])

    else:
        raise ValueError(f"Error getting embedding. There is no provider with name: {model['provider']}")

    return embeddings


class BaseVectorLoader:
    """
    Base class for loading documents into a vector database.
    """

    def __init__(self, index_name, embedding_model):
        """
        Initializes the BaseVectorLoader.

        :param index_name: The name of the index.
        :param embedding_model: The embedding model to be used.
        """
        self.index_name = index_name
        self.embedding_model = embedding_model
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

    def create_index(self, index_name=None, embedding_model=None):
        raise NotImplementedError

    def delete_index(self, index_name=None):
        raise NotImplementedError


class BaseVectorQuery:
    """
    Base class for querying a vector database.
    """

    def __init__(self, index_name, embedding_model):
        """
        Initializes the BaseVectorQuery class.

        :param index_name: The name of the index.
        :param embedding_model: The embedding model to be used.
        """
        self.index_name = index_name
        self.embedding_model = embedding_model
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
