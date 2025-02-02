import os
from time import sleep

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from vector_database_loader.base_vector_db import (
    BaseVectorLoader,
    get_embedding,
    BaseVectorQuery
)
from pinecone.exceptions import NotFoundException


def list_indexes():
    """
    Lists all available Pinecone indexes.

    :return: A list of index names.
    """
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=pinecone_api_key)
    indexes = pc.list_indexes()
    return indexes


class PineconeVectorLoader(BaseVectorLoader):
    """
    Handles loading document embeddings into a Pinecone vector database index.
    """

    def load_document_batch(self, document_set, delete_index=False):
        """
        Loads a batch of document embeddings into the vector database.

        :param document_set: A list of document chunks to be embedded and stored.
        :param delete_index: Boolean flag indicating whether to delete the index before loading.
        :return: The Pinecone vector database instance.
        """
        embeddings = get_embedding(self.embedding_model)
        print(f"   Loading {len(document_set)} document chunks into VDB index {self.index_name}")
        vdb = PineconeVectorStore.from_documents(document_set, embeddings, index_name=self.index_name)
        return vdb

    def index_exists(self, index_name=None):
        """
        Checks if the specified Pinecone index exists.

        :param index_name: The name of the index to check. Defaults to self.index_name.
        :return: Boolean indicating whether the index exists.
        """
        if index_name is None:
            index_name = self.index_name

        index_info = self.describe_index(index_name)
        return bool(index_info)

    def create_index(self, index_name=None, embedding_model=None):
        """
        Creates a Pinecone index with the appropriate dimension size based on the embedding model.

        :param index_name: The name of the index to create.
        :param embedding_model: The embedding model used to determine the index's dimension size.
        :return: Boolean indicating whether the index was successfully created.
        """
        if embedding_model is None:
            embedding_model = self.embedding_model

        if index_name is None:
            index_name = self.index_name

        embeddings = get_embedding(embedding_model)
        embedding_vector = embeddings.embed_query(
            "Some string to determine embedding dimensional size to create the index")
        dimension_size = len(embedding_vector)

        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pc = Pinecone(api_key=pinecone_api_key)

        pc.create_index(
            name=index_name,
            dimension=dimension_size,
            metric="euclidean",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # Wait for the index to be created and available
        while True:
            try:
                desc = pc.describe_index(index_name)
                if desc.get('status', {}).get('state') == 'Ready':
                    return True
            except NotFoundException:
                pass  # Index is not yet available
            sleep(5)

    def delete_index(self, index_name=None):
        """
        Deletes a Pinecone index.

        :param index_name: The name of the index to delete.
        :return: Boolean indicating whether the deletion was successful.
        """
        if index_name is None:
            index_name = self.index_name

        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pc = Pinecone(api_key=pinecone_api_key)
        try:
            print(f"Deleting index={index_name}")
            pc.delete_index(index_name)
            return True
        except NotFoundException as e:
            print(f"Error deleting Pinecone index={index_name}: Not found. error={e}")
            return False

    def describe_index(self, index_name=None):
        """
        Retrieves information about a Pinecone index.

        :param index_name: The name of the index to describe.
        :return: Dictionary containing index information, or None if the index is not found.
        """
        if index_name is None:
            index_name = self.index_name

        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pc = Pinecone(api_key=pinecone_api_key)
        try:
            print(f"Getting index description for {index_name}")
            return pc.describe_index(index_name)
        except NotFoundException:
            return None


class PineconeVectorQuery(BaseVectorQuery):
    """
    Handles querying a Pinecone vector database.
    """

    def get_client(self):
        """
        Initializes and returns a Pinecone vector database client.

        :return: PineconeVectorStore client instance.
        """
        embedding = get_embedding(self.embedding_model)
        vdb = PineconeVectorStore(index_name=self.index_name, embedding=embedding)
        return vdb

    def status_check(self):
        """
        Retrieves status and statistics of the Pinecone index.

        :return: A dictionary containing the index status and statistics.
        """
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pc = Pinecone(api_key=pinecone_api_key)
        index_info = pc.describe_index(self.index_name)
        index_dimension = index_info.dimension

        index = pc.Index(self.index_name)
        index_stats = index.describe_index_stats()

        return {
            "status": "OK",
            "index_info": index_info,
            "index_dimension": index_dimension,
            "index_stats": index_stats
        }
