import os

from pymilvus import MilvusClient
from langchain_community.vectorstores import Milvus

from vector_database_loader.base_vector_db import (
    BaseVectorLoader,
    BaseVectorQuery
)

# https://python.langchain.com/docs/integrations/vectorstores/zilliz/

def get_milvus_client():
    """
    Get a Milvus client instance.

    :return: A Milvus client instance.
    """
    milvus_cloud_uri = os.getenv('ZILLIZ_CLOUD_URI')
    if milvus_cloud_uri is None:
        raise ValueError("ZILLIZ_CLOUD_URI environment variable not set.  This is your hosted endpoint URL")

    milvus_username = os.getenv('ZILLIZ_CLOUD_USERNAME')
    if milvus_username is None:
        raise ValueError("ZILLIZ_CLOUD_USERNAME environment variable not set.  This is your userid")

    milvus_password = os.getenv('ZILLIZ_CLOUD_PASSWORD')
    if milvus_password is None:
        raise ValueError("ZILLIZ_CLOUD_PASSWORD environment variable not set.  This is your password")

    milvus_client = MilvusClient(
        uri=milvus_cloud_uri,
        token=f"{milvus_username}:{milvus_password}"
    )
    return milvus_client



class MilvusVectorLoader(BaseVectorLoader):
    """
    Handles loading document embeddings into a Milvus vector database index.
    """
    milvus_client = None

    def load_document_batch(self, document_set):
        """
        Loads a batch of document embeddings into the vector database.

        :param document_set: A list of document chunks to be embedded and stored.
        :return: The Pinecone vector database instance.
        """

        milvus_cloud_uri = os.getenv('ZILLIZ_CLOUD_URI')
        if milvus_cloud_uri is None:
            raise ValueError("ZILLIZ_CLOUD_URI environment variable not set.  This is your hosted endpoint URL")

        milvus_username = os.getenv('ZILLIZ_CLOUD_USERNAME')
        if milvus_username is None:
            raise ValueError("ZILLIZ_CLOUD_USERNAME environment variable not set.  This is your userid")

        milvus_password = os.getenv('ZILLIZ_CLOUD_PASSWORD')
        if milvus_password is None:
            raise ValueError("ZILLIZ_CLOUD_PASSWORD environment variable not set.  This is your password")

        # TODO: Handle serverless clusters with a token

        vector_db = Milvus.from_documents(
            document_set,
            self.embedding_client,
            collection_name=self.index_name,
            # drop_old=delete_index,
            # auto_id=True,
            connection_args={
                "uri": milvus_cloud_uri,
                "user": milvus_username,
                "password": milvus_password,
                # "token": ZILLIZ_CLOUD_API_KEY,  # API key, for serverless clusters which can be used as replacements for user and password
                "secure": True,
            },
        )


    def index_exists(self, index_name=None):
        """
        Checks if the specified Milvus collection exists.

        :param index_name: The name of the index to check. Defaults to self.index_name. This is the Milvus collection name.
        :return: Boolean indicating whether the index exists.
        """
        if index_name is None:
            index_name = self.index_name

        if self.milvus_client is None:
            self.milvus_client = get_milvus_client()

        has_collection = self.milvus_client.has_collection(index_name, timeout=5)
        return has_collection

    def delete_index(self, index_name=None):
        """
        Deletes a Milvus collection.

        :param index_name: The name of the index to delete. This is the Milvus collection name.
        :return: Boolean indicating whether the deletion was successful.
        """
        if index_name is None:
            index_name = self.index_name

        if self.milvus_client is None:
            self.milvus_client = get_milvus_client()

        if self.index_exists(index_name):
            self.milvus_client.drop_collection(index_name)
            return True

    def create_index(self, index_name=None, embedding_client=None):
        """
        Creates a Pinecone index with the appropriate dimension size based on the embedding model.

        :param index_name: The name of the index to create.
        :param embedding_client: The LangChain embedding client to be used. Used to determine the index's dimension size.
        :return: Boolean indicating whether the index was successfully created.
        """
        if embedding_client is None:
            embedding_client = self.embedding_client

        if index_name is None:
            index_name = self.index_name

        dimension_size = self.get_vector_dimension_size()

        if self.milvus_client is None:
            self.milvus_client = get_milvus_client()

        self.milvus_client.create_collection(
            index_name,
            dimension_size,
            consistency_level="Strong",
            metric_type="L2",
            auto_id=True)

    def describe_index(self, index_name=None):
        """
        Describes a Milvus collection.

        :param index_name: The name of the index to describe. This is the Milvus collection name.
        :return: The description of the index.
        """
        if index_name is None:
            index_name = self.index_name

        if self.milvus_client is None:
            self.milvus_client = get_milvus_client()

        return self.milvus_client.describe_collection(index_name)


class MilvusVectorQuery(BaseVectorQuery):
    """
    Handles querying a Milvus vector database index.
    """

    def get_client(self):

        milvus_cloud_uri = os.getenv('ZILLIZ_CLOUD_URI')
        if milvus_cloud_uri is None:
            raise ValueError("ZILLIZ_CLOUD_URI environment variable not set.  This is your hosted endpoint URL")

        milvus_username = os.getenv('ZILLIZ_CLOUD_USERNAME')
        if milvus_username is None:
            raise ValueError("ZILLIZ_CLOUD_USERNAME environment variable not set.  This is your userid")

        milvus_password = os.getenv('ZILLIZ_CLOUD_PASSWORD')
        if milvus_password is None:
            raise ValueError("ZILLIZ_CLOUD_PASSWORD environment variable not set.  This is your password")

        # TODO: Handle serverless clusters with a token

        vdb = Milvus(
            self.embedding_client,
            collection_name=self.index_name,
            connection_args={
                "uri": milvus_cloud_uri,
                "user": milvus_username,
                "password": milvus_password,
                # "token": ZILLIZ_CLOUD_API_KEY,  # API key, for serverless clusters which can be used as replacements for user and password
                "secure": True,
            },
        )
        return vdb