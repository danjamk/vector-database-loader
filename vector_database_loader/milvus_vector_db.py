import os

from pymilvus import MilvusClient

from vector_database_loader.base_vector_db import (
    BaseVectorLoader,
    BaseVectorQuery
)


def create_vdb_row_from_document(document, embedding_client):
    """
    Create a vector from a document using the embedding client.

    :param document: The document to create a vector from.
    :param embedding_client: The LangChain embedding client to be used.
    :return: A vector representation of the document.
    """
    vector = embedding_client.embed_query(document.page_content)
    doc_metadata = document.metadata

    # TODO: make this generic, and enumerate the fields in the document metadata

    vdb_row = {
        "vector": vector,
        "source": doc_metadata.get("source", ""),
        "title": doc_metadata.get("title", ""),
        "description": doc_metadata.get("description", ""),
        "language": doc_metadata.get("language", ""),
        "page_content": document.page_content
    }

    return vdb_row


class MilvusVectorLoader(BaseVectorLoader):
    """
    Handles loading document embeddings into a Milvus vector database index.
    """
    milvus_client = None

    def __init__(self, index_name, embedding_client):
        # execute base class constructor
        super().__init__(index_name, embedding_client)
        # Now create a Milvus client
        # Authentication enabled with a cluster user and password
        milvus_uri = os.getenv('MILVUS_HOST')
        if milvus_uri is None:
            raise ValueError("MILVUS_HOST environment variable not set.  This is your hosted endpoint URL")

        milvus_token = os.getenv('MILVUS_TOKEN') # this is actually a userid and password, not a token
        if milvus_token is None:
            raise ValueError("MILVUS_TOKEN environment variable not set.  This is your userid and password")

        self.milvus_client = MilvusClient(
            uri=milvus_uri,
            token=milvus_token
        )
        print("Connected to Milvus")

    def load_document_batch(self, document_set, delete_index=False):
        """
        Loads a batch of document embeddings into the vector database.

        :param document_set: A list of document chunks to be embedded and stored.
        :param delete_index: Boolean flag indicating whether to delete the index before loading.
        :return: The Pinecone vector database instance.
        """

        # Convert the documents to the proper structure to load into Milvus
        document_rows = []
        print(f"   Preparing {len(document_set)} document chunks into vector rows")
        for document in document_set:
            vdb_row = create_vdb_row_from_document(document, self.embedding_client)
            document_rows.append(vdb_row)
            # print(f" vector row count: {len(document_rows)}")



        print(f"   Loading {len(document_set)} document chunks into VDB index {self.index_name}")
        print("Start inserting entities")
        insert_result = self.milvus_client.insert(self.index_name, document_rows, progress_bar=True)
        print("Inserting entities done")
        print(len(insert_result))
        return len(insert_result)

    def index_exists(self, index_name=None):
        """
        Checks if the specified Milvus collection exists.

        :param index_name: The name of the index to check. Defaults to self.index_name. This is the Milvus collection name.
        :return: Boolean indicating whether the index exists.
        """
        if index_name is None:
            index_name = self.index_name

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

        return self.milvus_client.describe_collection(index_name)


class MilvusVectorQuery(BaseVectorQuery):
    """
    Handles querying a Milvus vector database index.
    """
    milvus_client = None

    def __init__(self, index_name, embedding_client):
        self.index_name = index_name
        self.embedding_client = embedding_client
       # Now create a Milvus client
        # Authentication enabled with a cluster user and password
        milvus_uri = os.getenv('MILVUS_HOST')
        if milvus_uri is None:
            raise ValueError("MILVUS_HOST environment variable not set.  This is your hosted endpoint URL")

        milvus_token = os.getenv('MILVUS_TOKEN') # this is actually a userid and password, not a token
        if milvus_token is None:
            raise ValueError("MILVUS_TOKEN environment variable not set.  This is your userid and password")

        self.milvus_client = MilvusClient(
            uri=milvus_uri,
            token=milvus_token
        )
        print("Connected to Milvus")

    def query(self, query, num_results=5):
        """
        Queries the Milvus collection for the nearest neighbors of a query vector.

        :param query: The query text.
        :param num_results: The number of nearest neighbors to return.
        :return: The nearest neighbors of the query vector.
        """

        # Convert the query text to a vevtor
        query_vector = self.embedding_client.embed_query(query)

        query_results = self.milvus_client.search(
            collection_name=self.index_name,
            data=[query_vector],
            limit=num_results,
            output_fields=["title", "description", "source", "language", "page_content"]
        )
        # TODO: Convert this intp a list of document objects

        return query_results[0]