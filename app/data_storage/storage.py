from typing import List

from langchain_core.documents import Document
from pymilvus import (
    MilvusClient,
    FieldSchema,
    CollectionSchema,
    DataType, Function, FunctionType,
)


from app import config
from app.utils.log_tools import logger


class KnowledgeStorage:
    dense_dim = 4096
    database_name = "test_database"

    def __init__(self):
        self.conn = MilvusClient(uri=config.milvus_host_url)

    def create_database(self, database_name):
        self.conn.list_databases()
        for db in self.conn.list_databases():
            if db == database_name:
                logger.info(f"Database '{database_name}' already exists, skipping creation.")
                return
        self.conn.create_database(database_name)

    def create_milvus_collection(self, collection_name):
        self.conn.use_database(self.database_name)
        if not self.conn.has_collection(collection_name):
            fields = [
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=65535, description="块id",is_primary=True),
                FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535, default_value="",
                            description="上下文文本",enable_analyzer=True ),
                FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=65535, description="父级块id"),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=65535, description="文档名称"),
                FieldSchema(name='sparse_vector', dtype=DataType.SPARSE_FLOAT_VECTOR, description="稀疏索引"),
                FieldSchema(name='dense_vector', dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim,
                            description="稠密索引"),

            ]
            schema = CollectionSchema(fields=fields, description="知识库")
            functions = Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["chunk_text"],
                output_field_names=["sparse_vector"],
            )
            schema.add_function(functions)
            #  创建集合
            self.conn.create_collection(collection_name, schema=schema, consistency_level="Strong")
            index_params = self.conn.prepare_index_params()
            index_params.add_index(index_name="sparse_vectors_index",
                                   field_name="sparse_vector",
                                   index_type="AUTOINDEX",
                                   metric_type="BM25"
                                   )
            index_params.add_index(field_name="dense_vector",
                                   index_name="dense_vectors_index",
                                   index_type="FLAT",
                                   metric_type="L2"
                                   )
            # 为字段创建索引
            self.conn.create_index(collection_name, index_params=index_params)
            logger.info(f"Collection '{collection_name}' created.")

    def drop_collection(self, collection_name):
        self.conn.use_database(self.database_name)
        if self.conn.has_collection(collection_name):
            self.conn.drop_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' dropped.")
        else:
            logger.info(f"Collection '{collection_name}' does not exist.")










if __name__ == '__main__':
    knowledge_storage = KnowledgeStorage()
    # knowledge_storage.drop_collection("TestInfo_4096")
    knowledge_storage.create_milvus_collection("TestInfo_4096")

