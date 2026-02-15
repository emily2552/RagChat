from typing import Optional, List

from pymilvus import MilvusClient

from app import config
from app.schema import ChunkModel
from app.utils.log_tools import logger


class MilvusOperator:
    def __init__(self,database: str = "test_database", collection_name: Optional[str] = None):
        self.client = MilvusClient(uri=config.milvus_host_url)
        self.database = database
        self.collection_name = collection_name if collection_name else "TestInfo"
        self.conn = MilvusClient(uri=config.milvus_host_url)
        self.conn.use_database(database)

    def insert_entities(self,chunks: List[ChunkModel]):
        """
        插入数据
        :param collection_name: 集合名称
        :param data: 数据
        :return:
        """
        # 转换为 Milvus 标准数据
        data = [
            chunk.to_dict()
            for chunk in chunks
        ]
        logger.info(f"向数据库{self.collection_name}中插入数据")
        self.conn.insert(collection_name=self.collection_name, data=data)
        logger.info(f"向数据库{self.collection_name}中插入数据成功")