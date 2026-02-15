from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from typing import List, Dict, Any

from app import config
from app.schema import ChunkModel
from app.service.embedding.embedding import EmbeddingModel
from app.service.embedding.jina_embedding import JinaEmbeddings


from app.service.reranker import Reranker
from app.utils.log_tools import logger


class MilvusRetriever:
    """
    Milvus 检索器类，封装了 Milvus 向量数据库的检索功能
    """

    def __init__(self, collection_name: str, database: str, embedding_model: EmbeddingModel):
        """
        初始化 MilvusRetriever

        Args:
            collection_name: 集合名称
            database: 数据库名称
        """
        self.collection_name = collection_name
        self.database = database
        self.client = MilvusClient(uri=config.milvus_host_url)
        self.client.use_database(self.database)
        self.embedding = embedding_model
        self.reranker = Reranker(**config.reranker_config)
        self.output_fields = ["chunk_text", "file_name","parent_id"]

    def full_text_search(self, query: str, anns_field: str = "sparse_vector", limit: int = 10,) -> List[Dict[str, Any]]:

        """
        执行全文搜索
        Args:
            query: 搜索查询字符串
            anns_field: 用于搜索的向量字段名称，默认为 "sparse_vector"
            limit: 返回结果的最大数量，默认为 10
            output_fields: 需要返回的字段列表，默认为 ["chunk_text", "file_name"]

        Returns:
            搜索结果列表
        """

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field=anns_field,
            limit=limit,
            output_fields=self.output_fields,
        )

        # 返回第一个查询的结果（因为我们只传入了一个查询）
        return results[0] if results else []

    def dense_search(self, query: str, anns_field: str = "dense_vector", limit: int = 10,
                     rerank: bool = True, top_k: int = 15) -> List[Dict[str, Any]]:

        results = self.client.search(
            collection_name=self.collection_name,
            data=[self.embedding.embed_query(query)],
            anns_field=anns_field,
            limit=limit,
            output_fields=self.output_fields,
        )

        search_results = results[0] if results else []
        # 父块召回过程
        final_results = []
        processed_ids = set() # 用于去重
        for result in search_results:
            entity = result.get('entity', {})
            parent_id = entity.get('parent_id', '')
            # 如果存在父块ID，优先召回父块
            if parent_id and parent_id not in processed_ids:
                parent_results = self.client.query(
                    collection_name=self.collection_name,
                    filter= f"chunk_id == '{parent_id}'",
                    output_fields=self.output_fields
                )
                if parent_results:
                    parent_results = parent_results[0]
                    final_results.append(parent_results)
                    processed_ids.add(parent_id)

            elif not parent_id:
                final_results.append(result)

        # 如果启用了重排序
        if rerank and final_results:
            final_reranked_results = self._rerank_results(final_results, query, top_k)
            return [result["entity"]["chunk_text"] for result in final_reranked_results]
        return [result["entity"]["chunk_text"] for result in final_results]

    def _rerank_results(self, final_results: List[Dict[str, Any]], query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        对搜索结果进行重排序
        """
        # 将搜索结果转换为 ChunkModel 格式用于重排序
        chunks = []
        for result in final_results:
            entity = result.get('entity', {})
            chunk = ChunkModel(
                chunk_text=entity.get('chunk_text', ''),
                chunk_id=result.get('id', ''),
                file_name=entity.get('file_name', ''),
                parent_id=entity.get('parent_id', '')  # 添加parent_id字段
            )
            chunks.append(chunk)

        # 使用 Reranker 进行重排序

        reranked_results = self.reranker.rerank_chunks(
            query=query,
            chunks=chunks,
            top_n=top_k
        )

        # 根据重排序结果顺序输出块
        final_reranked_results = []
        for rerank_result in reranked_results:
            original_result_idx = rerank_result.get('index', -1)
            if 0 <= original_result_idx < len(final_results):
                original_result = final_results[original_result_idx].copy()
                # 添加重排序分数
                original_result['rerank_score'] = rerank_result['relevance_score']
                final_reranked_results.append(original_result)

        return final_reranked_results


    def hybrid_search(self, query: str, sparse_anns_field: str = "sparse_vector",
                      dense_anns_field: str = "dense_vector", limit: int = 30,
                      rerank: bool = True, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        执行混合搜索（结合稀疏向量和密集向量搜索）
        Args:
            query: 搜索查询字符串
            sparse_anns_field: 稀疏向量字段名称，默认为 "sparse_vector"
            dense_anns_field: 密集向量字段名称，默认为 "dense_vector"
            limit: 返回结果的最大数量，默认为 5
            rerank: 是否启用重排序功能，默认为 False
            top_k: 重排序后返回的顶部结果数量，默认为 3
            output_fields: 需要返回的字段列表，默认为 ["chunk_text", "file_name"]

        Returns:
            包含 chunk_text 和 similarity 的结果列表
        """
        # 生成密集向量查询嵌入
        logger.info("正在对用户问题进行向量化")
        query_embedding = self.embedding.embed_query(query)

        # 创建稀疏向量搜索请求
        sparse_search_params = {"metric_type": "BM25"}
        sparse_request = AnnSearchRequest(
            [query], sparse_anns_field, sparse_search_params, limit=limit
        )

        # 创建密集向量搜索请求
        dense_search_params = {"metric_type": "L2"}
        dense_request = AnnSearchRequest(
            [query_embedding], dense_anns_field, dense_search_params, limit=limit
        )

        # 执行混合搜索
        logger.info("正在进行数据库检索")
        results = self.client.hybrid_search(
            self.collection_name,
            [sparse_request, dense_request],
            limit=limit,
            ranker=RRFRanker(),
            output_fields=self.output_fields,
        )

        search_results = results[0] if results else []

        # 父块召回过程
        final_results = []
        processed_ids = set()
        for result in search_results:
            entity = result.get('entity', {})
            parent_id = entity.get('parent_id', '')

            if parent_id and parent_id not in processed_ids:
                parent_results = self.client.query(
                    collection_name=self.collection_name,
                    filter=f"chunk_id == '{parent_id}'",
                    output_fields=self.output_fields
                )
                if parent_results:
                    # 构造统一格式的结果
                    parent_doc = parent_results[0]
                    formatted_result = {
                        'entity': parent_doc,
                        'id': parent_doc.get('chunk_id', ''),
                        'distance': parent_doc.get('distance', 0.0),  # 默认距离
                    }
                    final_results.append(formatted_result)
                    processed_ids.add(parent_id)
            elif not parent_id:
                final_results.append(result)

        # 如果启用了重排序
        if rerank and final_results:
            final_reranked_results = self._rerank_results(final_results, query, top_k)
            # 返回 chunk_text 和 rerank_score
            logger.info("重排序结束，正在返回召回文档")
            return [
                {
                    "chunk_text": result["entity"]["chunk_text"],
                    "similarity": result.get("rerank_score", 0.0),
                    "file_name": result["entity"]["file_name"]
                }
                for result in final_reranked_results
            ]


        # 如果未启用重排序，返回 chunk_text 和 distance
        return [
            {
                "chunk_text": result["entity"]["chunk_text"],
                "similarity": result.get("distance", 0.0),
                "file_name": result["entity"]["file_name"]
            }
            for result in final_results
        ]


    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息

        Returns:
            集合描述信息
        """
        return self.client.describe_collection(self.collection_name)

    def close(self):
        """
        关闭客户端连接
        """
        self.client.close()


if __name__ == "__main__":
    retriever = MilvusRetriever(collection_name="TestInfo_1024", database="test_database", embedding_model=EmbeddingModel(**config.embedding_config))
    query = "Individuals Onboarding在哪一个界面的哪一个位置"

    results = retriever.hybrid_search(query=query)

    print("\nreranker向量搜索结果:")
    print(results)


