import requests
import json
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.llm_models.schema import ChunkModel
from app.utils.log_tools import logger


class Reranker:
    """
    Jina AI Reranker 服务封装类
    用于对文档列表进行重排序
    """

    def __init__(self, base_url: str,api_key: str , model_name: str):
        """
        初始化 Reranker 类

        Args:
            api_key: Jina API 密钥
            model: 使用的重排序模型名称
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

    def rerank_chunks(self, query: str, chunks: List[ChunkModel], top_n: int = 10) -> List[Dict[str, Any]]:
        """
        对切块列表进行重排序

        Args:
            query: 查询字符串
            chunks: 切块后的对象列表
            top_n: 返回前 n 个结果


        Returns:
            重排序后的结果列表，包含文档、相关性分数等信息
        """
        # 提取所有文档的 page_content
        texts = [doc.chunk_text for doc in chunks]

        # 调用 API 进行重排序
        results = self._rerank(query, texts, top_n, return_documents=True )



        return results

    def rerank_texts(self, query: str, texts: List[str], top_n: int = 3,
                    return_documents: bool = False) -> List[Dict[str, Any]]:
        """
        对文本列表进行重排序
        Args:
            query: 查询字符串
            texts: 文本列表
            top_n: 返回前 n 个结果
            return_documents: 是否返回文档对象（对纯文本不起作用）
        Returns:
            重排序后的结果列表
        """
        return self._rerank(query, texts, top_n, return_documents)

    def _rerank(self, query: str, texts: List[str], top_n: int,
               return_documents: bool) -> List[Dict[str, Any]]:
        """
        内部方法：调用 API 进行重排序

        Args:
            query: 查询字符串
            texts: 文本列表
            top_n: 返回前 n 个结果
            return_documents: 是否返回文档对象

        Returns:
            重排序后的结果列表
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model_name,
            "query": query,
            "top_n": top_n,
            "documents": texts,
            "return_documents": return_documents
        }
        logger.info(f"正在进行重排序")

        response = requests.post(self.base_url, headers=headers, data=json.dumps(data))

        if response.status_code != 200:
            raise Exception(f"API 请求失败: {response.status_code}, {response.text}")

        result = response.json()

        # 返回重排序的结果
        return result.get('results', [])






# 使用示例
if __name__ == "__main__":

    from app.service.fileloader.loader import UniversalFileLoader



    # 1. 加载文档
    file_path = "/Users/emilyguo/Desktop/TestFiles/Biz Onboarding 1205.pdf"
    documents = UniversalFileLoader(file_path).load()

    # 2. 切分文档
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    doc = [Document(page_content=documents[0].page_content)]
    split_docs = rec_splitter.split_documents(doc)
    texts = [doc.page_content for doc in split_docs]

    # 3. 创建 Reranker 实例
    reranker_config = {
        "base_url": "https://api.siliconflow.cn/v1/rerank",
        "model_name": "BAAI/bge-reranker-v2-m3",
        "api_key": "sk-puvjenmrxfxzzhapjkosikdqvavnarzknnmccipnvalvrfgg"
    }
    reranker = Reranker(**reranker_config)

    # 4. 重排序文档
    query = "Add New Package具体内容"
    reranked_results = reranker.rerank_texts(
        query=query,
        texts=texts,
        top_n=5
    )

    # 5. 输出结果
    print(f"Query: {query}")
    print(f"Reranked results count: {len(reranked_results)}")
    for result in reranked_results:
        print(f"Index: {result['index']}, Relevance Score: {result['relevance_score']}")



