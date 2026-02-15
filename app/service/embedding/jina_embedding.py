import requests
import json
from typing import List
from langchain_core.documents import Document


class JinaEmbeddings:
    """
    Jina AI Embeddings 服务封装类
    用于将 Document 列表的 page_content 转换为向量
    """

    def __init__(self, api_key: str = None, model: str = "jina-embeddings-v3", task: str = "response_text-matching"):
        """
        初始化 Embedding 类

        Args:
            api_key: Jina API 密钥
            model: 使用的模型名称
            task: 任务类型 (response_text-matching, classification, etc.)
        """
        self.api_key = api_key or "jina_430c842c761e4cd4ad81459cd08c6b3dt-xAGCEto4Ah52EOVXxcrLHnSdv4"
        self.model = model
        self.task = task
        self.url = "https://api.jina.ai/v1/embeddings"


    def embed_query(self, query: str) -> List[float]:
        """
        将单个查询字符串转换为嵌入向量

        Args:
            query: 查询字符串

        Returns:
            单个嵌入向量
        """
        embeddings = self._get_embeddings([query])
        return embeddings[0]  # 返回第一个（也是唯一一个）向量

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        内部方法：调用 API 获取嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "task": self.task,
            "input": texts
        }

        response = requests.post(self.url, headers=headers, data=json.dumps(data))

        if response.status_code != 200:
            raise Exception(f"API 请求失败: {response.status_code}, {response.text}")

        result = response.json()

        # 提取所有嵌入向量
        embeddings = [item['embedding'] for item in result['data']]

        return embeddings


# 使用示例
if __name__ == "__main__":
    from app.service.fileloader.loader import UniversalFileLoader
    from app.service.splitter.parentchild_splitter import RecursiveSplitter

    # 加载文档
    file_path = "/files/student_profile_1.pdf"
    documents = UniversalFileLoader(file_path).load()

    # 切分文档
    rec_splitter = RecursiveSplitter(
        [doc.page_content for doc in documents],
        chunk_size=100,
        chunk_overlap=20
    )
    split_docs = rec_splitter.split_documents()

    # 创建 Embedding 实例
    embedder = JinaEmbeddings()



    print(f"Documents count: {len(split_docs)}")
    print(f"Embeddings count: {len(embeddings)}")
    print(f"First embedding dims: {len(embeddings[0]) if embeddings else 0}")

    # 测试单个查询
    query_embedding = embedder.embed_query("学生档案相关信息")
    print(f"Query embedding dims: {len(query_embedding)}")




