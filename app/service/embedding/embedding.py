import json
from typing import List

import requests

from app import config
from app.schema import DocumentBaseModel
from app.service.embedding.model_provider import construct_silicon_params
from app.utils.log_tools import logger


class EmbeddingModel:
    def __init__(self,base_url: str, model_name: str,api_key: str, model_provider: str):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.model_provider = model_provider

    def _get_payload(self,text: str)-> dict:
        if self.model_provider == "siliconflow":
            payload = construct_silicon_params(text, self.base_url, self.model_name, self.api_key)
            return payload

        return {}

    def embed_query(self, query: str) -> List[float]:
        payload= self._get_payload(query)

        response = requests.post(**payload)
        if response.status_code != 200:
            url = payload.get("url")
            logger.error(f"请求错误，错误码：{response.status_code}")
            logger.error(f"错误信息：{response.text}")
            raise Exception(f"embedding请求错误:{url}")
        embedding = response.json()["data"][0]["embedding"]
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            payload = self._get_payload(text)
            response = requests.post(**payload)
            embeddings.append(response.json()["data"][0]["embedding"])

        return embeddings



if __name__ == "__main__":
    embedding_model = EmbeddingModel(**config.embedding_config)

    # 输入字符串列表
    texts = [
        "学生档案相关信息",
        "课程安排与教学计划",
        "学生成绩统计表"
    ]

    # 调用 embed_documents 方法生成向量
    # embeddings = embedding_model.embed_documents(texts)
    # # 输出结果
    # for i, emb in enumerate(embeddings):
    #     print(f"文本 {i + 1} 的向量维度: {len(emb)}")
    embedding = embedding_model.embed_query("学生档案相关信息")
    print(embedding)

