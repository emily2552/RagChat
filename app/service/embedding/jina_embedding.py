import requests
import json
from typing import List
from langchain_core.documents import Document
from app import config

class JinaEmbeddings:
    def __init__(self, api_key: str = None, model: str = None, task: str = None):
        self.api_key = api_key or config.jina_config["api_key"]
        self.model = model or config.jina_config["model_name"]
        self.task = task or config.jina_config["task"]
        self.url = config.jina_config["base_url"]

    def embed_query(self, query: str) -> List[float]:
        embeddings = self._get_embeddings([query])
        return embeddings[0]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {"model": self.model, "task": self.task, "input": texts}
        response = requests.post(self.url, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            raise Exception(f"API 请求失败: {response.status_code}, {response.text}")
        result = response.json()
        embeddings = [item['embedding'] for item in result['data']]
        return embeddings
