from typing import List

import requests


def construct_silicon_params(text: str,base_url: str, model_name:str, api_key: str) -> dict:
    """
    构造 requests.post 所需的参数
    """
    payload = {
        "model": model_name,
        "input": text
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    return {
        "url": base_url,
        "json": payload,
        "headers": headers
    }
