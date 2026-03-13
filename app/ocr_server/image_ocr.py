import base64
import json
import tempfile
import requests
from pathlib import Path

from sympy import false

from app import config

# BIGMODEL_OCR_URL = "https://open.bigmodel.cn/api/paas/v4/files/ocr"
# BIGMODEL_API_KEY = "535e3be69676401d9520124f606b912b.J3JPERm50o6NpdDC"




def get_ocr_result(image_base64: str):

    b64_image = "data:image/jpeg;base64," + image_base64

    # 👇 2. 构造 API 请求
    url = config.glm_config["base_url"]
    api_key = config.glm_config["api_key"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 👇 3. 构造 messages，包含图片和文本提示
    payload = {
        "model": config.glm_config.get("model_name", "glm-4v-flash"),
        "messages": [
            {
                "role": "user",
                "content": [
                    # 图片输入部分
                    {
                        "type": "image_url",
                        "image_url": { "url": b64_image }
                    },
                    # 文本提示部分
                    {
                        "type": "text",
                        "text": "请描述这张图片内容。"
                    }
                ]
            }
        ]
    }

    # 👇 4. 发起 POST 请求
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # 👇 5. 解析返回结果
    if response.status_code == 200:
        result = response.json()
        # 输出模型生成的文字描述
        return   result["choices"][0]["message"]["content"]
    else:
        raise Exception(f"请求失败，状态码：{response.status_code}+{response.text}")




def ocr_by_path(image_path: str) -> str:
    """
    通过 OCR 服务解析图像内容
    调用方式保持不变
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = "data:image/png;base64," + image_b64
    ocr_result = json.loads(get_ocr_result(data_uri)).get("words_result")
    image_info = ''
    for item in ocr_result:
        image_info += str(item)+ "\n"

    return image_info


if __name__ == "__main__":
    image_path = "/Users/emilyguo/Desktop/Snipaste_2026-01-19_09-38-31.png"


