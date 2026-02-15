import base64
import requests
import json

from app import config





def send_image_to_glm(image_path: str):
    # 👇 1. 把图片编码成 Base64
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    b64_image = "data:image/jpeg;base64," + b64_data

    # 👇 2. 构造 API 请求
    url = "https://api.z.ai/api/paas/v4/chat/completions"  # 示例 API 端点，替换成你的服务地址
    api_key = config.glm_config["api_key"]                        # 把这个替换成你自己的 API Key

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 👇 3. 构造 messages，包含图片和文本提示
    payload = {
        "model": "glm-4v-flash",  # 或者你要使用的 GLM 多模态模型名称
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



if __name__ == "__main__":
    image_path = "/Users/emilyguo/Desktop/Snipaste_2026-01-19_09-38-31.png"
    result = send_image_to_glm(image_path)
    print(result)

