import os


reranker_config = {
        "base_url": "https://api.siliconflow.cn/v1/rerank",
        "model_name": "BAAI/bge-reranker-v2-m3",
        "api_key": "sk-puvjenmrxfxzzhapjkosikdqvavnarzknnmccipnvalvrfgg"
    }



# # jina embedding 模型配置
# embedding_config= {
#     "base_url" : "https://api.jina.ai/v1/embeddings",
#     "model_name" : "jina-embeddings-v3",
#     "api_key":"jina_430c842c761e4cd4ad81459cd08c6b3dt-xAGCEto4Ah52EOVXxcrLHnSdv4",
#     "model_provider" : "jina"
# }

# siliconflow embedding 模型配置

# embedding_config= {
#     "base_url" : "https://api.siliconflow.cn/v1/embeddings",
#     "model_name" : "BAAI/bge-m3", # 1024 多语言
#     "api_key": "sk-puvjenmrxfxzzhapjkosikdqvavnarzknnmccipnvalvrfgg",
#     "model_provider": "siliconflow"
# }


# 千问-8B
embedding_config= {
    "base_url" : "https://api.siliconflow.cn/v1/embeddings",
    "model_name" : "Qwen/Qwen3-Embedding-8B", # 4096 多语言
    "api_key":"sk-puvjenmrxfxzzhapjkosikdqvavnarzknnmccipnvalvrfgg",
    "model_provider" : "siliconflow"
}





glm_ocr_config = {
    "base_url" : "https://open.bigmodel.cn/api/paas/v4/files/ocr",
    "api_key" : "535e3be69676401d9520124f606b912b.J3JPERm50o6NpdDC",
    "model_name" : "glm-4.6v"
}



# milvus_host_url = "http://t420.doublechaintech.cn:29530"
milvus_host_url = "http://127.0.0.1:19530"


# 基础模型配置
model_config = {
    "model": "Pro/zai-org/GLM-4.7",
    "model_provider": "openai",
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key": "sk-puvjenmrxfxzzhapjkosikdqvavnarzknnmccipnvalvrfgg"
}
