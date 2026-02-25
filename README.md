# RagDemo

一个**工程化、可扩展的检索增强生成（RAG）系统**，采用 **前后端分离架构**，后端基于 **FastAPI + LangChain + Milvus**，前端为独立运行的 Web 页面，适用于知识库问答、文档理解与内部 AI 工具场景。

RagDemo 不是教学 Demo，而是一个**面向真实使用场景设计的 RAG 工程骨架**，强调模块边界清晰、检索可控、组件可替换。

---

## 项目整体说明

- **RagDemo 是项目根目录**
- **后端服务**：负责文档处理、向量化、检索、RAG 推理
- **前端页面**：独立打开运行，通过 HTTP 接口与后端交互
- **Milvus 向量库**：需自行部署

---
## 系统架构

```text
Frontend (HTML)
   │
   ▼
FastAPI API 层
   │
   ▼
RAG Service Pipeline
   │   ├── 文档加载 / OCR
   │   ├── 文本切分（Chunking）
   │   ├── Embedding
   │   ├── 检索 & 重排序
   │   └── LLM 生成
   │
   ▼
Milvus 向量数据库
---

## 项目结构

RagDemo/
├── app/                         # 后端核心代码
│   ├── data_storage/            # Milvus 数据管理
│   ├── llm_models/              # LLM Schema
│   ├── ocr_server/              # OCR 服务
│   ├── prompts/                 # Prompt 模板
│   ├── service/                 # RAG 核心逻辑
│   ├── utils/                   # 工具模块
│   ├── config.py
│   └── main.py
│
├── frontend/                    # 前端页面（独立运行）
│   ├── RAG_chat/
│   └── rag_free/
│
├── README.md
└── requirements.txt

```

## 环境准备

### 1. 克隆项目

git clone https://github.com/yourname/RagDemo.git  
cd RagDemo  

### 2. 安装依赖

pip install -r requirements.txt  

---

## Milvus 向量数据库部署（需要自行部署）

- 官方文档：https://milvus.io/docs  
- Docker 单机部署：https://milvus.io/docs/install_standalone-docker.md  

Docker 示例：

```code
docker run -d \
  --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest  

```

## 初始化数据库与集合

执行一次：

python app/data_storage/storage.py  

---

## 启动后端

uvicorn app.main:app --reload  

访问接口文档：http://localhost:8000/docs  

---

## 前端使用方式

直接用浏览器打开：

frontend/RAG_chat/index.html  
frontend/rag_free/index.html  

确保后端服务已启动。

---

## Roadmap

- [x] RAG Pipeline
- [x] Milvus 混合检索
- [x] 前端页面
- [ ] 多模态 RAG
- [ ] Benchmark & 可观测性
