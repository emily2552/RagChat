# RagDemo

一个**面向生产环境的检索增强生成（RAG）系统**，基于 **FastAPI + LangChain + Milvus** 构建，强调 **可控性、可扩展性与工程实践规范**。

---

## 项目背景

在真实的知识问答系统中，RAG 常见问题包括：

- 文档切分策略粗糙，语义割裂  
- 向量检索相关性低、不可控  
- 元数据缺失，无法做结构化约束  
- 缺乏工程层面的可扩展性与可观测性  



---

## 核心特性

- 📄 **多格式文档接入**（PDF / Word / Markdown）
- 🧩 **灵活的切分策略**（递归切分 / 语义切分 / 父子块）
- 🔍 **基于 Milvus 的向量检索**
- 🧠 **可插拔 Embedding 模型**（OpenAI / BGE / Qwen）
- 🚀 **FastAPI 服务化封装**（支持 REST / Streaming）
- 🧱 **工程优先的模块设计**，职责边界清晰

---

## 系统架构

```
Client
  │
  ▼
FastAPI API 层
  │
  ▼
RAG Service 层
  │
  ▼
Milvus 向量数据库
```

---

## 项目结构说明

```
ragdemo/
├── app/
│   ├── api/                 # FastAPI 路由层
│   ├── service/
│   │   ├── embedding/       # Embedding 模型封装
│   │   ├── retriever/       # 检索逻辑
│   │   └── rag/             # RAG 主流程
│   ├── prompts/             # Prompt 模板
│   ├── schema/              # Pydantic 数据结构
│   ├── config.py            # 全局配置
│   └── main.py              # 服务入口
├── app/data_storage/
│   └── storage.py           # Milvus 数据库 / 集合管理
├── scripts/                 # 文档入库脚本
├── tests/
├── requirements.txt
└── README.md
```

---

## 环境准备

### 1️⃣ 克隆项目

```bash
git clone https://github.com/yourname/ragdemo.git
cd ragdemo
```

### 2️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

---

## Milvus 向量数据库部署（需要自行部署）

RagDemo **不内置 Milvus**，需要你提前部署向量数据库。

- 官方文档：https://milvus.io/docs  
- Docker 部署：https://milvus.io/docs/install_standalone-docker.md  

### Docker 快速启动（示例）

```bash
docker run -d \
  --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

---

## Milvus 数据库 & 集合初始化

初始化逻辑位于：

```
app/data_storage/storage.py
```

执行：

```bash
python app/data_storage/storage.py
```

即可完成数据库与集合创建，之后可直接启动服务。

---

## 启动服务

```bash
uvicorn app.main:app --reload
```

访问：

```
http://localhost:8000/docs
```

---

## 接口示例

### 知识库问答

**POST /chat**

```json
{
  "query": "系统整体架构是怎样的？",
  "collection": "TestInfo_4096"
}
```

---

## 配置说明

| 配置项 | 含义 | 示例 |
|------|------|------|
| EMBEDDING_MODEL | 向量模型 | bge-m3 |
| LLM_MODEL | 对话模型 | gpt-4o-mini |
| MILVUS_HOST | Milvus 地址 | localhost |
| MILVUS_PORT | Milvus 端口 | 19530 |
| CHUNK_SIZE | 切分长度 | 512 |
| CHUNK_OVERLAP | 重叠长度 | 50 |

---

## Roadmap

- [x] 核心 RAG Pipeline  
- [x] FastAPI 服务化  
- [x] Milvus 混合检索（Dense + BM25）  
- [ ] 多模态 RAG  
- [ ] 自动化评测 / Benchmark  
- [ ] 可观测性 & Trace  
