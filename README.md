# RagDemo

A production-oriented Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangChain**, and **Milvus**, focusing on controllability, extensibility, and engineering best practices.

---

## Overview

RagDemo is an open-source RAG framework designed to address common issues in real-world knowledge-based QA systems, such as:

- poor chunking strategies  
- low retrieval relevance  
- lack of metadata control  
- limited observability and extensibility  

Instead of being a simple demo, this project aims to provide a **clean, modular, and extensible RAG architecture** that can be adapted for research, internal tools, or production systems.

---

## Key Features

- 📄 **Multi-format document ingestion** (PDF / Word / Markdown)
- 🧩 **Flexible chunking strategies** (recursive, semantic, parent–child)
- 🔍 **Vector-based retrieval** powered by Milvus
- 🧠 **Pluggable embedding models** (OpenAI / BGE / Qwen)
- 🚀 **FastAPI service** with REST & streaming support
- 🧱 **Engineering-first design** with clear modular boundaries

---

## Architecture

```
Client
  │
  ▼
FastAPI
  │
  ▼
RAG Service Layer
  │
  ▼
Milvus (Vector Store)
```

---

## Project Structure

```
ragdemo/
├── app/
│   ├── api/                 # FastAPI routes
│   ├── service/
│   │   ├── embedding/       # Embedding models
│   │   ├── retriever/       # Retrieval logic
│   │   └── rag/             # RAG pipeline
│   ├── prompts/             # Prompt templates
│   ├── schema/              # Pydantic models
│   ├── config.py            # Global configuration
│   └── main.py              # Entry point
├── scripts/                 # Data ingestion scripts
├── tests/
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourname/ragdemo.git
cd ragdemo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Milvus (Docker)

```bash
docker run -d \
  --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

### 4. Run the service

```bash
uvicorn app.main:app --reload
```

Visit: http://localhost:8000/docs

---

## Usage

### Chat with knowledge base

**POST /chat**

```json
{
  "query": "What is the system architecture?",
  "collection": "documents"
}
```

---

## Configuration

| Variable | Description | Example |
|--------|-------------|---------|
| EMBEDDING_MODEL | Embedding backend | bge-m3 |
| LLM_MODEL | Chat model | gpt-4o-mini |
| MILVUS_HOST | Milvus host | localhost |
| MILVUS_PORT | Milvus port | 19530 |
| CHUNK_SIZE | Chunk size | 512 |
| CHUNK_OVERLAP | Chunk overlap | 50 |

---

## Roadmap

- [x] Core RAG pipeline
- [x] FastAPI service layer
- [x] Milvus integration
- [ ] Multi-modal RAG
- [ ] Evaluation & benchmark module
- [ ] Observability & tracing



