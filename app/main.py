import json
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig
from langchain_core.chat_history import InMemoryChatMessageHistory

# 假设这些是你项目中的模块，保持原样导入
from app.data_storage.files_data_insert import process_file
from app.ptompts.rag_prompt import system_prompt, doc_prompt
from app.llm_models.schema import EmbeddingRequest, BaseChatConfig
from app.service.embedding.embedding import EmbeddingModel
from app.service.retriver import MilvusRetriever
from app.utils.chat_tools import need_retrieval
from app.utils.log_tools import logger


# ————— 初始化 FastAPI —————

app = FastAPI()

# 跨域允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ————— 数据类定义 —————

class RAGChatRequest(BaseModel):
    session_id: str  # 必填，用于区分不同会话历史
    message: str
    embedding: EmbeddingRequest
    collection: str
    chat_config: BaseChatConfig



class FileUploadRequest(BaseModel):
    file_paths: List[str]
    embedding: EmbeddingRequest
    collection: str
    chat_config: BaseChatConfig

class ChatConfig(BaseModel):
    session_id: str = Field(..., description="会话ID")
    question: str = Field(..., description="用户问题")
    base_model_config: BaseChatConfig = Field(..., description="模型配置")

# ————— 会话历史存储 —————

session_store = {}


def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


@app.post("/chat")
async def chat_api(req: ChatConfig):
    # 1. 提取参数
    session_id = req.session_id
    question = req.question
    model = req.base_model_config.to_dict()


    # 3. 初始化 LLM
    llm = init_chat_model(**model)

    # 4. 定义无 RAG 时的闲聊 Chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业、严谨、结构化表达的 AI 助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm
    chat_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 闲聊模式
    logger.info("正在生成答案")
    result = chat_chain.invoke(
        {"input": question},
        config=RunnableConfig(configurable={"session_id": session_id})
    )
    answer = result.content
    logger.info("")

    # 返回完整 JSON
    return {
        "reply": answer
    }



# ————— 核心接口：流式对话 —————

@app.post("/stream_chat")
async def stream_chat_api(req: RAGChatRequest):
    # 1. 提取参数
    session_id = req.session_id
    question = req.message
    embedding_model = EmbeddingModel(**req.embedding.to_dict())
    collect_name = req.collection
    model_config = req.chat_config

    # 2. 初始化检索器
    retriever = MilvusRetriever(
        collection_name=collect_name,
        database="test_database",
        embedding_model=embedding_model
    )

    # 3. 初始化 LLM
    llm = init_chat_model(**model_config.to_dict())

    # 4. 定义无 RAG 时的闲聊 Chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业、严谨、结构化表达的 AI 助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm
    chat_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 5. 定义流式生成器函数
    async def stream_generator():
        # 判断是否需要检索
        if need_retrieval(question):
            # 获取检索结果：[{"chunk_text": "...", "similarity": 0.95, "file_name": "..."}, ...]
            docs_data = retriever.hybrid_search(question)


            # [阶段一]：先发送参考文档数据给前端
            if docs_data:
                # 直接将包含 similarity 的字典列表发送给前端
                yield json.dumps({
                    "type": "docs",
                    "data": docs_data
                }, ensure_ascii=False) + "\n"

            # [阶段二]：构建 Prompt 上下文 (仅提取文本)

            logger.info("文档返回结束，正在生成回答")
            context_parts = []
            for i, item in enumerate(docs_data):
                # 容错处理：确保能取到文本，处理可能是对象或字典的情况
                if isinstance(item, dict):
                    text = item.get("chunk_text", "")
                else:
                    # 如果万一返回的不是字典，做个兼容
                    text = str(item)

                context_parts.append(f"[{i + 1}] {text}")

            context = "\n\n".join(context_parts)

            # 定义 RAG Chain
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", doc_prompt),
            ])

            rag_chain = RunnableWithMessageHistory(
                rag_prompt | llm,
                get_session_history,
                input_messages_key="question",
                history_messages_key="history",
            )
            logger.info("答案生成成功！")

            # [阶段三]：流式发送 LLM 回答
            async for chunk in rag_chain.astream(
                    {"context": context, "question": question},
                    config=RunnableConfig(configurable={"session_id": session_id})
            ):
                if chunk.content:
                    yield json.dumps({
                        "type": "text",
                        "data": chunk.content
                    }, ensure_ascii=False) + "\n"

        else:
            # 不需要检索，直接闲聊
            async for chunk in chat_chain.astream(
                    {"input": question},
                    config=RunnableConfig(configurable={"session_id": session_id})
            ):
                if chunk.content:
                    yield json.dumps({
                        "type": "text",
                        "data": chunk.content
                    }, ensure_ascii=False) + "\n"

    # 返回流式响应，使用 NDJSON 格式
    return StreamingResponse(
        stream_generator(),
        media_type="application/x-ndjson"
    )




@app.post("/upload_file")
def up_load_files(payload: FileUploadRequest):
    for file_name in payload.file_paths:
        file_path = "/Users/emilyguo/Desktop/TestFiles/" + file_name
        logger.info(f"开始处理文件：{file_name}")
        embedding_model = EmbeddingModel(**payload.embedding.to_dict())
        process_file(file_path,payload.collection,embedding_model)
        logger.info(f"处理完成文件：{file_name}")
    return {"message": "true"}


# ————— 启动入口 —————

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)