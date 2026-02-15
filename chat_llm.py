# chat_llm.py
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig
from langchain_core.chat_history import InMemoryChatMessageHistory

from app.ptompts.rag_prompt import system_prompt, doc_prompt, intention_prompt



# 创建 Prompt 模板

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个专业、严谨、结构化表达的 AI 助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


# 初始化llm

llm =  init_chat_model(
    model="Pro/zai-org/GLM-4.7",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1",
    api_key= "sk-puvjenmrxfxzzhapjkosikdqvavnarzknnmccipnvalvrfgg"
)
# llm =  init_chat_model(
#     model="allenai/molmo-2-8b:free",
#     model_provider="openai",
#     base_url="https://openrouter.ai/api/v1",
#     api_key= "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"
# )
#


router_llm = llm



# session_id -> ChatMessageHistory
_session_store = {}


def get_session_history(session_id: str):
    """
    为每个会话维护独立对话上下文
    """
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


chain = prompt | llm

def need_retrieval(question: str) -> bool:
    retrieval_router_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", intention_prompt),
            ("human", "{question}")
        ]
    )
    decision = router_llm.invoke(
        retrieval_router_prompt.format_messages(
            question=question
        )
    )

    return "YES" in decision.content.upper()


chat_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", doc_prompt),
])

rag_chain  = RunnableWithMessageHistory(
    rag_prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# ============================
# 5️⃣ CLI 对话入口
# ============================

def chat():
    print("✅ LangChain ChatBot + RAG 已启动（exit 退出）\n")
    session_id = "user_001"
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        # =========================
        # 1️⃣ 判断是否需要检索
        # =========================
        if need_retrieval(user_input):
            print("需要进行检索")
            docs = retriever.hybrid_search(user_input)
            context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(docs))
            print("📚 检索结果：",context )
            result = rag_chain.invoke(
                {"context": context, "question": user_input},
                config=RunnableConfig(configurable={"session_id": session_id})
            )
            print("📚 RAG:\n", result.content)
            print()
        # =========================
        # 2️⃣ 普通对话
        # =========================
        else:
            result = chat_chain.invoke(
                {"input": user_input},
                config=RunnableConfig(configurable={"session_id": session_id})
            )
            print("📚AI:", result.content)
            print()


if __name__ == "__main__":
    chat()
