from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from app.ptompts.rag_prompt import intention_prompt

llm =  init_chat_model(
    model="Pro/zai-org/GLM-4.7",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1",
    api_key= "sk-puvjenmrxfxzzhapjkosikdqvavnarzknnmccipnvalvrfgg"
)


router_llm = llm


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
