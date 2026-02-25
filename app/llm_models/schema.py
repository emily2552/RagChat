from typing import List, Optional

from langchain.chat_models import init_chat_model
from pydantic import BaseModel,Field


class UploadResponse(BaseModel):
    success: bool
    message: str

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source_docs: list

class ChunkModel(BaseModel):
    chunk_id: str = Field(..., description="文档块的唯一标识符")
    chunk_text: str = Field(..., description="文档块的文本内容")
    file_name: str = Field(default="", description="源文件名称")
    parent_id: str = Field(default="", description="父文档块的ID")
    dense_vector: Optional[List[float]] = Field(default=None, description="密集向量表示")
    def to_dict(self):
        return self.model_dump()


class DocumentBaseModel(BaseModel):
    source_file: str = Field(..., description="文件名称")
    category: str = Field(default=None, description="文档类别")
    page_content:str = Field(..., description="文档内容")

    def to_dict(self):
        return self.model_dump()

class EmbeddingRequest(BaseModel):
    base_url: str
    model_name: str
    api_key: str
    model_provider: str

    def to_dict(self):
        return self.model_dump()



class BaseChatConfig(BaseModel):
    model: str = Field(..., description="模型名称")
    model_provider: str =  Field(..., description="模型提供商")
    base_url: str = Field(..., description="API基础URL")
    api_key: str = Field(..., description="API密钥")

    def to_dict(self):
        return self.model_dump()


