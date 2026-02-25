from typing import List, Union
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.llm_models.schema import DocumentBaseModel, ChunkModel
from app.service.embedding.embedding import EmbeddingModel
from app.service.fileloader.pdf_loader import PDFLoader
from app.service.splitter.parentchild_splitter import ParentChildSplitter
from app.utils.log_tools import logger


class SemanticSplitter(ParentChildSplitter):
    """基于 LangChain 和 向量嵌入 的语义拆分类"""

    def __init__(self,file_name,embedding_model: EmbeddingModel,similarity_threshold: float = 0.8):

        super().__init__(file_name,embedding_model)
        """
        Args:
            embedder (Embeddings): LangChain 兼容的嵌入模型
            chunk_size (int): 文本拆分基础大小
            chunk_overlap (int): 拆分重叠大小
            similarity_threshold (float): 段落合并的语义相似度阈值
        """
        self.file_name = file_name
        self.embedder = embedding_model
        self.similarity_threshold = similarity_threshold
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)


    def _embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """批量嵌入文本，返回向量列表"""
        embeddings = self.embedder.embed_documents(texts)
        return [np.array(v) for v in embeddings]

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """计算余弦相似度"""
        return float(np.dot(v1, v2) /
                     (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

    def semantic_merge(self, docs: List[Document]) -> List[DocumentBaseModel]:
        """
        语义合并函数f
        根据 similarity_threshold 决定是否合并相邻块
        Args:
            docs (List[Document]): 基础拆分后的文档块
        """
        texts = [doc.page_content for doc in docs]
        vectors = self._embed_texts(texts)

        merged_docs: List[DocumentBaseModel] = []
        buffer_text = texts[0]
        buffer_vec = vectors[0]

        for text, vec in zip(texts[1:], vectors[1:]):
            sim = self._cosine_similarity(buffer_vec, vec)
            if sim >= self.similarity_threshold:
                # 语义相似 → 合并
                buffer_text += "\n" + text
                buffer_vec = (buffer_vec + vec) / 2
            else:
                # 不相似 → 推入并重置
                merged_docs.append(DocumentBaseModel(
                    source_file=self.file_name,page_content=buffer_text))
                buffer_text = text
                buffer_vec = vec

        # 最后一个
        merged_docs.append(DocumentBaseModel(source_file=self.file_name,page_content=buffer_text))
        return merged_docs

    def split(self, texts: Union[list[str],list[DocumentBaseModel]], semantic_merge: bool = True) -> List[ChunkModel]:
        """
        对单个长文本进行语义切分
        Args:
            text (str): 原始长文本
            semantic_merge (bool): 是否进行语义合并
        """
        # 1．基础字符拆分
        if isinstance(texts[0], DocumentBaseModel):
            texts = [text.page_content for text in texts]
        else:
            texts = texts
        docs = self.splitter.create_documents(texts)

        # 2．语义合并（可选）
        if semantic_merge:
            docs = self.semantic_merge(docs)

        merged_texts = [doc.page_content for doc in docs]


        parent_chunks = [self.create_chunk(text=text, parent_id="") for text in merged_texts]
        final_chunks = self.parent_child_split(parent_chunks)
        logger.info(f"{self.file_name} 切分完成，共生成 {len(final_chunks)} 个 chunks")
        for chunk in final_chunks:
            chunk.dense_vector = self.embedding_model.embed_query(chunk.chunk_text)
        return final_chunks


if __name__ == "__main__":
    embedding_model = EmbeddingModel(**config.embedding_config)

    # 原始文本
    pdf_path = "/Users/emilyguo/Desktop/TestFiles/Biz Onboarding 1205.pdf"
    long_text = PDFLoader(pdf_path).load()
    chunker = SemanticSplitter(file_name="test.pdf", embedding_model=embedding_model)
    segments = chunker.split(long_text)


    # splitter = MarkdownHeaderTextSplitter(headers_to_split_on=["#", "##"])
    # segments = splitter.split_text(long_text[0].page_content)

    print("分块数量：", len(segments))
    for i, seg in enumerate(segments):
        if seg.parent_id == "":
            print(f"[块 {i + 1}]", seg.chunk_text)

