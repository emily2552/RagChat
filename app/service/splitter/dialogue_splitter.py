import os
import re
from typing import List

from app import config
from app.schema import DocumentBaseModel, ChunkModel
from app.service.embedding.embedding import EmbeddingModel
from app.service.fileloader.pdf_loader import PDFLoader
from app.service.splitter.parentchild_splitter import ParentChildSplitter
from app.utils.log_tools import logger
from app.utils.snowflake import generate_unique_id


class DialogueDocumentSplitter(ParentChildSplitter):
    def __init__(self, file_name: str, embedding_model: EmbeddingModel):
        """
        初始化PDF OCR处理器

        Args:
            pdf_path: PDF文件路径
            image_dir: 临时图片存储目录
            dpi: 图像渲染分辨率
        """

        super().__init__(file_name=file_name,embedding_model=embedding_model, max_tokens=1000)

    def cut_key_words(self, documents: List[DocumentBaseModel]) -> List[ChunkModel]:
        """
        根据 'severity' 关键词将文本分割成多个块。
        Args:
            documents: 输入的原始文本块。
        Returns:
            分割后的块文本。
        """
        # 使用正则表达式匹配 'severity' 并分割文本
        pattern = r"(?<=\bSeverity\b)"  # 匹配 'severity' 后的位置
        if len(documents) != 1:
            raise ValueError("对话文档加载不止一个块，请检查加载过程")
        text = documents[0].page_content
        splits = re.split(pattern, text, flags=re.IGNORECASE)


        results = [part.strip() for part in splits]
        parent_chunks = [self.create_chunk(text=result, parent_id="") for result in results]
        final_chunks = self.parent_child_split(parent_chunks)

        return final_chunks
    def split(self, documents:List[DocumentBaseModel]) -> List[ChunkModel]:
        """
        将文档进行结构块切分。
        Args:
            documents: 输入的文档列表。
        Returns:
            切分后的结构块列表。
        """
        final_chunks = self.cut_key_words(documents)
        for chunk in final_chunks:
            chunk.dense_vector = self.embedding_model.embed_query(chunk.chunk_text)
        logger.info(f"{self.file_name} 切分完成，共生成 {len(final_chunks)} 个块")
        return final_chunks



if __name__ == "__main__":
    pdf_path = "/Users/emilyguo/Desktop/TestFiles/DRI Group 2026年群消息问答.pdf"
    splitter = DialogueDocumentSplitter(pdf_path, embedding_model=EmbeddingModel(**config.embedding_config))
    pdf_loader = PDFLoader(pdf_path)
    documents = pdf_loader.load()
    chunks = splitter.split(documents)
    i = 0
    for chunk in chunks:
        i = i + 1
        print(f"------------------------这是第{i}个块------------------")
        print(chunk.chunk_text)

