from typing import List, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.llm_models.schema import ChunkModel, DocumentBaseModel
from app.service.embedding.embedding import EmbeddingModel
from app.service.embedding.jina_embedding import JinaEmbeddings
from app.utils.log_tools import logger
from app.utils.snowflake import generate_unique_id


class ParentChildSplitter:
    """
    结构块  父块(1000)  子块(500)
    """
    def __init__(self, file_name: str= "",embedding_model: EmbeddingModel =  None,max_tokens=1000):
        self.max_tokens = max_tokens
        self.file_name = file_name
        self.embedding_model = embedding_model if embedding_model else JinaEmbeddings()
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80,
                                                         separators=["\n\n", "\n", "。", ".", "!", "?", ""])
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20,
                                                        separators=["\n\n", "\n", "。", ".", "!", "?", ""])

    def create_chunk(self, text: str, parent_id):
        return ChunkModel(
            chunk_id=generate_unique_id(),
            chunk_text=text,
            file_name=self.file_name,
            parent_id=parent_id,
            dense_vector=None,  # 暂时不嵌入文本

        )


    def parent_child_split(self, input_chunks: List[ChunkModel]) -> List[ChunkModel]:
        """
        结构块  父块(1000)  子块(200)

        Returns:
            List[ChunkModel]
        """
        final_chunks: List[ChunkModel] = []

        for chunk in input_chunks:
            if len(chunk.chunk_text) <= 500:
                final_chunks.append(chunk)
            elif 500 < len(chunk.chunk_text) <= self.max_tokens:
                parent_chunk = chunk
                final_chunks.append(parent_chunk)
                child_texts = self.child_splitter.split_text(parent_chunk.chunk_text)
                for child_text in child_texts:
                    child_chunk = self.create_chunk(text=child_text, parent_id=parent_chunk.chunk_id)
                    final_chunks.append(child_chunk)
            else:
                parent_texts = self.parent_splitter.split_text(chunk.chunk_text)
                for parent_text in parent_texts:
                    parent_chunk = self.create_chunk(text=parent_text, parent_id="")
                    final_chunks.append(parent_chunk)
                    child_texts = self.child_splitter.split_text(parent_text)
                    for child_text in child_texts:
                        child_chunk = self.create_chunk(text=child_text, parent_id=parent_chunk.chunk_id)
                        final_chunks.append(child_chunk)

        return final_chunks
    def split(self,texts: Union[List[str], List[DocumentBaseModel]]) -> List[ChunkModel]:
        if isinstance(texts[0], DocumentBaseModel):
            new_texts = [text.page_content for text in texts]
        else:
            new_texts = texts
        parent_chunks = [self.create_chunk(text=text, parent_id="") for text in new_texts]
        final_chunks = self.parent_child_split(parent_chunks)
        logger.info(f"{self.file_name} 切分完成，共生成 {len(final_chunks)} 个 chunks")
        for chunk in final_chunks:
            chunk.dense_vector = self.embedding_model.embed_query(chunk.chunk_text)
        return final_chunks






if __name__ == "__main__":

    # 1. 创建分割器实例
    splitter = ParentChildSplitter(file_name="example_file.txt", max_tokens=1000)

    # 2. 准备待分割的文本列表
    texts = [
        "这是第一个较长的文本段落，包含多个句子。这个段落会根据设定的最大token数进行父块和子块分割。",
        "这是第二个文本段落，同样会被分割成父块和子块。每个父块最多1000个字符，子块最多500个字符。"
    ]

    # 3. 调用 split 方法进行分割
    chunks = splitter.split(texts)

    # 4. 输出结果
    print(f"总共生成了 {len(chunks)} 个块。")

    # 分别统计父块和子块数量
    parent_chunks = [chunk for chunk in chunks if chunk.parent_id == ""]
    child_chunks = [chunk for chunk in chunks if chunk.parent_id != ""]

    print(f"父块数量: {len(parent_chunks)}")
    print(f"子块数量: {len(child_chunks)}")

    # 查看前几个块的内容
    for i, chunk in enumerate(chunks[:5]):  # 只显示前5个
        chunk_type = "父块" if chunk.parent_id == "" else "子块"
        print(f"\n{i+1}. [{chunk_type}] ID: {chunk.chunk_id}")
        print(f"   内容: {chunk.chunk_text[:100]}...")
        if chunk.parent_id:
            print(f"   父ID: {chunk.parent_id}")
