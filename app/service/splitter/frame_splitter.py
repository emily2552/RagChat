from typing import List

from app.schema import DocumentBaseModel, ChunkModel
from app.service.fileloader.word_loader import WordLoader
from app.service.splitter.parentchild_splitter import ParentChildSplitter

from app.utils.log_tools import logger


class TextFrameSplitter(ParentChildSplitter):
    def __init__(self, file_name,embedding_model, texts_limit =  1000):
        super().__init__(file_name, embedding_model,texts_limit)
        """
        Args:
            texts_limit：最大字符数


        """


    def split_by_structure(self, documents: List[DocumentBaseModel]) -> List[ChunkModel]:
        """
        按结构边界组合元素为初始大块（每个章节/标题一次一个块）。
        """

        texts = []  # 存放最终的文本块
        current_block = ""  # 当前正在累积的块内容


        logger.info(f"正在对文件{self.file_name}进行结构化切分")
        for doc in documents:
            # 如果遇到一个新的结构边界（标题/头部）：
            if doc.category == "Title" or doc.category == "Header" or doc.category == "Image":
                # 如果 current_block 不是空（说明已有内容），则把它加入结果
                if current_block:
                    texts.append(current_block)
                # 开始一个新的章节块，用当前标题初始化
                current_block = doc.page_content

            else:
                # 该元素是正文/表格等内容，将其累积到当前块
                current_block += "\n" + doc.page_content

        # 遍历结束后把最后一个块加入
        if current_block:
            texts.append(current_block)

        parent_chunks = [self.create_chunk(text=text, parent_id="") for text in texts]

        return parent_chunks

    def split(self,documents: List[DocumentBaseModel]):
        structured_texts = self.split_by_structure(documents)
        logger.info(f"{self.file_name} 结构切分完成，共生成 {len(structured_texts)} 个结构块")
        final_chunks = self.parent_child_split(structured_texts)
        for chunk in final_chunks:
            chunk.dense_vector = self.embedding_model.embed_query(chunk.chunk_text)
        return final_chunks



if __name__ == "__main__":
    docx_path = "/Users/emilyguo/Downloads/Regtank Business Onboarding PRD v2.1.docx"
    loader = WordLoader(docx_path)
    docs = loader.load()
    temp = 0
    word_spliter = TextFrameSplitter("v2.1.docx")
    chunks = word_spliter.split(docs)
    print(len(chunks))
    for chunk in chunks:
            print(len(chunk.chunk_text))
    print(temp)

