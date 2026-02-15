import os.path
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Optional

from pymilvus import MilvusClient
from tqdm import tqdm

from app import config
from app.data_storage.milvus_operation import MilvusOperator
from app.service.embedding.embedding import EmbeddingModel
from app.service.fileloader.loader import UniversalFileLoader
from app.service.splitter.dialogue_splitter import DialogueDocumentSplitter
from app.service.splitter.frame_splitter import TextFrameSplitter
from app.service.splitter.semantic_spilitter import SemanticSplitter

from app.utils.log_tools import logger


def process_file(file_path:str,collection_name:str,embedding_model:EmbeddingModel):
    loader = UniversalFileLoader(file_path)
    file_name = os.path.basename(file_path)

    file_doc = loader.load()
    _, file_extension = os.path.splitext(file_path)
    if file_extension in [".docx"]:
        logger.info(f"开始对{file_name} 使用框架切分")
        splitter = TextFrameSplitter(os.path.basename(file_path),embedding_model=embedding_model)

    elif file_extension in [".pdf"] and "群消息问答" not in os.path.basename(file_path): # 普通pdf使用语义切分
        logger.info(f"开始对{file_name} 使用语义切分")
        splitter = SemanticSplitter(os.path.basename(file_path),embedding_model=embedding_model)

    elif "群消息问答" in os.path.basename(file_path):
        logger.info(f"开始对{file_name} 使用对话切分")
        splitter = DialogueDocumentSplitter(os.path.basename(file_path),embedding_model=embedding_model)

    else:
        pass

    chunks = splitter.split(file_doc)
    conn = MilvusOperator(collection_name=collection_name)
    logger.info(f"{file_name} 所有的切分块插入数据库开始")
    for chunk in tqdm(chunks, desc=f"插入 {os.path.basename(file_path)} 数据", unit="chunk"):
        conn.insert_entities([chunk])
    logger.info(f"{file_name}所有切分块插入数据库完成")






if __name__ == "__main__":
    # 指定要处理的文件夹路径
    # folder_path = "/Users/emilyguo/Desktop/TestFiles"

    # 获取文件夹中所有文件的路径
    # file_paths = get_all_file_paths(folder_path)
    embedding_model = EmbeddingModel(**config.embedding_config)
    collection_name = "TestInfo_4096"
    # collection_name="TestInfo_1024"
    file_path = ["/Users/emilyguo/Desktop/TestFiles/Biz Onboarding 1205.pdf"]

    for file_path in file_path:
        logger.info(f"开始处理文件：{file_path}")
        process_file(file_path,collection_name,embedding_model)
