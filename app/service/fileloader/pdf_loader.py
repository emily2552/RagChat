# PDFtoMD
import base64
import os.path
from typing import List

from openai import OpenAI

from app import config
from app.schema import DocumentBaseModel

from app.utils.files_operations import pdf_to_images, cleanup_files
from app.utils.log_tools import logger
from ocr_server.layout_ocr import ocr_pdf_to_markdown



class PDFLoader:
    def __init__(self, file_path: str):
        """
        初始化PDF OCR处理器

        Args:
            pdf_path: PDF文件路径
            image_dir: 临时图片存储目录
            dpi: 图像渲染分辨率
        """
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)



    def load(self)-> List[DocumentBaseModel]:

        logger.info(f"开始使用ocr加载文件：{self.file_name}为md格式")
        html_str = ocr_pdf_to_markdown(self.file_path)
        document = DocumentBaseModel(source_file=self.file_name, page_content=html_str)
        logger.info(f"{self.file_name} 文件加载成功")

        return [document]


if __name__ == "__main__":
    pdf_path = "/Users/emilyguo/Desktop/TestFiles/DRI Group 2026年群消息问答.pdf"
    pdf_loader = PDFLoader(pdf_path)
    documents = pdf_loader.load()
    for document in documents:
        print(document.page_content)

