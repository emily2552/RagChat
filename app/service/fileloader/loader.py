from app.schema import DocumentBaseModel
from app.service.fileloader.pdf_loader import PDFLoader
from app.service.fileloader.word_loader import WordLoader
from app.utils.log_tools import logger

class UniversalFileLoader:
    """
    通用 Loader：根据后缀自动选择具体 Loader
    """

    def __init__(self, file_path: str):
        self.file_path = file_path.lower()

    def load(self) -> list[DocumentBaseModel]:
        if self.file_path.endswith(".pdf"):
            logger.info("文件为pdf，使用PDFOCRLoader对文件进行加载")
            loader = PDFLoader(self.file_path)
        elif self.file_path.endswith(".docx"):
            logger.info("文件为docx，使用WordLoader")
            loader = WordLoader(self.file_path)
        else:
            raise ValueError(f"Unsupported file type: {self.file_path}")
        return loader.load()


if __name__ == "__main__":
    # file_pdf = "/Users/emilyguo/Downloads/f058e7c5-d32e-437f-9950-42aa882f5a22_拆分文档.pdf"
    file_docx = ""

    # pdf_docs = UniversalFileLoader(file_pdf).load()
    docx_docs = UniversalFileLoader(file_docx).load()

    # print(f"PDF segments: {len(pdf_docs)}")
    # print(f"DOCX segments: {len(docx_docs)}")
