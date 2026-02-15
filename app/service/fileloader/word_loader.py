import base64
import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from PIL import Image
from pymilvus.client.abstract import logger
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from tqdm import tqdm
from app.schema import DocumentBaseModel
from ocr_server.image_ocr import get_ocr_result




class WordLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = UnstructuredWordDocumentLoader(self.file_path,mode="elements")

    from typing import List

    def _extract_images_from_docx(self) -> List[bytes]:
        """
        从 docx 中提取所有原始图片 bytes
        """
        if not self.file_path.lower().endswith(".docx"):
            raise ValueError("Only .docx files are supported")

        image_bytes_list = []
        with zipfile.ZipFile(self.file_path, "r") as docx_zip:
            logger.info(f"正在抽取文件 {self.file_path}图片部分")
            for file_info in docx_zip.infolist():
                # 跳过目录
                if file_info.is_dir():
                    continue
                if not file_info.filename.startswith("word/media/"):
                    continue

                image_bytes = docx_zip.read(file_info.filename)
                image_bytes_list.append(image_bytes)

        return image_bytes_list

    import base64
    from typing import List
    from io import BytesIO
    from PIL import Image

    MIN_IMAGE_SIZE = 30
    MIN_IMAGE_BYTES = 1024
    INVALID_EXT = (".emf", ".wmf")

    def _preprocess_images(self) -> List[str]:
        """
        图片预处理：
        - 过滤无效图片
        - 校验尺寸
        - 转 base64
        """
        MIN_IMAGE_SIZE = 30
        MIN_IMAGE_BYTES = 1024
        image_bytes_list = self._extract_images_from_docx()
        base64_images = []
        for image_bytes in image_bytes_list:
            # 文件体积过滤
            if len(image_bytes) < MIN_IMAGE_BYTES:
                continue

            try:
                image = Image.open(BytesIO(image_bytes))
                width, height = image.size
            except Exception:
                continue

            if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                continue

            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            base64_images.append(image_base64)

        return base64_images

    def images_load(self) -> List[str]:
        image_base64_list = self._preprocess_images()

        with ThreadPoolExecutor(max_workers=5) as executor:
            # 使用 map 并添加进度条

            results = list(tqdm(
                executor.map(get_ocr_result, image_base64_list),
                total=len(image_base64_list),
                desc="Processing images with OCR"
            ))


        # 过滤掉可能的 None 或空结果
        return [res for res in results if res]
    def load(self)->List[DocumentBaseModel]:
        docs = self.loader.load() # 加载文本信息
        file_name = docs[0].metadata.get("filename")
        results = []
        # 加载图片信息
        ocr_results = self.images_load()
        # 处理图片信息
        for i,ocr_result in enumerate(ocr_results):
            results.append(DocumentBaseModel(
                page_content=ocr_result,
                category="Image",
                source_file = file_name
            ))

        # 处理文本信息
        for doc in docs:
            results.append(DocumentBaseModel(
                page_content=doc.page_content,
                category=doc.metadata.get("category") if doc.metadata.get("category") else None,
                source_file = file_name

            ))
        return results


if __name__=="__main__":
    docx_path = "/Users/emilyguo/Desktop/TestFiles/Biz Onboarding 1207.docx"
    loader = WordLoader(docx_path)
    docs = loader.load()
    print(len(docs))
