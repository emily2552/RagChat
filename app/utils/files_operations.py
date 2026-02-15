import os
from typing import List
import fitz  # PyMuPDF

from app.utils.log_tools import logger


def get_all_file_paths(folder_path: str) -> List[str]:
    """
    递归获取文件夹中所有文件的路径列表
    Args:
        folder_path: 目标文件夹路径
    Returns:
        List[str]: 所有文件路径的列表
    """
    file_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths

# pdf转图片
def pdf_to_images(pdf_path:str,image_dir:str) -> list[str]:
    """
    将PDF转换为图片
    Args:
        pdf_path:需要转化的图片的pdf路径
        image_dir:存放图片的文件夹路径，如果没有，则在同级目录下创建
    """

    os.makedirs(image_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        zoom = 300 / 72  # PDF 默认 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(image_dir, f"page_{page_num+1}.png")
        pix.save(img_path)
        image_paths.append(img_path)

    return image_paths

def cleanup_files(paths: list[str]):
    """清理文件"""
    try:
        for img_path in paths:
            if os.path.exists(img_path):
                os.remove(img_path)
        logger.info("文件批量删除成功")
    except Exception as e:
        logger.error(f"文件批量删除失败: {e}")

# 清理临时文件
if __name__ == "__main__":
    pdf_path = "/Users/emilyguo/Downloads/DRI Group 2026年群消息问答(1).pdf"
    images = pdf_to_images(pdf_path, "pdf_images")

    # cleanup_images(images)