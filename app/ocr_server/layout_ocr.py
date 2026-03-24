import os
import time

import time
from pathlib import Path
from paddleocr import PPStructureV3
from tqdm import tqdm

from app.utils.log_tools import logger


def ocr_pdf_to_markdown(file_path):
    start_time = time.time()
    logger.info(f"🚀 开始使用ocr加载文件：{Path(file_path).name}为md格式")
    pipeline = PPStructureV3()
    output = pipeline.predict(input=file_path)

    markdown_list = []
    page_count = 0
    for res in tqdm(output,desc="处理文件中..."):
        markdown_list.append(res.markdown)
        page_count += 1


    logger.info(f"✅ 共处理 {page_count} 页")
    final_markdown_string = pipeline.concatenate_markdown_pages(markdown_list)

    end_time = time.time()
    total_duration = end_time - start_time
    logger.info("🏁 解析任务完成！")
    logger.info(f"统计信息 -> 总页数: {page_count} 页 | 运行总时间: {total_duration:.2f} 秒 | 平均每页耗时: {total_duration/page_count:.2f} 秒")
    return final_markdown_string.get("markdown_texts")


if __name__ == "__main__":
    pdf_path = "/Users/emilyguo/Desktop/pdf-sample_0.pdf"
    print(ocr_pdf_to_markdown(pdf_path))
