import json
import requests
import base64
import os
from pypdf import PdfReader, PdfWriter

OCR_URL = "https://open.bigmodel.cn/api/paas/v4/layout_parsing"
API_KEY = "535e3be69676401d9520124f606b912b.J3JPERm50o6NpdDC"

MAX_SIZE_MB = 50
MAX_PAGES = 100


def _send_ocr(pdf_bytes):
    b64 = base64.b64encode(pdf_bytes).decode()
    payload = {"model": "GLM-OCR", "file": f"data:application/pdf;base64,{b64}"}
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    r = requests.post(OCR_URL, json=payload, headers=headers)
    r.raise_for_status()
    return json.loads(r.text)["md_results"]


def _check_pdf(file_path):
    size_ok = os.path.getsize(file_path) / (1024 * 1024) <= MAX_SIZE_MB
    pages_ok = len(PdfReader(file_path).pages) <= MAX_PAGES
    return size_ok and pages_ok


def _split_pdf(file_path):
    reader = PdfReader(file_path)
    total = len(reader.pages)
    pdf_slices = []
    for start in range(0, total, MAX_PAGES):
        writer = PdfWriter()
        end = min(start + MAX_PAGES, total)
        for i in range(start, end):
            writer.add_page(reader.pages[i])
        temp = f"{file_path}_chunk_{start}_{end}.pdf"
        with open(temp, "wb") as f:
            writer.write(f)
        pdf_slices.append(temp)
    return pdf_slices


def ocr_pdf_to_markdown(file_path):
    if _check_pdf(file_path):
        with open(file_path, "rb") as f:
            return _send_ocr(f.read())

    chunks = _split_pdf(file_path)
    results = []
    for chunk in chunks:
        with open(chunk, "rb") as f:
            results.append(_send_ocr(f.read()))
        os.remove(chunk)

    return "\n".join(results)

if __name__ == "__main__":
    pdf_path = "/Users/emilyguo/Desktop/TestFiles/589ebf8e5bb13.pdf"
    print(ocr_pdf_to_markdown(pdf_path))
