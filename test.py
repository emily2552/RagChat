from pathlib import Path
from paddleocr import PPStructureV3

input_file = "/home/emily/RagChat/OpenClaw完全使用手册.pdf"
output_path = Path("/home/emily/RagChat")

pipeline = PPStructureV3()
output = pipeline.predict(input=input_file)

# 收集每页的Markdown结果
markdown_list = []
for res in output:
    md_info = res.markdown
    markdown_list.append(md_info)

# 合并多页Markdown
markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

# 保存到指定文件夹
mkd_file_path = output_path / f"{Path(input_file).stem}.md"
mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

with open(mkd_file_path, "w", encoding="utf-8") as f:
    f.write(markdown_texts)