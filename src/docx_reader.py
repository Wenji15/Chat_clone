"""Word 文档（.docx）纯标准库文本提取。

.docx 本质是一个 zip 包，正文在 `word/document.xml`。这里用 zipfile +
xml.etree 解析，无需 python-docx 等第三方库。

只提取正文段落与表格文字（按原顺序），忽略页眉页脚、批注、图片等。
"""
from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _para_text(p: ET.Element) -> str:
    """提取一个段落（w:p）里的可见文字，保留制表符与换行。"""
    parts: list[str] = []
    for node in p.iter():
        tag = node.tag
        if tag == W + "t":
            parts.append(node.text or "")
        elif tag == W + "tab":
            parts.append("\t")
        elif tag in (W + "br", W + "cr"):
            parts.append("\n")
    return "".join(parts)


def _table_text(tbl: ET.Element) -> str:
    """把表格转成「单元格用 | 分隔、每行一行」的纯文本。"""
    rows: list[str] = []
    for tr in tbl.findall(W + "tr"):
        cells: list[str] = []
        for tc in tr.findall(W + "tc"):
            cell_paras = [_para_text(p) for p in tc.findall(W + "p")]
            cells.append(" ".join(x.strip() for x in cell_paras if x.strip()))
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_text(path: str | Path) -> str:
    """读取 .docx 文件的正文文本。"""
    with zipfile.ZipFile(path) as z:
        if "word/document.xml" not in z.namelist():
            raise ValueError(f"不是有效的 .docx 文件（缺少 word/document.xml）：{path}")
        xml_bytes = z.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    body = root.find(W + "body")
    if body is None:
        return ""

    blocks: list[str] = []
    for child in body:
        if child.tag == W + "p":
            text = _para_text(child)
            if text.strip():
                blocks.append(text)
        elif child.tag == W + "tbl":
            text = _table_text(child)
            if text.strip():
                blocks.append(text)
    return "\n".join(blocks)
