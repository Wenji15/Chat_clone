"""OCR 验证脚本：对一张截图做中文文字识别，打印结果。

用法：
    python scripts/ocr_test.py <图片路径>
默认读取项目根目录的 _screen_test.png。
"""
from __future__ import annotations

import sys
from pathlib import Path

from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (ROOT / "_screen_test.png")
    if not img_path.exists():
        print(f"图片不存在：{img_path}")
        return 1

    engine = RapidOCR()
    result, elapse = engine(str(img_path))
    print(f"用时：{elapse}")
    if not result:
        print("没有识别到文字")
        return 0

    print(f"共 {len(result)} 个文本框：")
    for item in result:
        box, text, score = item[0], item[1], item[2]
        score = float(score)
        if score >= 0.5:
            # box 是四个角点，取左上角 x/y 便于观察版面
            x = int(min(p[0] for p in box))
            y = int(min(p[1] for p in box))
            print(f"[{score:.2f}] ({x:>5},{y:>4})  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
