"""Jupyter ノートブックを marimo ノートブックに変換する。

    uv run tools/marimo_convert.py                                  # marimo/ に生成
    uv run --env-file .env marimo edit marimo/01_homework_direct_method.py

`marimo convert` はセルをそのまま移し、複数のセルで再定義されている名前
（`fig, ax` など）は `_fig, _ax` のセル内変数に直してくれる。残る非互換は
IPython の `display(...)` だけなので、marimo の `mo.output.append(...)` に置き換える。
出力セルは捨てられる（marimo が実行時に作り直す）。

生成物は `marimo/` に置き、git にも入れる（.ipynb を変えたら作り直す）。正本は .ipynb の方。
marimo 側で編集した内容を戻すなら `marimo export ipynb marimo/xxx.py -o xxx.ipynb`。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ["01_homework_direct_method.ipynb", "02_experiments.ipynb"]
OUT_DIR = ROOT / "marimo"


def convert(ipynb: Path) -> Path:
    """1 冊を変換して、生成したファイルのパスを返す。"""
    out = OUT_DIR / ipynb.with_suffix(".py").name
    subprocess.run(
        [sys.executable, "-m", "marimo", "convert", str(ipynb), "-o", str(out)], check=True
    )
    src = out.read_text()
    src = re.sub(r"\bdisplay\(", "mo.output.append(", src)
    out.write_text(src)
    # 置き換えで `mo` を使うようになったセルのシグネチャを整える
    subprocess.run([sys.executable, "-m", "marimo", "check", "--fix", str(out)], check=True)
    return out


def main() -> None:
    """01, 02 を marimo/ に変換する。"""
    OUT_DIR.mkdir(exist_ok=True)
    for name in NOTEBOOKS:
        print(convert(ROOT / name))


if __name__ == "__main__":
    main()
