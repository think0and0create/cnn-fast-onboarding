"""ex3_file_io_cli.py — 文件 I/O + argparse CLI 演示。

本章用这个文件演示：
- pathlib.Path 处理路径（推荐替代 os.path）
- with 语句管理文件资源
- json 模块读配置文件
- argparse 构建 CLI
- 文件不存在时给清晰错误（输出到 stderr）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_json(path: str | Path) -> dict:
    """Read a UTF-8 JSON file and return its contents as a dict.

    Raises FileNotFoundError with a clear message if the path is missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def count_lines(path: str | Path) -> int:
    """Count lines in a text file using `with` for resource management."""
    p = Path(path)
    count = 0
    with p.open(encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="file_io_demo",
        description="Read a file and report either its line count or its JSON contents.",
    )
    parser.add_argument("path", type=Path, help="Path to the input file.")
    parser.add_argument(
        "--mode",
        choices=["lines", "json"],
        default="lines",
        help="What to report (default: lines).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.mode == "lines":
            print(count_lines(args.path))
        else:
            data = read_json(args.path)
            print(json.dumps(data, ensure_ascii=False))
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())