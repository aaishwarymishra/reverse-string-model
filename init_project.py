#!/usr/bin/env python3
"""CLI to create default project files: main.py, train.py, handler.py, config.yaml"""

from __future__ import annotations
import argparse
from pathlib import Path

MAIN_PY = Path("./templates/main.py").read_text()
HANDLER_PY = Path("./templates/handler.py").read_text()
CONFIG_YAML = Path("./templates/config.yaml").read_text()
TRAIN_PY = Path("./templates/train.py").read_text()


def write_file(path: Path, content: str, force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialize default project files")
    parser.add_argument(
        "--force", "-f", action="store_true", help="overwrite existing files"
    )
    parser.add_argument("--dir", "-d", default=".", help="target directory")
    args = parser.parse_args(argv)
    root = Path(args.dir)

    files = [
        (root / "main.py", MAIN_PY),
        (root / "train.py", TRAIN_PY),
        (root / "handler.py", HANDLER_PY),
        (root / "config.yaml", CONFIG_YAML),
    ]

    created = []
    skipped = []
    for path, content in files:
        ok = write_file(path, content, args.force)
        if ok:
            created.append(path)
        else:
            skipped.append(path)

    for p in created:
        print(f"Created: {p}")
    for p in skipped:
        print(f"Skipped (exists): {p}")

    if not created and not skipped:
        print("No files were written.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
