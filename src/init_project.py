#!/usr/bin/env python3
"""CLI to create default project files: main.py, train.py, handler.py, config.yaml"""

from __future__ import annotations
import argparse
from pathlib import Path

# Get directory relative to this file
PACKAGE_DIR = Path(__file__).parent.resolve()

MAIN_PY = (PACKAGE_DIR / "templates/main.py").read_text()
HANDLER_PY = (PACKAGE_DIR / "templates/handler.py").read_text()
CONFIG_YAML = (PACKAGE_DIR / "templates/config.yaml").read_text()
TRAIN_PY = (PACKAGE_DIR / "templates/train.py").read_text()


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
    parser = argparse.ArgumentParser(description="Toolbox for training language models")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    init_parser = subparsers.add_parser(
        "init", help="Initialize reusable training script"
    )
    init_parser.add_argument(
        "--dataset",
        type=str,
        default="string-reverse",
        help="Dataset to initialize (e.g. string-reverse)",
    )
    init_parser.add_argument(
        "--force", "-f", action="store_true", help="overwrite existing files"
    )
    init_parser.add_argument("--dir", "-d", default=".", help="target directory")

    args = parser.parse_args(argv)

    if args.command == "init":
        root = Path(args.dir)

        files = [
            (root / "main.py", MAIN_PY),
            (root / "train.py", TRAIN_PY),
            (root / "handler.py", HANDLER_PY),
        ]

        if args.dataset == "string-reverse":
            files.append((root / "config.yaml", CONFIG_YAML))
        else:
            files.append((root / "config.yaml", CONFIG_YAML))

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
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
