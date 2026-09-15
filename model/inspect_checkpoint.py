"""Inspect `model/SegFormer_large_parking.ckpt` (Lightning SegFormer weights).

Run from the pl-hot repo root:

    uv run --extra train python model/inspect_checkpoint.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pl_hot.checkpoint import inspect_checkpoint

DEFAULT_CKPT = Path(__file__).resolve().parent / "SegFormer_large_parking.ckpt"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    path = Path(args[0]) if args else DEFAULT_CKPT
    if not path.exists():
        print(f"Checkpoint not found: {path}", file=sys.stderr)
        return 2
    print(json.dumps(inspect_checkpoint(path), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
