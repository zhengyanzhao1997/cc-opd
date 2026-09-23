#!/usr/bin/env python3
"""Resolve student and teacher model paths into a shell env file.

If the input is a filesystem path that already contains a
``config.json``, it is used directly. Otherwise it is treated as a Hugging
Face repo id and downloaded via ``huggingface_hub.snapshot_download``.

The script writes ``output-env`` as a shell file with ``STUDENT_MODEL_PATH``
and ``TEACHER_MODEL_PATH`` so that ``train.sh`` can source the local paths
without embedding model weights in this code release.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def normalize_model_path(path):
    value = str(path).strip()
    if value != "/":
        value = value.rstrip("/")
    if not value:
        raise ValueError("resolved model path is empty")
    return value


def resolve_model_path(spec: str, cache_root: Path) -> str:
    candidate = Path(spec)
    if (candidate / "config.json").exists():
        return normalize_model_path(candidate.resolve())

    local_dir = cache_root / spec.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)
    if (local_dir / "config.json").exists():
        return normalize_model_path(local_dir)

    from huggingface_hub import snapshot_download

    max_workers = int(os.getenv("HF_HUB_MAX_WORKERS", "8"))
    resolved = snapshot_download(repo_id=spec, local_dir=str(local_dir), max_workers=max_workers)
    return normalize_model_path(resolved or local_dir)


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve student and teacher model paths.")
    parser.add_argument("--student", required=True, help="Local path or HF repo id of the student model")
    parser.add_argument("--teacher", required=True, help="Local path or HF repo id of the teacher model")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.getenv("MODEL_CACHE_ROOT", str(Path.home() / ".cache" / "cc_opd_models"))),
        help="Directory where HF-downloaded weights are cached.",
    )
    parser.add_argument("--output-env", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Skip download; write specs as paths for validation only")
    args = parser.parse_args()

    if args.dry_run:
        student_path = normalize_model_path(args.student)
        teacher_path = normalize_model_path(args.teacher)
    else:
        student_path = resolve_model_path(args.student, args.cache_root)
        teacher_path = resolve_model_path(args.teacher, args.cache_root)

    args.output_env.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"export STUDENT_MODEL_PATH={shell_quote(student_path)}",
        f"export TEACHER_MODEL_PATH={shell_quote(teacher_path)}",
    ]
    args.output_env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_json = args.output_env.with_name(args.output_env.name + ".json")
    output_json.write_text(
        json.dumps({"student": student_path, "teacher": teacher_path}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"student": student_path, "teacher": teacher_path}, sort_keys=True))


if __name__ == "__main__":
    main()
