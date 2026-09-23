#!/usr/bin/env python3
"""Prepare HIR-16K for sampled-token OPD / CC-OPD training.

The raw dataset has mixed nested `ground_truth` schemas, so this script keeps
the raw verifier metadata as JSON strings and emits a stable VeRL-compatible
Parquet/JSONL schema. It never executes checker code embedded in the dataset.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_REPO_ID = "sastpg/HIR-16K"
DEFAULT_REVISION = "2a95f69eb56cc47edc16a45f939cde479673a4cb"
DEFAULT_FILE = "HIR_trainv1.jsonl"


@dataclass(frozen=True)
class PreparedRow:
    data_source: str
    task_type: str
    prompt: list[dict[str, str]]
    ability: str
    reward_model: dict[str, str]
    extra_info: dict[str, object]
    env_kwargs: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "data_source": self.data_source,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "ability": self.ability,
            "reward_model": self.reward_model,
            "extra_info": self.extra_info,
            "env_kwargs": self.env_kwargs,
        }


def load_raw_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def download_hir16k(repo_id: str, revision: str, filename: str, cache_dir: Path | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required when --input-jsonl is not provided") from exc

    if cache_dir is not None:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                revision=revision,
                cache_dir=str(cache_dir),
            )
        )
    return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, revision=revision))


def stringify_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def classify_ground_truth(ground_truth: object) -> str:
    if not isinstance(ground_truth, dict):
        return "unknown"
    keys = set(ground_truth)
    if {"instruction_id_list", "kwargs"}.issubset(keys):
        return "ifeval_like"
    if {"constraints", "constraint_pattern"}.issubset(keys):
        return "constraint_table"
    if {"checker", "functions"}.issubset(keys):
        return "llm_checker"
    return "unknown"


def render_instruction(prompt_text: str, criteria: list[str]) -> str:
    if criteria:
        constraint_block = "\n".join(f"- {criterion}" for criterion in criteria)
    else:
        constraint_block = "- Follow the user request faithfully."
    return f"{prompt_text.strip()}\n\nConstraints:\n{constraint_block}".strip()


def normalize_row(row: dict[str, object], index: int) -> PreparedRow:
    prompt_text = row.get("prompt") or row.get("question") or row.get("instruction") or row.get("context")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError(f"row {index} has no usable prompt/question/instruction field")

    criteria = stringify_list(row.get("criteria") or row.get("constraints") or row.get("rubrics"))
    source = str(row.get("source", "unknown"))
    raw_id = str(row.get("id", index))
    ground_truth = row.get("ground_truth", {})
    ground_truth_kind = classify_ground_truth(ground_truth)
    ground_truth_json = json.dumps(ground_truth, ensure_ascii=False, sort_keys=True)
    instruction = render_instruction(prompt_text, criteria)
    data_source = f"hir16k:{source}:{ground_truth_kind}"

    extra_info: dict[str, object] = {
        "index": index,
        "id": raw_id,
        "source": source,
        "criteria": criteria,
        "base_prompt": prompt_text.strip(),
        "full_instruction": instruction,
        "ground_truth_kind": ground_truth_kind,
        "ground_truth_json": ground_truth_json,
        "need_tools_kwargs": False,
        "tools_kwargs": {"unused": ""},
    }
    return PreparedRow(
        data_source=data_source,
        task_type="instruction_following",
        prompt=[{"role": "user", "content": instruction}],
        ability="instruction_following",
        reward_model={"style": "hir16k_ground_truth_json", "ground_truth": ground_truth_json},
        extra_info=extra_info,
        env_kwargs={"question": instruction, "ground_truth": ground_truth_json, "data_source": data_source},
    )


def split_rows(rows: list[dict[str, object]], val_size: int, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    val_count = min(max(val_size, 0), len(shuffled))
    return shuffled[val_count:], shuffled[:val_count]


def write_jsonl(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_parquet(rows: list[dict[str, object]], path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for parquet output") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def build_eval_manifest(path: Path) -> None:
    manifest = {
        "hir_train": {
            "dataset_id": DEFAULT_REPO_ID,
            "revision": DEFAULT_REVISION,
            "config": "default",
            "split": "train",
            "file": DEFAULT_FILE,
        },
        "eval_benchmarks": [
            {"name": "IFEval", "adapter": "prompt_response_jsonl", "local_input": "instruction_following_eval/data/input_data.jsonl"},
            {"name": "IFBench", "adapter": "prompt_response_jsonl", "local_input": "IFBench/data/IFBench_test.jsonl"},
            {"name": "MulDimIF", "adapter": "muldimif_conversation_append", "local_input": "MulDimIF/Data/test.json"},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HIR-16K for CC-OPD training.")
    parser.add_argument("--input-jsonl", type=Path, default=None, help="Optional local raw HIR_trainv1.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--filename", default=DEFAULT_FILE)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Limit raw rows before split; <=0 keeps all rows")
    parser.add_argument("--val-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jsonl-only", action="store_true", help="Skip parquet emission")
    args = parser.parse_args()

    raw_path = args.input_jsonl or download_hir16k(args.repo_id, args.revision, args.filename, args.cache_dir)
    raw_rows = load_raw_rows(raw_path)
    if args.limit and args.limit > 0:
        raw_rows = raw_rows[: args.limit]
    train_raw, val_raw = split_rows(raw_rows, args.val_size, args.seed)

    train_rows = [normalize_row(row, index).as_dict() for index, row in enumerate(train_raw)]
    val_rows = [normalize_row(row, index).as_dict() for index, row in enumerate(val_raw)]

    output_dir: Path = args.output_dir
    write_jsonl(train_rows, output_dir / "hir16k_train.jsonl")
    write_jsonl(val_rows, output_dir / "hir16k_val.jsonl")
    if not args.jsonl_only:
        write_parquet(train_rows, output_dir / "hir16k_train.parquet")
        write_parquet(val_rows, output_dir / "hir16k_val.parquet")
    build_eval_manifest(output_dir / "eval_manifest.json")

    summary = {
        "raw_path": str(raw_path),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "output_dir": str(output_dir),
        "parquet": not args.jsonl_only,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
