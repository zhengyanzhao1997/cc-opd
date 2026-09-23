#!/usr/bin/env python3
"""Transcribe teacher's HIR_trainv1_rubrics_processed.jsonl to VeRL parquet
matching the schema produced by ``prepare_hir16k.py`` so that the student
sees prompts in the exact format the teacher (the rubric-conditioned teacher) was
RL-trained on (rubrics inlined as natural-language sentences, NOT bullet
block).

Schema parity with ``prepare_hir16k.py`` is the goal:

  data_source, task_type, prompt, ability, reward_model, extra_info, env_kwargs

Key differences vs. ``prepare_hir16k.py``:

  - prompt[0].content     := teacher row's ``messages[0].content`` VERBATIM
                            (== ``prompt`` field; rubrics inlined)
  - extra_info.full_instruction := same verbatim string
  - extra_info.base_prompt      := teacher row's ``question`` field
                                   (bare question, no rubrics).  Used by
                                   verl/trainer/ppo/cc_opd.py:138 so that
                                   LOO scoring does not have to reverse-
                                   engineer the inline format.
  - extra_info.criteria         := list[str] of rubric ``description`` fields
                                   (preserves CC-OPD invariant from
                                   verl/trainer/ppo/cc_opd.py:104-114).

Reward shape (``reward_model``), ground-truth classification, and env_kwargs
mirror prepare_hir16k.py exactly so the existing reward worker / evaluator /
HIR environment continue to work without any code change.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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


def load_jsonl(path: Path) -> list[dict[str, object]]:
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


def classify_ground_truth(ground_truth: object) -> str:
    """Mirror prepare_hir16k.classify_ground_truth so reward routing stays identical."""
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


def normalize_row(row: dict[str, object], index: int) -> PreparedRow:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {index} has no 'messages' field")
    first_msg = messages[0]
    if not isinstance(first_msg, dict):
        raise ValueError(f"row {index} messages[0] is not a dict")
    instruction = first_msg.get("content")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"row {index} messages[0].content is empty / non-string")
    role = first_msg.get("role", "user")

    base_prompt_field = row.get("question")
    if not isinstance(base_prompt_field, str) or not base_prompt_field.strip():
        raise ValueError(f"row {index} has no 'question' field")
    base_prompt = base_prompt_field.strip()

    rubrics_field = row.get("rubrics")
    if rubrics_field is None:
        criteria: list[str] = []
    elif isinstance(rubrics_field, list):
        criteria = []
        for j, rubric in enumerate(rubrics_field):
            if isinstance(rubric, dict) and isinstance(rubric.get("description"), str):
                criteria.append(rubric["description"])
            elif isinstance(rubric, str):
                criteria.append(rubric)
            else:
                raise ValueError(
                    f"row {index} rubrics[{j}] must be a dict with 'description' or a str; got {type(rubric).__name__}"
                )
    else:
        raise ValueError(f"row {index} 'rubrics' must be a list, got {type(rubrics_field).__name__}")

    raw_id = str(row.get("id", index))
    source = str(row.get("source", "unknown"))
    ground_truth = row.get("ground_truth", {})
    ground_truth_kind = classify_ground_truth(ground_truth)
    ground_truth_json = json.dumps(ground_truth, ensure_ascii=False, sort_keys=True)
    data_source = f"hir16k:{source}:{ground_truth_kind}"

    extra_info: dict[str, object] = {
        "index": index,
        "id": raw_id,
        "source": source,
        "criteria": criteria,
        "base_prompt": base_prompt,
        "full_instruction": instruction,
        "ground_truth_kind": ground_truth_kind,
        "ground_truth_json": ground_truth_json,
        "need_tools_kwargs": False,
        "tools_kwargs": {"unused": ""},
        # Marker so downstream readers can tell this came from the
        # teacher-aligned (inline) variant rather than the original block
        # variant produced by prepare_hir16k.py.
        "prompt_format": "teacher_inline_rubrics",
    }
    return PreparedRow(
        data_source=data_source,
        task_type="instruction_following",
        prompt=[{"role": role, "content": instruction}],
        ability="instruction_following",
        reward_model={"style": "hir16k_ground_truth_json", "ground_truth": ground_truth_json},
        extra_info=extra_info,
        env_kwargs={
            "question": instruction,
            "ground_truth": ground_truth_json,
            "data_source": data_source,
        },
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe teacher-aligned HIR-16K to VeRL parquet (inline rubric format)."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path(
            "<placeholder>/HIR_train_rubrics.jsonl"
        ),
        help="Path to teacher's HIR_trainv1_rubrics_processed.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--train-name",
        default="hir16k_train_teacher_aligned",
        help="Base name (without extension) for output train files",
    )
    parser.add_argument("--jsonl-only", action="store_true", help="Skip parquet emission")
    args = parser.parse_args()

    raw_rows = load_jsonl(args.input_jsonl)
    train_rows = [normalize_row(row, index).as_dict() for index, row in enumerate(raw_rows)]

    output_dir: Path = args.output_dir
    write_jsonl(train_rows, output_dir / f"{args.train_name}.jsonl")
    if not args.jsonl_only:
        write_parquet(train_rows, output_dir / f"{args.train_name}.parquet")

    src_counter: dict[str, int] = {}
    kind_counter: dict[str, int] = {}
    for row in train_rows:
        ei = row["extra_info"]
        src_counter[ei["source"]] = src_counter.get(ei["source"], 0) + 1
        kind_counter[ei["ground_truth_kind"]] = kind_counter.get(ei["ground_truth_kind"], 0) + 1

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_dir": str(output_dir),
        "train_rows": len(train_rows),
        "sources": src_counter,
        "ground_truth_kinds": kind_counter,
        "parquet": not args.jsonl_only,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
