#!/usr/bin/env python3
"""Prepare IFEval, IFBench, and MulDimIF validation data for CC-OPD.

The generated parquet/jsonl keeps evaluator-critical fields in ``extra_info`` so
validation generation can later export exact evaluator response files. In
particular, IFEval/IFBench prompt strings are preserved byte-for-byte, and
MulDimIF examples keep their original conversation and constraint objects.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PreparedBenchmarkRow:
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


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(value)
    return rows


def read_json_array(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array")
    rows: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{index}] is not a JSON object")
        rows.append(item)
    return rows


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


def normalize_benchmark_root(path: Path) -> Path:
    if (path / "Benchmark").is_dir():
        return path / "Benchmark"
    return path


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def maybe_limit(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    if limit <= 0:
        return rows
    return rows[:limit]


def build_if_row(benchmark: str, example: dict[str, object], index: int) -> PreparedBenchmarkRow:
    prompt = example.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError(f"{benchmark} row {index} has no string prompt")

    benchmark_id = str(example.get("key", index))
    original_example_json = json.dumps(example, ensure_ascii=False, sort_keys=True)
    extra_info: dict[str, object] = {
        "index": index,
        "id": benchmark_id,
        "benchmark": benchmark,
        "benchmark_id": benchmark_id,
        "adapter": "prompt_response_jsonl",
        "original_prompt": prompt,
        "instruction_id_list": example.get("instruction_id_list", []),
        "kwargs_json": json.dumps(example.get("kwargs", []), ensure_ascii=False, sort_keys=True),
        "original_example_json": original_example_json,
        "need_tools_kwargs": False,
        "tools_kwargs": {"unused": ""},
    }
    data_source = f"benchmark:{benchmark}"
    return PreparedBenchmarkRow(
        data_source=data_source,
        task_type="instruction_following",
        prompt=[{"role": "user", "content": prompt}],
        ability="instruction_following",
        reward_model={"style": "external_instruction_benchmark", "ground_truth": original_example_json},
        extra_info=extra_info,
        env_kwargs={"question": prompt, "ground_truth": original_example_json, "data_source": data_source},
    )


def normalize_conversations(value: object, index: int) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"MulDimIF row {index} has no conversations list")
    conversations: list[dict[str, str]] = []
    for turn_index, turn in enumerate(value):
        if not isinstance(turn, dict):
            raise ValueError(f"MulDimIF row {index} conversation {turn_index} is not an object")
        role = str(turn.get("role", "user"))
        content = turn.get("content")
        if not isinstance(content, str):
            raise ValueError(f"MulDimIF row {index} conversation {turn_index} has no string content")
        conversations.append({"role": role, "content": content})
    return conversations


def first_user_prompt(conversations: list[dict[str, str]], index: int) -> str:
    for turn in conversations:
        if turn["role"] == "user":
            return turn["content"]
    raise ValueError(f"MulDimIF row {index} has no user conversation turn")


def build_muldimif_row(example: dict[str, object], index: int) -> PreparedBenchmarkRow:
    conversations = normalize_conversations(example.get("conversations"), index)
    prompt = first_user_prompt(conversations, index)
    benchmark_id = str(example.get("id", index))
    original_example = dict(example)
    original_example["conversations"] = conversations
    original_example_json = json.dumps(original_example, ensure_ascii=False, sort_keys=True)
    constraints = example.get("constraints", [])
    extra_info: dict[str, object] = {
        "index": index,
        "id": benchmark_id,
        "benchmark": "muldimif",
        "benchmark_id": benchmark_id,
        "adapter": "muldimif_conversation_append",
        "original_prompt": prompt,
        "constraints": constraints,
        "constraint_pattern": example.get("constraint_pattern"),
        "difficulty": example.get("difficulty"),
        "original_example_json": original_example_json,
        "need_tools_kwargs": False,
        "tools_kwargs": {"unused": ""},
    }
    data_source = "benchmark:muldimif"
    return PreparedBenchmarkRow(
        data_source=data_source,
        task_type="instruction_following",
        prompt=conversations,
        ability="instruction_following",
        reward_model={"style": "external_instruction_benchmark", "ground_truth": original_example_json},
        extra_info=extra_info,
        env_kwargs={"question": prompt, "ground_truth": original_example_json, "data_source": data_source},
    )


def build_manifest(summary: dict[str, object], benchmark_root: Path) -> dict[str, object]:
    return {
        "source": "TURLEing/Rubrics-To-Tokens/Benchmark",
        "benchmark_root": str(benchmark_root),
        "generation_settings": {
            "temperature": 0.6,
            "max_output_tokens": 4096,
            "note": "Generation settings only; evaluator commands consume response files.",
        },
        "benchmarks": {
            "ifeval": {
                "enabled": True,
                "adapter": "prompt_response_jsonl",
                "response_schema": {"prompt": "exact original prompt", "response": "model answer"},
            },
            "ifbench": {
                "enabled": True,
                "adapter": "prompt_response_jsonl",
                "response_schema": {"prompt": "exact original prompt", "response": "model answer"},
            },
            "muldimif": {
                "enabled": True,
                "adapter": "muldimif_conversation_append",
                "response_schema": "original object with generated assistant response appended to conversations",
            },
        },
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare instruction-following benchmark validation parquet/jsonl.")
    parser.add_argument("--benchmark-root", type=Path, required=True, help="Path to Rubrics-To-Tokens repo root or Benchmark directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--combined-name", default="instruction_benchmarks_val")
    parser.add_argument("--ifeval-input", type=Path, default=None)
    parser.add_argument("--ifbench-input", type=Path, default=None)
    parser.add_argument("--muldimif-input", type=Path, default=None)
    parser.add_argument("--limit-per-benchmark", type=int, default=0)
    parser.add_argument("--jsonl-only", action="store_true")
    args = parser.parse_args()

    benchmark_root = normalize_benchmark_root(args.benchmark_root)
    ifeval_input = require_file(args.ifeval_input or benchmark_root / "instruction_following_eval" / "data" / "input_data.jsonl", "IFEval input")
    ifbench_input = require_file(args.ifbench_input or benchmark_root / "IFBench" / "data" / "IFBench_test.jsonl", "IFBench input")
    muldimif_input = require_file(args.muldimif_input or benchmark_root / "MulDimIF" / "Data" / "test.json", "MulDimIF input")

    ifeval_examples = maybe_limit(read_jsonl(ifeval_input), args.limit_per_benchmark)
    ifbench_examples = maybe_limit(read_jsonl(ifbench_input), args.limit_per_benchmark)
    muldimif_examples = maybe_limit(read_json_array(muldimif_input), args.limit_per_benchmark)

    rows_by_benchmark = {
        "ifeval": [build_if_row("ifeval", example, index).as_dict() for index, example in enumerate(ifeval_examples)],
        "ifbench": [build_if_row("ifbench", example, index).as_dict() for index, example in enumerate(ifbench_examples)],
        "muldimif": [build_muldimif_row(example, index).as_dict() for index, example in enumerate(muldimif_examples)],
    }
    combined_rows = rows_by_benchmark["ifeval"] + rows_by_benchmark["ifbench"] + rows_by_benchmark["muldimif"]

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in rows_by_benchmark.items():
        write_jsonl(rows, output_dir / f"{name}_val.jsonl")
        if not args.jsonl_only:
            write_parquet(rows, output_dir / f"{name}_val.parquet")

    write_jsonl(combined_rows, output_dir / f"{args.combined_name}.jsonl")
    if not args.jsonl_only:
        write_parquet(combined_rows, output_dir / f"{args.combined_name}.parquet")

    summary: dict[str, object] = {
        "ifeval_rows": len(rows_by_benchmark["ifeval"]),
        "ifbench_rows": len(rows_by_benchmark["ifbench"]),
        "muldimif_rows": len(rows_by_benchmark["muldimif"]),
        "combined_rows": len(combined_rows),
        "output_dir": str(output_dir),
        "parquet": not args.jsonl_only,
    }
    manifest = build_manifest(summary=summary, benchmark_root=benchmark_root)
    (output_dir / "instruction_benchmarks_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
