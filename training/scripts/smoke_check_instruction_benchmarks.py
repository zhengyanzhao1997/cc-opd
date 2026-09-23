#!/usr/bin/env python3
"""Smoke test benchmark prep and response export without official evaluators."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AssertionError(f"non-object row in {path}")
                rows.append(value)
    return rows


def run(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cc_opd_benchmark_smoke_") as tmp:
        root = Path(tmp)
        benchmark_root = root / "Benchmark"
        out_dir = root / "prepared"
        eval_dir = root / "eval"

        write_jsonl(
            [
                {
                    "key": 1000,
                    "prompt": "Write a haiku without commas.",
                    "instruction_id_list": ["punctuation:no_comma"],
                    "kwargs": [{}],
                }
            ],
            benchmark_root / "instruction_following_eval" / "data" / "input_data.jsonl",
        )
        write_jsonl(
            [
                {
                    "key": "ifb-0",
                    "prompt": "Mention kaleidoscope exactly once. ",
                    "instruction_id_list": ["count:keywords_multiple"],
                    "kwargs": [{"keyword1": "kaleidoscope"}],
                }
            ],
            benchmark_root / "IFBench" / "data" / "IFBench_test.jsonl",
        )
        muldimif_data = [
            {
                "id": "mul-0",
                "conversations": [{"role": "user", "content": "Return XML with one element."}],
                "constraints": [["XML", "Number of attributes", "Each XML element must have no more than three attributes"]],
                "constraint_pattern": "Listing",
                "difficulty": "Level 1",
            }
        ]
        muldimif_path = benchmark_root / "MulDimIF" / "Data" / "test.json"
        muldimif_path.parent.mkdir(parents=True, exist_ok=True)
        muldimif_path.write_text(json.dumps(muldimif_data, ensure_ascii=False), encoding="utf-8")

        run(
            [
                sys.executable,
                str(SCRIPT_DIR / "prepare_instruction_benchmarks.py"),
                "--benchmark-root",
                str(benchmark_root),
                "--output-dir",
                str(out_dir),
            ]
        )
        rows = read_jsonl(out_dir / "instruction_benchmarks_val.jsonl")
        if len(rows) != 3:
            raise AssertionError(f"expected 3 prepared benchmark rows, got {len(rows)}")
        by_source = {str(row["data_source"]): row for row in rows}
        ifbench_extra = by_source["benchmark:ifbench"].get("extra_info")
        if not isinstance(ifbench_extra, dict):
            raise AssertionError("IFBench extra_info is not a dict")
        if ifbench_extra.get("original_prompt") != "Mention kaleidoscope exactly once. ":
            raise AssertionError("IFBench trailing prompt whitespace was not preserved")

        generations: list[dict[str, object]] = []
        for row in rows:
            extra_info = row["extra_info"]
            if not isinstance(extra_info, dict):
                raise AssertionError("extra_info must stay as a dict")
            generations.append(
                {
                    "input": extra_info["original_prompt"],
                    "output": "<think>scratch</think>final answer",
                    "score": 0.0,
                    "data_source": row["data_source"],
                    "extra_info": extra_info,
                }
            )
        gen_path = root / "generations.jsonl"
        write_jsonl(generations, gen_path)

        run(
            [
                sys.executable,
                str(SCRIPT_DIR / "run_instruction_benchmark_eval.py"),
                "--generation-jsonl",
                str(gen_path),
                "--output-dir",
                str(eval_dir),
                "--metrics-json",
                str(eval_dir / "metrics.json"),
            ]
        )

        ifeval_rows = read_jsonl(eval_dir / "ifeval_responses.jsonl")
        if ifeval_rows[0] != {"prompt": "Write a haiku without commas.", "response": "final answer"}:
            raise AssertionError("IFEval response export is not exact")
        ifbench_rows = read_jsonl(eval_dir / "ifbench_responses.jsonl")
        if ifbench_rows[0]["prompt"] != "Mention kaleidoscope exactly once. ":
            raise AssertionError("IFBench response export lost exact prompt")
        muldimif_rows = read_jsonl(eval_dir / "muldimif_responses.jsonl")
        conversations = muldimif_rows[0].get("conversations")
        if not isinstance(conversations, list):
            raise AssertionError("MulDimIF conversations is not a list")
        if conversations[-1] != {"role": "assistant", "content": "final answer"}:
            raise AssertionError("MulDimIF response was not appended as final assistant turn")

        metrics_payload = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
        if not isinstance(metrics_payload, dict):
            raise AssertionError("metrics payload is not a dict")
        counts = metrics_payload.get("metrics")
        if not isinstance(counts, dict):
            raise AssertionError("metrics field is not a dict")
        expected_counts = {
            "benchmark/ifeval/response_count": 1.0,
            "benchmark/ifbench/response_count": 1.0,
            "benchmark/muldimif/response_count": 1.0,
        }
        if counts != expected_counts:
            raise AssertionError(f"unexpected benchmark counts: {counts}")
        print(json.dumps({"prepared_rows": len(rows), "metrics": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
