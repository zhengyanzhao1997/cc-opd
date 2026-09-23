#!/usr/bin/env python3
"""Export CC-OPD validation generations and optionally run benchmark evaluators."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


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


def write_jsonl(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_json_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def strip_thinking_trace(text: str) -> str:
    marker = "</think>"
    if marker in text:
        return text.rsplit(marker, 1)[-1].strip()
    return text


def normalize_benchmark_root(path: Path | None) -> Path | None:
    if path is None:
        return None
    if (path / "Benchmark").is_dir():
        return path / "Benchmark"
    return path


def benchmark_name(entry: dict[str, object], extra_info: dict[str, object]) -> str | None:
    raw = extra_info.get("benchmark") or entry.get("benchmark")
    if isinstance(raw, str) and raw:
        return raw.lower()
    data_source = entry.get("data_source")
    if isinstance(data_source, str) and data_source.startswith("benchmark:"):
        return data_source.split(":", 1)[1].lower()
    return None


def extract_rows(generations: list[dict[str, object]], strip_thinking: bool) -> dict[str, list[dict[str, object]]]:
    rows: dict[str, list[dict[str, object]]] = {"ifeval": [], "ifbench": [], "muldimif": []}
    for entry in generations:
        extra_info = load_json_object(entry.get("extra_info"))
        bench = benchmark_name(entry, extra_info)
        if bench not in rows:
            continue
        output = entry.get("output", "")
        response = output if isinstance(output, str) else str(output)
        if strip_thinking:
            response = strip_thinking_trace(response)

        if bench in {"ifeval", "ifbench"}:
            original_prompt = extra_info.get("original_prompt") or entry.get("original_prompt") or entry.get("input")
            if not isinstance(original_prompt, str):
                raise ValueError(f"{bench} generation is missing an exact original prompt")
            rows[bench].append({"prompt": original_prompt, "response": response})
            continue

        original_example = load_json_object(extra_info.get("original_example_json"))
        if not original_example:
            raise ValueError("MulDimIF generation is missing original_example_json")
        conversations = original_example.get("conversations")
        if not isinstance(conversations, list):
            raise ValueError("MulDimIF original example has no conversations list")
        output_example = dict(original_example)
        output_example["conversations"] = list(conversations) + [{"role": "assistant", "content": response}]
        rows[bench].append(output_example)
    return rows


def parse_eval_results(path: Path, prefix: str) -> dict[str, float]:
    if not path.is_file():
        return {}
    rows = read_jsonl(path)
    if not rows:
        return {f"{prefix}_prompt_accuracy": 0.0, f"{prefix}_instruction_accuracy": 0.0}
    prompt_hits = 0
    instruction_hits = 0
    instruction_total = 0
    for row in rows:
        if bool(row.get("follow_all_instructions")):
            prompt_hits += 1
        flags = row.get("follow_instruction_list", [])
        if isinstance(flags, list):
            instruction_hits += sum(1 for flag in flags if bool(flag))
            instruction_total += len(flags)
    metrics = {f"{prefix}_prompt_accuracy": prompt_hits / len(rows)}
    metrics[f"{prefix}_instruction_accuracy"] = instruction_hits / instruction_total if instruction_total else 0.0
    return metrics


def parse_ratio_string(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    if "=" in value:
        candidate = value.rsplit("=", 1)[-1]
    else:
        candidate = value
    try:
        return float(candidate)
    except ValueError:
        return None


def flatten_muldimif_scores(score_path: Path) -> dict[str, float]:
    if not score_path.is_file():
        return {}
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    metrics: dict[str, float] = {}
    overall = parse_ratio_string(payload.get("Overall"))
    if overall is not None:
        metrics["benchmark/muldimif/overall_accuracy"] = overall
    for group_key, metric_prefix in [
        ("constraint_pattern_list", "pattern"),
        ("constraint_difficulty_list", "difficulty"),
    ]:
        group_value = payload.get(group_key)
        if not isinstance(group_value, dict):
            continue
        for name, ratio in group_value.items():
            parsed = parse_ratio_string(ratio)
            if parsed is not None:
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_").lower()
                metrics[f"benchmark/muldimif/{metric_prefix}/{safe_name}"] = parsed
    return metrics


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    result = subprocess.run(command, cwd=str(cwd), env=env, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with code {result.returncode}: {' '.join(command)}")


def run_official_evaluators(benchmark_root: Path, response_paths: dict[str, Path], output_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{benchmark_root}:{env.get('PYTHONPATH', '')}"

    if response_paths["ifeval"].is_file():
        ifeval_out = output_dir / "ifeval_official"
        ifeval_out.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                sys.executable,
                "-m",
                "instruction_following_eval.evaluation_main",
                f"--input_data={benchmark_root / 'instruction_following_eval' / 'data' / 'input_data.jsonl'}",
                f"--input_response_data={response_paths['ifeval']}",
                f"--output_dir={ifeval_out}",
            ],
            cwd=benchmark_root,
            env=env,
        )
        metrics.update(parse_eval_results(ifeval_out / "eval_results_strict.jsonl", "benchmark/ifeval/strict"))
        metrics.update(parse_eval_results(ifeval_out / "eval_results_loose.jsonl", "benchmark/ifeval/loose"))

    if response_paths["ifbench"].is_file():
        ifbench_out = output_dir / "ifbench_official"
        ifbench_out.mkdir(parents=True, exist_ok=True)
        ifbench_cwd = benchmark_root / "IFBench"
        ifbench_env = dict(env)
        ifbench_env["PYTHONPATH"] = f"{ifbench_cwd}:{ifbench_env.get('PYTHONPATH', '')}"
        # Filter the official IFBench input to only include prompts we have
        # responses for. The vendored IFBench_test.jsonl can contain malformed
        # records (line 269 has an 8665-char prompt with embedded TSV-style
        # rows from rows 269-274) that our validation dataloader filters out
        # via filter_overlong_prompts. IFBench's run_eval.py raises KeyError
        # when an official prompt is missing from responses, so we pre-filter
        # the input file to match the response set.
        response_prompts: set[str] = set()
        with response_paths["ifbench"].open("r", encoding="utf-8") as fh:
            for raw in fh:
                stripped = raw.strip()
                if not stripped:
                    continue
                response_prompts.add(json.loads(stripped)["prompt"])
        official_input = ifbench_cwd / "data" / "IFBench_test.jsonl"
        filtered_input = ifbench_out / "IFBench_test_filtered.jsonl"
        skipped: list[int] = []
        kept_lines: list[str] = []
        with official_input.open("r", encoding="utf-8") as fh:
            for idx, raw in enumerate(fh, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    skipped.append(idx)
                    continue
                if record.get("prompt") in response_prompts:
                    kept_lines.append(stripped)
                else:
                    skipped.append(idx)
        filtered_input.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
        if skipped:
            preview = skipped[:10]
            suffix = "..." if len(skipped) > 10 else ""
            print(
                f"[ifbench] filtered {len(skipped)} official record(s) without matching response "
                f"(line indices: {preview}{suffix}); kept {len(kept_lines)}",
                file=sys.stderr,
            )
        try:
            run_command(
                [
                    sys.executable,
                    "run_eval.py",
                    f"--input_data={filtered_input}",
                    f"--input_response_data={response_paths['ifbench']}",
                    f"--output_dir={ifbench_out}",
                ],
                cwd=ifbench_cwd,
                env=ifbench_env,
            )
            metrics["benchmark/ifbench/skipped_official_prompts"] = float(len(skipped))
            metrics.update(parse_eval_results(ifbench_out / "eval_results_strict.jsonl", "benchmark/ifbench/strict"))
            metrics.update(parse_eval_results(ifbench_out / "eval_results_loose.jsonl", "benchmark/ifbench/loose"))
        except RuntimeError as exc:
            print(
                f"[ifbench] official evaluator failed: {exc}; "
                "skipping IFBench metrics for this validation cycle (other benchmarks proceed).",
                file=sys.stderr,
            )
            metrics["benchmark/ifbench/eval_failed"] = 1.0
            metrics["benchmark/ifbench/skipped_official_prompts"] = float(len(skipped))

    if response_paths["muldimif"].is_file():
        muldimif_out = output_dir / "muldimif_score.json"
        muldimif_cwd = benchmark_root / "MulDimIF"
        muldimif_env = dict(env)
        muldimif_env["PYTHONPATH"] = f"{muldimif_cwd / 'Code'}:{muldimif_env.get('PYTHONPATH', '')}"
        run_command(
            [
                sys.executable,
                "Code/evaluation/evaluation.py",
                f"--file_path={response_paths['muldimif']}",
                f"--save_path={muldimif_out}",
            ],
            cwd=muldimif_cwd,
            env=muldimif_env,
        )
        metrics.update(flatten_muldimif_scores(muldimif_out))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and optionally evaluate instruction benchmark generations.")
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--run-evaluators", action="store_true")
    parser.add_argument("--strip-thinking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generations = read_jsonl(args.generation_jsonl)
    rows = extract_rows(generations, strip_thinking=args.strip_thinking)
    response_paths = {
        "ifeval": output_dir / "ifeval_responses.jsonl",
        "ifbench": output_dir / "ifbench_responses.jsonl",
        "muldimif": output_dir / "muldimif_responses.jsonl",
    }
    for name, benchmark_rows in rows.items():
        if benchmark_rows:
            write_jsonl(benchmark_rows, response_paths[name])

    metrics: dict[str, float] = {
        "benchmark/ifeval/response_count": float(len(rows["ifeval"])),
        "benchmark/ifbench/response_count": float(len(rows["ifbench"])),
        "benchmark/muldimif/response_count": float(len(rows["muldimif"])),
    }
    benchmark_root = normalize_benchmark_root(args.benchmark_root)
    if args.run_evaluators:
        if benchmark_root is None or not benchmark_root.is_dir():
            raise FileNotFoundError(f"--run-evaluators requires a valid --benchmark-root, got {args.benchmark_root}")
        metrics.update(run_official_evaluators(benchmark_root, response_paths, output_dir))

    payload = {
        "generation_jsonl": str(args.generation_jsonl),
        "output_dir": str(output_dir),
        "response_files": {name: str(path) for name, path in response_paths.items() if path.is_file()},
        "metrics": metrics,
    }
    metrics_json = args.metrics_json or output_dir / "benchmark_metrics.json"
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
