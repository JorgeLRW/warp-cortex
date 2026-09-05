from __future__ import annotations

import argparse
import json
from typing import List

from .runner import run_scorecard
from .schema import ScorecardConfig


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="warp-cortex-scorecard", description="Run Warp Cortex backend scorecards.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a scorecard and write artifacts.")
    run_parser.add_argument("--suite", default="builtin", help="Suite label to stamp into artifacts.")
    run_parser.add_argument("--trace-file", default="", help="JSON or JSONL trace file. Defaults to built-in smoke traces.")
    run_parser.add_argument("--out-dir", default="local_artifacts/scorecards/latest", help="Output artifact directory.")
    run_parser.add_argument("--candidate", action="append", default=[], help="Candidate: local, api, hybrid, deterministic, deterministic_bad, hybrid_demo.")
    run_parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Local HF model ID or path.")
    run_parser.add_argument("--api-model", default="gpt-4o-mini", help="OpenAI-compatible API model.")
    run_parser.add_argument("--hf-home", default="", help="Hugging Face cache root.")
    run_parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    run_parser.add_argument("--max-tokens", type=int, default=96)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--timeout-seconds", type=float, default=60.0)
    run_parser.add_argument("--limit", type=int, default=0, help="Limit trace cases for smoke runs.")
    run_parser.add_argument("--online", action="store_true", help="Allow Hugging Face network lookups instead of offline cache-only mode.")
    run_parser.add_argument("--evidence-db", default="", help="SQLite shared-manifold evidence DB path. Defaults to OUT/evidence.sqlite.")
    run_parser.add_argument("--json", action="store_true", help="Print the full scorecard JSON to stdout.")

    list_parser = subparsers.add_parser("list-suites", help="List built-in suites and candidates.")
    list_parser.set_defaults(command="list-suites")

    args = parser.parse_args(argv)
    if args.command == "list-suites":
        print("Built-in suite: builtin")
        print("Candidates: local, api, hybrid, deterministic, deterministic_bad, hybrid_demo")
        return 0
    if args.command != "run":
        parser.print_help()
        return 2

    config = ScorecardConfig(
        suite=args.suite,
        out_dir=args.out_dir,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
        device=args.device,
        hf_home=args.hf_home,
        offline=not args.online,
        evidence_db=args.evidence_db,
        limit=args.limit,
    )
    report = run_scorecard(
        config=config,
        candidate_names=args.candidate,
        trace_file=args.trace_file,
        local_model=args.model,
        api_model=args.api_model,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Scorecard: {report['artifacts']['scorecard_md']}")
        for name, summary in report["aggregate"]["candidate_summary"].items():
            print(
                f"{name}: pass_rate={summary['pass_rate']:.2f} "
                f"remote_call_rate={summary['remote_call_rate']:.2f} "
                f"fallback_rate={summary['fallback_rate']:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())