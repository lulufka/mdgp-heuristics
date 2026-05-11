from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the heuristics script multiple times with independent random "
            "seeds."
        )
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results/random_runs")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name of the dataset for the output files (defaults to folder name)",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).with_name("run_heuristics.py")
    project_root = script_path.parent.parent
    src_path = project_root / "src"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(src_path)
    )

    manifest_rows = []

    for run_idx in range(args.runs):
        run_number = run_idx + 1
        seed = args.base_seed + run_idx
        run_results_dir = results_dir / f"random_run_{run_number:02d}_seed_{seed}"

        dataset_name = (
            f"{args.dataset_name}_run_{run_number:02d}"
            if args.dataset_name is not None
            else None
        )

        cmd = [
            sys.executable,
            str(script_path),
            "--data-dir",
            args.data_dir,
            "--results-dir",
            str(run_results_dir),
            "--random-seed",
            str(seed),
        ]

        if dataset_name is not None:
            cmd.extend(["--dataset-name", dataset_name])

        print(
            f"\n=== Independent run {run_number}/{args.runs} | seed={seed} ===",
            flush=True,
        )
        subprocess.run(cmd, check=True, env=env)

        manifest_rows.append(
            {
                "run": run_number,
                "seed": seed,
                "results_dir": str(run_results_dir),
                "dataset_name": dataset_name or "",
            }
        )

    manifest_path = results_dir / "random_runs_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "seed", "results_dir", "dataset_name"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nSaved manifest to {manifest_path.resolve()}")



if __name__ == "__main__":
    main()
