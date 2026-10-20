#!/usr/bin/env python3
"""Re-evaluate benchmark policies against a frozen, already-scored RL panel.

The frozen RL rows are copied from a completed evaluation ledger.  Only the
benchmark policies are rerun, on the exact same scenario and policy seeds and
with the recorded final-stage simulator overrides.  This is the appropriate
transaction when a benchmark's information contract changes but the learned
policy and physical simulator do not.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import _run_episode, _write_csv  # noqa: E402
from DecisionInterface import BENCHMARK_POLICY_CONTRACTS  # noqa: E402


DEFAULT_STRATEGIES = (
    "objective_greedy",
    "forecast_greedy",
    "deadline_greedy",
    "random",
)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_int(row: dict, key: str) -> int:
    return int(float(row[key]))


def _as_float(row: dict, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _bootstrap_interval(values: np.ndarray, *, seed: int, draws: int) -> tuple[float, float]:
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(draws, values.size), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(sample, (0.025, 0.975)))


def _summaries(
    rows: list[dict],
    *,
    launch_seed: int,
    draws: int,
) -> list[dict]:
    by_strategy: dict[str, dict[int, dict]] = {}
    for row in rows:
        strategy = str(row["deployment_strategy"])
        replication = _as_int(row, "city_scenario_replication")
        by_strategy.setdefault(strategy, {})[replication] = row
    rl = by_strategy.get("rl", {})
    if not rl:
        raise RuntimeError("Combined ledger has no frozen RL rows")
    output = []
    for strategy in ("rl", *DEFAULT_STRATEGIES):
        strategy_rows = by_strategy.get(strategy, {})
        if set(strategy_rows) != set(rl):
            raise RuntimeError(
                f"{strategy} does not cover the same matched replications as RL"
            )
        ordered = [strategy_rows[index] for index in sorted(rl)]
        costs = np.asarray(
            [_as_float(row, "objective_normalized_cost") for row in ordered],
            dtype=np.float64,
        )
        unfinished = np.asarray(
            [_as_float(row, "unfinished") for row in ordered], dtype=np.float64
        )
        casualties = np.asarray(
            [_as_float(row, "casualty") for row in ordered], dtype=np.float64
        )
        summary = {
            "strategy": strategy,
            "replications": int(costs.size),
            "mean_normalized_cost": float(costs.mean()),
            "standard_error_normalized_cost": float(
                costs.std(ddof=1) / math.sqrt(costs.size)
            ),
            "mean_unfinished": float(unfinished.mean()),
            "mean_realized_casualties": float(casualties.mean()),
            "paired_rl_minus_benchmark_cost": 0.0,
            "paired_improvement_of_rl_percent": 0.0,
            "rl_better_fraction": float("nan"),
            "paired_difference_ci_low": 0.0,
            "paired_difference_ci_high": 0.0,
        }
        if strategy != "rl":
            rl_costs = np.asarray(
                [
                    _as_float(rl[index], "objective_normalized_cost")
                    for index in sorted(rl)
                ],
                dtype=np.float64,
            )
            differences = rl_costs - costs
            low, high = _bootstrap_interval(
                differences,
                seed=int(launch_seed) ^ sum(ord(char) for char in strategy),
                draws=draws,
            )
            summary.update(
                {
                    "paired_rl_minus_benchmark_cost": float(differences.mean()),
                    "paired_improvement_of_rl_percent": float(
                        100.0 * (costs.mean() - rl_costs.mean()) / costs.mean()
                    ),
                    "rl_better_fraction": float(np.mean(rl_costs < costs)),
                    "paired_difference_ci_low": low,
                    "paired_difference_ci_high": high,
                }
            )
        output.append(summary)
    return output


def _write_markdown(path: Path, summaries: list[dict]) -> None:
    lines = [
        "# Observation-matched benchmark re-evaluation",
        "",
        "Lower normalized cost is better. Negative `RL − benchmark` favors RL.",
        "",
        "| Policy | Mean cost | Mean unfinished | RL − benchmark | RL improvement | RL win rate | 95% paired bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        ci = (
            "—"
            if row["strategy"] == "rl"
            else f"[{row['paired_difference_ci_low']:.4f}, {row['paired_difference_ci_high']:.4f}]"
        )
        win = (
            "—"
            if row["strategy"] == "rl"
            else f"{100.0 * row['rl_better_fraction']:.1f}%"
        )
        diff = (
            "—"
            if row["strategy"] == "rl"
            else f"{row['paired_rl_minus_benchmark_cost']:.4f}"
        )
        improvement = (
            "—"
            if row["strategy"] == "rl"
            else f"{row['paired_improvement_of_rl_percent']:.1f}%"
        )
        lines.append(
            f"| {row['strategy']} | {row['mean_normalized_cost']:.4f} | "
            f"{row['mean_unfinished']:.1f} | {diff} | {improvement} | {win} | {ci} |"
        )
    lines.extend(
        (
            "",
            "The RL rows are the immutable trajectories from the source run; only benchmark rows were regenerated.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
    )
    parser.add_argument("--machine", default="local")
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    args = parser.parse_args(argv)

    source = Path(args.source_run).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "experiment_manifest.json").read_text())
    source_rows = _read_csv(source / "evaluation_episode_summary.csv")
    rl_rows = [row for row in source_rows if row["deployment_strategy"] == "rl"]
    if not rl_rows:
        raise RuntimeError("Source evaluation does not contain RL rows")
    strategies = tuple(
        strategy.strip() for strategy in args.strategies.split(",") if strategy.strip()
    )
    if strategies != DEFAULT_STRATEGIES:
        raise ValueError(
            "This registered re-evaluation must use exactly "
            + ",".join(DEFAULT_STRATEGIES)
        )

    city_ids = tuple(manifest["city_ids"])
    if len(city_ids) != 1:
        raise ValueError("This re-evaluation script expects one source city")
    city_id = city_ids[0]
    overrides = dict(manifest["effective_overrides_by_city"][city_id])
    variants = list(manifest.get("evaluation_variant_schedule", []))
    if len(variants) != 1:
        raise ValueError("Source run must have one final evaluation variant")
    overrides.update(variants[0]["overrides"])

    benchmark_path = output / "benchmark_episode_summary.csv"
    benchmark_rows = _read_csv(benchmark_path) if benchmark_path.exists() else []
    completed = {
        (row["deployment_strategy"], _as_int(row, "city_scenario_replication"))
        for row in benchmark_rows
    }
    for rl_row in sorted(
        rl_rows, key=lambda row: _as_int(row, "city_scenario_replication")
    ):
        replication = _as_int(rl_row, "city_scenario_replication")
        for strategy in strategies:
            if (strategy, replication) in completed:
                continue
            print(
                f"[REEVAL] pair={replication}/{len(rl_rows)} strategy={strategy}",
                flush=True,
            )
            row = _run_episode(
                replication=replication,
                machine=args.machine,
                phase=str(
                    output
                    / "work"
                    / f"pair_{replication:03d}"
                    / strategy
                ),
                strategy=strategy,
                train_mode=False,
                scenario_seed=_as_int(rl_row, "scenario_seed"),
                policy_seed=_as_int(rl_row, "policy_seed"),
                checkpoint_path=str(
                    source / "policies" / "policy_001" / "regional_policy.pt"
                ),
                diagnostics_path=str(output / "unused_diagnostics.csv"),
                overrides=overrides,
                visualization_enabled=False,
            )
            row.update(
                {
                    "policy_replication": 0,
                    "city_id": city_id,
                    "city_scale_rank": _as_int(rl_row, "city_scale_rank"),
                    "city_scenario_replication": replication,
                    "evaluation_variant_id": rl_row["evaluation_variant_id"],
                    "evaluation_population": _as_int(
                        rl_row, "evaluation_population"
                    ),
                }
            )
            benchmark_rows.append(row)
            completed.add((strategy, replication))
            _write_csv(str(benchmark_path), benchmark_rows)

    combined_rows = [*rl_rows, *benchmark_rows]
    _write_csv(str(output / "evaluation_episode_summary.csv"), combined_rows)
    summaries = _summaries(
        combined_rows,
        launch_seed=int(manifest["launch_seed"]),
        draws=int(args.bootstrap_draws),
    )
    _write_csv(str(output / "summary.csv"), summaries)
    _write_markdown(output / "summary.md", summaries)
    audit = {
        "schema_version": 1,
        "source_run": str(source),
        "source_checkpoint": str(
            source / "policies" / "policy_001" / "regional_policy.pt"
        ),
        "frozen_rl_rows": len(rl_rows),
        "rerun_benchmark_rows": len(benchmark_rows),
        "scenario_seeds": [_as_int(row, "scenario_seed") for row in rl_rows],
        "policy_seeds": [_as_int(row, "policy_seed") for row in rl_rows],
        "strategies": list(strategies),
        "benchmark_policy_contracts": {
            strategy: BENCHMARK_POLICY_CONTRACTS[strategy]
            for strategy in strategies
        },
        "isolation": (
            "RL trajectories and physical objective are frozen; only benchmark "
            "policy decisions are regenerated under the current information contracts"
        ),
    }
    (output / "reevaluation_manifest.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output / "summary.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
