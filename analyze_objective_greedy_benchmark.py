#!/usr/bin/env python3
"""Compare frozen RL with the objective-matched myopic greedy benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    "episode_return": ("Installation reward", 1.0),
    "restricted_mean_time_to_safety": ("Restricted mean time-to-safety (min)", -1.0),
    "unfinished_share": ("Unfinished population share", -1.0),
    "safe_share": ("Safe-completion share", 1.0),
    "congested_person_minutes": ("Congested person-minutes", -1.0),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def values(row: dict[str, str]) -> dict[str, float]:
    population = float(row["initial_population"])
    congested = float(row["congestion_person_minutes"]) * float(
        row["congested_person_minute_share"]
    )
    return {
        "episode_return": float(row["episode_return"]),
        "restricted_mean_time_to_safety": float(row["restricted_mean_time_to_safety"]),
        "unfinished_share": float(row["unfinished"]) / population,
        "safe_share": float(row["safe_completed"]) / population,
        "congested_person_minutes": congested,
    }


def action_sequence(path: Path) -> tuple[int, ...]:
    rows = read_csv(path)
    return tuple(
        int(float(row["selected_cell"]))
        for row in rows
        if int(float(row["decision_made"])) == 1
    )


def bootstrap_interval(values_: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float]:
    if values_.size == 1:
        return float(values_[0]), float(values_[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values_.size, size=(draws, values_.size))
    means = values_[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch_dir")
    parser.add_argument("--reference-stem", default="evaluation_postfix")
    parser.add_argument(
        "--benchmark-stem",
        default="evaluation_objective_greedy_sequential_20260930",
    )
    parser.add_argument("--output-dir", default="objective_greedy_analysis_20260930")
    args = parser.parse_args()

    root = Path(args.launch_dir).expanduser().resolve()
    reference = read_csv(root / f"{args.reference_stem}_episode_summary.csv")
    benchmark = read_csv(root / f"{args.benchmark_stem}_episode_summary.csv")
    rl = {
        (row["evaluation_variant_id"], int(row["city_scenario_replication"])): row
        for row in reference
        if row["deployment_strategy"] == "rl"
    }
    greedy = {
        (row["evaluation_variant_id"], int(row["city_scenario_replication"])): row
        for row in benchmark
        if row["deployment_strategy"] == "objective_greedy"
    }
    if set(rl) != set(greedy):
        missing = sorted(set(rl).symmetric_difference(greedy))
        raise RuntimeError(f"RL and objective-greedy scenario matrices differ: {missing[:4]}")

    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    paired = []
    for variant, replication in sorted(rl):
        rl_row = rl[(variant, replication)]
        greedy_row = greedy[(variant, replication)]
        for field in (
            "scenario_seed",
            "initial_observation_digest",
            "random_stream_seeds",
            "deployments_made",
            "evaluation_shelter_capacity",
            "active_shelters",
            "initial_population",
        ):
            if rl_row[field] != greedy_row[field]:
                raise RuntimeError(
                    f"Matched-interface failure for {variant}/{replication}: {field}"
                )
        population = int(float(rl_row["evaluation_population"]))
        rl_values = values(rl_row)
        greedy_values = values(greedy_row)
        rl_progress = (
            root
            / args.reference_stem
            / "state_college_pa"
            / variant
            / "policy_001"
            / "rl"
            / f"rep_{replication:03d}_local"
            / "progress.csv"
        )
        greedy_progress = (
            root
            / args.benchmark_stem
            / "state_college_pa"
            / variant
            / "benchmarks"
            / "objective_greedy"
            / f"rep_{replication:03d}_local"
            / "progress.csv"
        )
        rl_actions = action_sequence(rl_progress)
        greedy_actions = action_sequence(greedy_progress)
        if len(rl_actions) != len(greedy_actions):
            raise RuntimeError(f"Action-count mismatch for {variant}/{replication}")
        row = {
            "variant": variant,
            "population": population,
            "replication": replication,
            "scenario_seed": int(rl_row["scenario_seed"]),
            "hazard_digest_equal": bool(
                rl_row["hazard_trajectory_digest"]
                == greedy_row["hazard_trajectory_digest"]
            ),
            "rl_horizon_transitions": int(float(rl_row["horizon_transitions"])),
            "objective_greedy_horizon_transitions": int(
                float(greedy_row["horizon_transitions"])
            ),
            "action_agreement_fraction": float(
                np.mean(np.asarray(rl_actions) == np.asarray(greedy_actions))
            ),
            "rl_action_sequence": "|".join(map(str, rl_actions)),
            "objective_greedy_action_sequence": "|".join(map(str, greedy_actions)),
        }
        for metric, (_, direction) in METRICS.items():
            row[f"rl_{metric}"] = rl_values[metric]
            row[f"objective_greedy_{metric}"] = greedy_values[metric]
            row[f"rl_advantage_{metric}"] = direction * (
                rl_values[metric] - greedy_values[metric]
            )
        paired.append(row)
    write_csv(output / "paired_results.csv", paired)

    summary = []
    populations = sorted({int(row["population"]) for row in paired})
    groups = [(str(population), [row for row in paired if row["population"] == population]) for population in populations]
    groups.append(("pooled", paired))
    for group_index, (population, group) in enumerate(groups):
        for metric, (label, _) in METRICS.items():
            advantage = np.asarray(
                [row[f"rl_advantage_{metric}"] for row in group], dtype=float
            )
            low, high = bootstrap_interval(advantage, 20260930 + group_index)
            summary.append(
                {
                    "population": population,
                    "metric": metric,
                    "metric_label": label,
                    "paired_scenarios": int(advantage.size),
                    "mean_rl_advantage": float(advantage.mean()),
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "rl_wins": int(np.sum(advantage > 0.0)),
                    "ties": int(np.sum(advantage == 0.0)),
                    "objective_greedy_wins": int(np.sum(advantage < 0.0)),
                }
            )
    write_csv(output / "summary.csv", summary)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()
    for metric_index, (metric, (label, _)) in enumerate(METRICS.items()):
        axis = axes[metric_index]
        data = [
            np.asarray(
                [row[f"rl_advantage_{metric}"] for row in paired if row["population"] == population]
            )
            for population in populations
        ]
        axis.boxplot(data, tick_labels=[f"{value:,}" for value in populations], showmeans=True)
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_title(label)
        axis.set_xlabel("Pedestrians")
        axis.set_ylabel("Paired RL advantage")
        axis.grid(axis="y", alpha=0.25)
    agreement_axis = axes[-1]
    agreement = [
        [row["action_agreement_fraction"] for row in paired if row["population"] == population]
        for population in populations
    ]
    agreement_axis.boxplot(
        agreement,
        tick_labels=[f"{value:,}" for value in populations],
        showmeans=True,
    )
    agreement_axis.set(
        title="RL/objective-greedy action agreement",
        xlabel="Pedestrians",
        ylabel="Fraction of 8 decisions",
        ylim=(-0.02, 1.02),
    )
    agreement_axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Frozen v36 RL versus objective-matched myopic greedy\n"
        "Positive outcome differences favor RL"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output / "rl_vs_objective_greedy.png", dpi=180)
    plt.close(figure)

    pooled = {
        row["metric"]: row
        for row in summary
        if row["population"] == "pooled"
    }
    payload = {
        "comparison": "frozen_v36_rl_vs_objective_matched_myopic_greedy",
        "paired_scenarios": len(paired),
        "populations": populations,
        "mean_action_agreement_fraction": float(
            np.mean([row["action_agreement_fraction"] for row in paired])
        ),
        "pooled": pooled,
        "interpretation_boundary": (
            "Outperforming this benchmark supports value from temporal or graph context, "
            "but does not by itself identify which feature caused the advantage."
        ),
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
