#!/usr/bin/env python3
"""Audit a recurrent PPO credit-assignment validation campaign.

The analysis preserves the experimental units used by the campaign: training
trends are first computed within policy seed, and held-out effects use the
paired policy/scenario analysis emitted by ``multicity_backtest.py``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from generate_experiment_results_chapter import fallacy_scan


REWARD_FIELDS = (
    "episode_return",
    "safe_completion_reward",
    "casualty_penalty",
    "evacuation_time_penalty",
    "hazard_exposure_penalty",
)
CRITIC_FIELDS = (
    "value_loss",
    "value_loss_safe_completion",
    "value_loss_casualty",
    "value_loss_evacuation_time",
    "value_loss_hazard_exposure",
    "explained_variance",
    "explained_variance_safe_completion",
    "explained_variance_casualty",
    "explained_variance_evacuation_time",
    "explained_variance_hazard_exposure",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {field}: {row[field]!r}")
    return value


def _summary(values: Iterable[float]) -> dict:
    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary requires finite nonempty values")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sample_standard_deviation": float(np.std(array, ddof=1))
        if array.size > 1
        else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _exact_sign_flip_p(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    observed = abs(float(np.mean(array)))
    statistics = [
        abs(float(np.mean(array * np.asarray(signs, dtype=float))))
        for signs in itertools.product((-1.0, 1.0), repeat=array.size)
    ]
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def _seed_bootstrap_ci(values: Iterable[float], *, draws: int, seed: int) -> list[float]:
    array = np.asarray(tuple(values), dtype=float)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        means[draw] = float(np.mean(rng.choice(array, size=array.size, replace=True)))
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _training_analysis(rows: list[dict[str, str]], *, draws: int, seed: int) -> dict:
    policies = sorted({int(row["policy_replication"]) for row in rows})
    if len(rows) != 96 or policies != [1, 2, 3]:
        raise ValueError(f"expected 96 rows for policies 1..3, got {len(rows)} {policies}")
    policy_results = []
    endpoint_changes = []
    slopes = []
    for policy in policies:
        subset = sorted(
            (row for row in rows if int(row["policy_replication"]) == policy),
            key=lambda row: int(row["replication"]),
        )
        if len(subset) != 32:
            raise ValueError(f"policy {policy} has {len(subset)} training rows")
        blocks = []
        for start in range(0, 32, 8):
            block_rows = subset[start : start + 8]
            blocks.append(
                {
                    "block": start // 8 + 1,
                    "episodes": [start + 1, start + 8],
                    **{
                        field: float(np.mean([_number(row, field) for row in block_rows]))
                        for field in REWARD_FIELDS
                    },
                    "casualty_mean": float(
                        np.mean([_number(row, "casualty") for row in block_rows])
                    ),
                    "casualty_nonzero_episodes": int(
                        sum(_number(row, "casualty") > 0.0 for row in block_rows)
                    ),
                }
            )
        returns = np.asarray([block["episode_return"] for block in blocks])
        endpoint_change = float(returns[-1] - returns[0])
        slope = float(np.polyfit(np.arange(1, 5, dtype=float), returns, 1)[0])
        endpoint_changes.append(endpoint_change)
        slopes.append(slope)
        policy_results.append(
            {
                "policy_replication": policy,
                "blocks": blocks,
                "final_minus_initial_block_return": endpoint_change,
                "block_return_linear_slope": slope,
            }
        )

    pooled_blocks = []
    for block in range(4):
        pooled_blocks.append(
            {
                "block": block + 1,
                **{
                    field: float(
                        np.mean(
                            [result["blocks"][block][field] for result in policy_results]
                        )
                    )
                    for field in REWARD_FIELDS
                },
                "casualty_mean": float(
                    np.mean(
                        [
                            result["blocks"][block]["casualty_mean"]
                            for result in policy_results
                        ]
                    )
                ),
            }
        )
    total_casualties = int(sum(_number(row, "casualty") for row in rows))
    reward_variation = {
        field: _summary(_number(row, field) for row in rows) for field in REWARD_FIELDS
    }
    return {
        "episodes": len(rows),
        "policy_replications": len(policies),
        "policy_results": policy_results,
        "pooled_seed_equal_blocks": pooled_blocks,
        "final_minus_initial_block_return_by_policy": endpoint_changes,
        "mean_final_minus_initial_block_return": float(np.mean(endpoint_changes)),
        "seed_bootstrap_95_ci_for_endpoint_change": _seed_bootstrap_ci(
            endpoint_changes,
            draws=draws,
            seed=seed,
        ),
        "exact_sign_flip_p_for_endpoint_change": _exact_sign_flip_p(endpoint_changes),
        "positive_endpoint_policy_count": int(sum(value > 0.0 for value in endpoint_changes)),
        "block_return_slope_by_policy": slopes,
        "mean_block_return_slope": float(np.mean(slopes)),
        "reward_variation": reward_variation,
        "total_casualties": total_casualties,
        "mean_casualties_per_episode": float(total_casualties / len(rows)),
        "episodes_with_casualty": int(sum(_number(row, "casualty") > 0.0 for row in rows)),
        "casualty_incidence_per_person_episode": float(total_casualties / (len(rows) * 3000)),
        "maximum_absolute_reward_accounting_gap": float(
            max(abs(_number(row, "reward_accounting_gap")) for row in rows)
        ),
        "positive_trend_criterion_passed": bool(np.mean(endpoint_changes) > 0.0),
    }


def _critic_analysis(run_dir: Path) -> dict:
    policy_results = []
    for policy in (1, 2, 3):
        path = run_dir / "policies" / f"policy_{policy:03d}" / "ppo_diagnostics.csv"
        rows = _read_csv(path)
        updates = [row for row in rows if _number(row, "optimizer_updated") == 1.0]
        if len(updates) != 4:
            raise ValueError(f"policy {policy} has {len(updates)} optimizer events")
        blocks = []
        for block, row in enumerate(updates, start=1):
            blocks.append(
                {
                    "block": block,
                    **{field: _number(row, field) for field in CRITIC_FIELDS},
                    "approximate_kl": _number(row, "approximate_kl"),
                    "clip_fraction": _number(row, "clip_fraction"),
                    "gradient_norm": _number(row, "gradient_norm"),
                    "residual_rms": _number(row, "residual_rms"),
                    "sequence_episodes": _number(row, "sequence_episodes"),
                    "observation_frames": _number(row, "observation_frames"),
                    "mean_observation_history": _number(row, "mean_observation_history"),
                    "mean_credit_duration": _number(row, "mean_credit_duration"),
                    "maximum_credit_duration": _number(row, "maximum_credit_duration"),
                }
            )
        policy_results.append(
            {
                "policy_replication": policy,
                "blocks": blocks,
                "first_to_final": {
                    field: float(blocks[-1][field] - blocks[0][field])
                    for field in CRITIC_FIELDS
                },
            }
        )

    block_means = []
    for block in range(4):
        block_means.append(
            {
                "block": block + 1,
                **{
                    field: float(
                        np.mean([result["blocks"][block][field] for result in policy_results])
                    )
                    for field in (*CRITIC_FIELDS, "gradient_norm", "residual_rms")
                },
            }
        )
    total_loss_changes = [
        result["first_to_final"]["value_loss"] for result in policy_results
    ]
    casualty_loss_changes = [
        result["first_to_final"]["value_loss_casualty"] for result in policy_results
    ]
    return {
        "policy_results": policy_results,
        "seed_equal_block_means": block_means,
        "total_value_loss_first_to_final_by_policy": total_loss_changes,
        "casualty_value_loss_first_to_final_by_policy": casualty_loss_changes,
        "policies_with_lower_final_total_value_loss": int(
            sum(value < 0.0 for value in total_loss_changes)
        ),
        "policies_with_lower_final_casualty_value_loss": int(
            sum(value < 0.0 for value in casualty_loss_changes)
        ),
        "all_batches_whole_episode_sequences": bool(
            all(
                block["sequence_episodes"] == 8.0
                for result in policy_results
                for block in result["blocks"]
            )
        ),
        "minimum_mean_observation_history": float(
            min(
                block["mean_observation_history"]
                for result in policy_results
                for block in result["blocks"]
            )
        ),
        "maximum_credit_duration": float(
            max(
                block["maximum_credit_duration"]
                for result in policy_results
                for block in result["blocks"]
            )
        ),
        "maximum_approximate_kl": float(
            max(
                block["approximate_kl"]
                for result in policy_results
                for block in result["blocks"]
            )
        ),
        "maximum_clip_fraction": float(
            max(
                block["clip_fraction"]
                for result in policy_results
                for block in result["blocks"]
            )
        ),
        "critic_improvement_criterion_passed": bool(
            all(value < 0.0 for value in total_loss_changes)
            and all(value < 0.0 for value in casualty_loss_changes)
        ),
    }


def _evaluation_analysis(
    evaluation_rows: list[dict[str, str]],
    paired_rows: list[dict[str, str]],
) -> dict:
    if len(evaluation_rows) != 32:
        raise ValueError(f"expected 32 evaluation rows, got {len(evaluation_rows)}")
    heuristic = {
        int(row["replication"]): row
        for row in evaluation_rows
        if row["deployment_strategy"] == "heuristic"
    }
    rl_rows = [row for row in evaluation_rows if row["deployment_strategy"] == "rl"]
    if len(heuristic) != 8 or len(rl_rows) != 24:
        raise ValueError("expected 8 heuristic and 24 RL evaluation rows")
    by_policy = []
    for policy in (1, 2, 3):
        rows = [row for row in rl_rows if int(row["policy_replication"]) == policy]
        return_differences = [
            _number(row, "episode_return")
            - _number(heuristic[int(row["replication"])], "episode_return")
            for row in rows
        ]
        casualty_avoidance = [
            _number(heuristic[int(row["replication"])], "casualty")
            - _number(row, "casualty")
            for row in rows
        ]
        by_policy.append(
            {
                "policy_replication": policy,
                "mean_return_difference": float(np.mean(return_differences)),
                "return_differences": return_differences,
                "mean_casualties_avoided": float(np.mean(casualty_avoidance)),
                "mean_heuristic_agreement_rate": float(
                    np.mean([_number(row, "heuristic_agreement_rate") for row in rows])
                ),
            }
        )
    macro = {
        row["metric"]: row
        for row in paired_rows
        if row["scope"] == "macro_all_cities"
    }
    objective = macro["episode_return"]
    casualty = macro["casualty"]
    return_improvement = float(objective["mean_rl_improvement"])
    casualty_improvement = float(casualty["mean_rl_improvement"])
    return_ci = [
        float(objective["bootstrap_95_ci_low"]),
        float(objective["bootstrap_95_ci_high"]),
    ]
    return {
        "rows": len(evaluation_rows),
        "rl_rows": len(rl_rows),
        "heuristic_rows": len(heuristic),
        "by_policy": by_policy,
        "macro_metrics": {
            metric: {
                key: float(row[key])
                for key in (
                    "rl_mean",
                    "heuristic_mean",
                    "mean_rl_improvement",
                    "bootstrap_95_ci_low",
                    "bootstrap_95_ci_high",
                    "two_sided_randomization_p",
                    "rl_win_rate",
                    "tie_rate",
                )
            }
            for metric, row in macro.items()
        },
        "maximum_absolute_reward_accounting_gap": float(
            max(abs(_number(row, "reward_accounting_gap")) for row in evaluation_rows)
        ),
        "held_out_positive_return_criterion_passed": bool(return_improvement > 0.0),
        "held_out_strong_return_criterion_passed": bool(return_ci[0] > 0.0),
        "held_out_casualty_nonworsening_criterion_passed": bool(
            casualty_improvement >= 0.0
        ),
    }


def _post_training_mechanism_analysis(run_dir: Path) -> dict:
    policies = []
    for policy in (1, 2, 3):
        path = (
            run_dir
            / "policies"
            / f"policy_{policy:03d}"
            / "post_training_credit_audit.json"
        )
        result = _read_json(path)
        recurrent = result["recurrent_gradient"]
        policies.append(
            {
                "policy_replication": policy,
                "path": str(path),
                "sha256": _sha256(path),
                "passed": bool(result["passed"]),
                "actor_full_minimum_gradient_norm": float(
                    recurrent["actor_full_minimum_gradient_norm"]
                ),
                "actor_reset_maximum_gradient_norm": float(
                    recurrent["actor_reset_maximum_gradient_norm"]
                ),
                "critic_full_minimum_gradient_norm": float(
                    recurrent["critic_full_minimum_gradient_norm"]
                ),
                "critic_reset_maximum_gradient_norm": float(
                    recurrent["critic_reset_maximum_gradient_norm"]
                ),
            }
        )
    return {
        "policies": policies,
        "all_final_checkpoints_passed": bool(all(row["passed"] for row in policies)),
        "minimum_actor_full_gradient_norm": float(
            min(row["actor_full_minimum_gradient_norm"] for row in policies)
        ),
        "minimum_critic_full_gradient_norm": float(
            min(row["critic_full_minimum_gradient_norm"] for row in policies)
        ),
        "maximum_actor_reset_gradient_norm": float(
            max(row["actor_reset_maximum_gradient_norm"] for row in policies)
        ),
        "maximum_critic_reset_gradient_norm": float(
            max(row["critic_reset_maximum_gradient_norm"] for row in policies)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mechanism-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    if args.bootstrap_draws <= 0:
        parser.error("--bootstrap-draws must be positive")
    return args


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    paths = {
        "manifest": run_dir / "experiment_manifest.json",
        "training": run_dir / "training_episode_summary.csv",
        "evaluation": run_dir / "evaluation_episode_summary.csv",
        "paired": run_dir / "paired_comparison_by_city.csv",
        "interface": run_dir / "interface_parity.json",
        "convergence": run_dir / "training_convergence_diagnostics.json",
        "mechanism": args.mechanism_json.resolve(),
    }
    training_rows = _read_csv(paths["training"])
    evaluation_rows = _read_csv(paths["evaluation"])
    paired_rows = _read_csv(paths["paired"])
    mechanism = _read_json(paths["mechanism"])
    interface = _read_json(paths["interface"])
    convergence = _read_json(paths["convergence"])
    training = _training_analysis(
        training_rows,
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    critic = _critic_analysis(run_dir)
    evaluation = _evaluation_analysis(evaluation_rows, paired_rows)
    post_training_mechanism = _post_training_mechanism_analysis(run_dir)
    fallacies = [
        {"name": name, "severity": severity, "finding": finding}
        for name, severity, finding in fallacy_scan(paired_rows)
    ]
    criteria = {
        "mechanism_gates_passed": bool(mechanism["passed"]),
        "all_final_checkpoint_gradient_audits_passed": bool(
            post_training_mechanism["all_final_checkpoints_passed"]
        ),
        "complete_training_accounting_passed": bool(
            training["maximum_absolute_reward_accounting_gap"] <= 1e-6
        ),
        "positive_training_reward_trend_passed": bool(
            training["positive_trend_criterion_passed"]
        ),
        "critic_improvement_passed": bool(
            critic["critic_improvement_criterion_passed"]
        ),
        "positive_held_out_return_passed": bool(
            evaluation["held_out_positive_return_criterion_passed"]
        ),
        "strong_held_out_return_passed": bool(
            evaluation["held_out_strong_return_criterion_passed"]
        ),
        "held_out_casualty_nonworsening_passed": bool(
            evaluation["held_out_casualty_nonworsening_criterion_passed"]
        ),
    }
    criteria["reward_improvement_supported"] = bool(
        criteria["positive_training_reward_trend_passed"]
        and criteria["positive_held_out_return_passed"]
    )
    criteria["learned_credit_improvement_supported"] = bool(
        criteria["mechanism_gates_passed"]
        and criteria["critic_improvement_passed"]
    )
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "origin_date": "2026-09-19",
            "verification_status": "ANALYZED",
            "version_label": "credit_validation_analysis_v1",
        },
        "run_dir": str(run_dir),
        "mechanism": mechanism,
        "training": training,
        "critic": critic,
        "evaluation": evaluation,
        "post_training_mechanism": post_training_mechanism,
        "interface_verified": bool(interface["verified"]),
        "training_converged": bool(convergence["all_policies_converged"]),
        "criteria": criteria,
        "fallacy_scan": fallacies,
        "fallacy_scan_coverage": f"{len(fallacies)}/11",
        "source_hashes": {name: _sha256(path) for name, path in paths.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(criteria, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
