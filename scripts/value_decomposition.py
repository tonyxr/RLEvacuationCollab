#!/usr/bin/env python3
"""Matched-tape v53r regime value decomposition.

Hindsight rollout is emitted only as ``hindsight_ceiling``.  It is never
included in the deployable-policy set or described as an operational policy.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import _run_episode
from StressScenarios import (
    expand_stress_families,
    out_of_support_keys,
    paired_dose_slope,
    training_support,
)


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _base_overrides(spec):
    curriculum = _load(ROOT / spec["base_curriculum"])
    overrides = dict(curriculum["learner_overrides"])
    overrides.update(curriculum["stages"][0]["variants"][0]["overrides"])
    profiles = _load(ROOT / "config" / "city_profiles.json")
    city = next(
        item for item in profiles["cities"]
        if item["city_id"] == "state_college_pa"
    )
    overrides.update({
        "cityID": city["city_id"],
        "address": city["address"],
        "mapQueryMode": "point",
        "mapCenterLat": city["center_lat"],
        "mapCenterLon": city["center_lon"],
        "mapRadiusM": city["radius_m"],
    })
    overrides.update(spec.get("screening_overrides", {}))
    return overrides


def _seed(scenario_index, tape_index):
    return int(
        np.random.SeedSequence((0x56353352, scenario_index, tape_index))
        .generate_state(1, dtype=np.uint32)[0]
    )


def _paired_se(values):
    values = np.asarray(values, dtype=np.float64)
    return 0.0 if values.size < 2 else float(values.std(ddof=1) / np.sqrt(values.size))


def _bootstrap_seed_ci(values, *, seed, draws=10_000):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(draws), values.size))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def _sign_flip_pvalue(values, *, seed, draws=100_000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    observed = abs(float(values.mean()))
    if values.size <= 16:
        signs = np.asarray(
            list(itertools.product((-1.0, 1.0), repeat=values.size)),
            dtype=np.float64,
        )
    else:
        rng = np.random.default_rng(int(seed))
        signs = rng.choice((-1.0, 1.0), size=(int(draws), values.size))
    randomized = np.abs(np.mean(signs * values[None, :], axis=1))
    return float((np.count_nonzero(randomized >= observed - 1e-15) + 1) / (len(randomized) + 1))


def _holm_adjust(summaries):
    eligible = [
        (index, float(row["rl_vs_best_p_value"]))
        for index, row in enumerate(summaries)
        if np.isfinite(row.get("rl_vs_best_p_value", math.nan))
    ]
    ordered = sorted(eligible, key=lambda item: item[1])
    running = 0.0
    total = len(ordered)
    for rank, (index, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_value))
        summaries[index]["rl_vs_best_holm_p"] = running
    for row in summaries:
        row.setdefault("rl_vs_best_holm_p", math.nan)


def _write_csv(path, rows):
    if not rows:
        return
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _tune_deadline(args, spec, output, scenario_index, overrides):
    """Tune on a fixed paired grid and retain defer only after a 1-SE win."""
    validation_tapes = (
        args.tapes or int(spec.get("validation_tapes", 10))
    )
    thresholds = tuple(float(value) for value in spec.get(
        "deadline_thresholds", (-1.0e30, 0.2, 0.4, 0.6, 0.8)
    ))
    if not any(value < -1.0e20 for value in thresholds):
        raise ValueError("deadline_thresholds must include the never-defer baseline")
    rows = []
    costs_by_threshold = {}
    for threshold in thresholds:
        local = dict(overrides)
        local["deadlineGreedyThreshold"] = threshold
        costs = []
        for tape in range(validation_tapes):
            result = _run_episode(
                replication=tape + 1,
                machine=args.machine,
                phase=str(output / "validation" / f"scenario_{scenario_index}"),
                strategy="deadline_greedy",
                train_mode=False,
                scenario_seed=_seed(scenario_index + 1000, tape),
                policy_seed=53053,
                checkpoint_path=str(output / "unused_checkpoint.pt"),
                diagnostics_path=str(output / "diagnostics.json"),
                overrides=local,
            )
            cost = float(result["objective_normalized_cost"])
            costs.append(cost)
            rows.append({"scenario_index": scenario_index, "threshold": threshold,
                         "tape": tape, "normalized_cost": cost})
            _write_csv(output / "deadline_validation.partial.csv", rows)
        costs_by_threshold[threshold] = np.asarray(costs, dtype=np.float64)
    baseline = next(value for value in thresholds if value < -1.0e20)
    base_costs = costs_by_threshold[baseline]
    qualified = [(float(base_costs.mean()), baseline)]
    for threshold in thresholds:
        if threshold == baseline:
            continue
        improvement = base_costs - costs_by_threshold[threshold]
        if float(improvement.mean()) >= _paired_se(improvement):
            qualified.append((float(costs_by_threshold[threshold].mean()), threshold))
    return float(min(qualified)[1])


def _scenario_entries(spec, base):
    """Plain regime scenarios plus expanded, pre-registered stress doses."""
    entries = []
    for scenario_index, (scenario, scenario_overrides) in enumerate(
        dict(spec.get("scenarios", {})).items()
    ):
        entries.append({
            "name": scenario,
            "family": scenario,
            "dose_key": "",
            "dose": math.nan,
            "seed_index": scenario_index,
            "tuning_key": scenario,
            "is_tuning_dose": True,
            "is_endpoint_dose": True,
            "overrides": dict(scenario_overrides),
            "mechanism": "",
        })
    entries.extend(expand_stress_families(spec, base))
    if not entries:
        raise ValueError("The evaluation specification defines no scenarios")
    return entries


def _support_for(spec):
    curriculum = _load(ROOT / spec["base_curriculum"])
    return training_support(curriculum, stage_ids=spec.get("training_stage_ids", ("P",)))


def _run_panel(args, spec, output):
    base = _base_overrides(spec)
    policies = [
        policy for policy in spec["policies"]
        if not policy.startswith("rl") or not args.skip_rl
    ]
    support = _support_for(spec)
    entries = _scenario_entries(spec, base)
    rows = []
    tuned_thresholds = []
    selected_by_key = {}
    # Tune each tuning group once, at its pre-registered dose, before any
    # held-out tape of that group is run.
    ordered = sorted(entries, key=lambda entry: (not entry["is_tuning_dose"],))
    for entry in ordered:
        if args.skip_tuning or not entry["is_tuning_dose"]:
            continue
        overrides = dict(base)
        overrides.update(entry["overrides"])
        selected_by_key[entry["tuning_key"]] = _tune_deadline(
            args, spec, output, entry["seed_index"], overrides
        )
        tuned_thresholds.append({
            "tuning_key": entry["tuning_key"],
            "tuned_on": entry["name"],
            "selected_threshold": selected_by_key[entry["tuning_key"]],
        })
        _write_csv(output / "deadline_selection.csv", tuned_thresholds)
    for entry in entries:
        scenario = entry["name"]
        overrides = dict(base)
        overrides.update(entry["overrides"])
        if entry["tuning_key"] in selected_by_key:
            overrides["deadlineGreedyThreshold"] = selected_by_key[entry["tuning_key"]]
        outside = out_of_support_keys(entry["overrides"], support)
        tape_count = args.tapes or int(spec["held_out_tapes"])
        for tape in range(tape_count):
            scenario_seed = _seed(entry["seed_index"], tape)
            for policy in policies:
                if (
                    policy == "hindsight_rollout"
                    and tape >= int(spec.get("hindsight_tapes", tape_count))
                ):
                    continue
                if (
                    policy in set(spec.get("endpoint_only_policies", ()))
                    and not entry["is_endpoint_dose"]
                ):
                    # Planner references that are too expensive for every
                    # dose are run at the lowest and highest dose only.
                    continue
                if policy.startswith("rl"):
                    checkpoints = args.rl_runs
                    if not checkpoints:
                        raise ValueError("RL evaluation requires --rl-runs")
                    expected = int(spec.get("policy_seeds", 5))
                    if len(checkpoints) != expected:
                        raise ValueError(
                            f"RL evaluation requires exactly {expected} checkpoints"
                        )
                else:
                    checkpoints = (str(output / "unused_checkpoint.pt"),)
                for policy_index, checkpoint in enumerate(checkpoints):
                    internal_policy = (
                        "hindsight_rollout"
                        if policy == "hindsight_rollout"
                        else policy
                    )
                    phase = output / scenario / f"{internal_policy}_seed_{policy_index}"
                    result = _run_episode(
                        replication=tape + 1,
                        machine=args.machine,
                        phase=str(phase),
                        strategy=internal_policy,
                        train_mode=False,
                        scenario_seed=scenario_seed,
                        policy_seed=530000 + policy_index,
                        checkpoint_path=str(checkpoint),
                        diagnostics_path=str(output / "diagnostics.json"),
                        overrides=overrides,
                    )
                    rows.append({
                        "scenario": scenario,
                        "family": entry["family"],
                        "dose_key": entry["dose_key"],
                        "dose": entry["dose"],
                        "in_training_support": not outside,
                        "out_of_support_keys": json.dumps(outside),
                        "tape": tape,
                        "policy_seed": policy_index,
                        "policy": (
                            "hindsight_ceiling"
                            if policy == "hindsight_rollout"
                            else policy
                        ),
                        "normalized_cost": float(
                            result["objective_normalized_cost"]
                        ),
                        "initial_population": int(result["initial_population"]),
                        "first_regime_event_time": min(
                            (
                                int(event["time"])
                                for event in result.get("hazard_regime_events", ())
                                if event.get("kind") != "false_alarm"
                            ),
                            default=-1,
                        ),
                        "deployment_times": json.dumps(
                            result.get("deployment_times", ())
                        ),
                        "progress_path": str(
                            phase
                            / internal_policy
                            / f"rep_{tape + 1:03d}_{args.machine}"
                            / "progress.csv"
                        ),
                    })
                    _write_csv(output / "episodes.partial.csv", rows)
    return rows


def _write_event_studies(output, rows):
    """Align physical outcomes and deployment timing on the first regime event."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observations = []
    for row in rows:
        event_time = int(row.get("first_regime_event_time", -1))
        progress_path = Path(row.get("progress_path", ""))
        if event_time < 0 or not progress_path.exists():
            continue
        with open(progress_path, "r", newline="", encoding="utf-8") as handle:
            for progress in csv.DictReader(handle):
                relative = int(progress["timestep"]) - event_time
                if -30 <= relative <= 60:
                    observations.append({
                        "scenario": row["scenario"],
                        "policy": row["policy"],
                        "relative_minute": relative,
                        "active_share": float(progress["active_remaining"])
                        / max(1, int(row["initial_population"])),
                        "decision_rate": float(progress["decision_made"]),
                    })
    _write_csv(output / "event_study.csv", observations)
    if not observations:
        return
    plot_dir = output / "event_studies"
    plot_dir.mkdir(parents=True, exist_ok=True)
    policies_to_plot = {
        "objective_greedy", "deadline_greedy", "belief_rollout", "rl"
    }
    for scenario in sorted({row["scenario"] for row in observations}):
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        for policy in sorted(policies_to_plot):
            subset = [
                row for row in observations
                if row["scenario"] == scenario and row["policy"] == policy
            ]
            if not subset:
                continue
            times = sorted({int(row["relative_minute"]) for row in subset})
            means = [
                float(np.mean([
                    row["active_share"] for row in subset
                    if int(row["relative_minute"]) == minute
                ]))
                for minute in times
            ]
            axis.plot(times, means, label=policy)
        axis.axvline(0, color="black", linestyle="--", linewidth=1.0)
        axis.set(
            xlabel="Minutes relative to first regime event",
            ylabel="Mean active population share",
            title=f"{scenario}: event-aligned evacuation",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(plot_dir / f"{scenario}.png", dpi=180)
        plt.close(figure)


def _summarize(rows):
    summaries = []
    scenarios = sorted({row["scenario"] for row in rows})
    for scenario in scenarios:
        subset = [row for row in rows if row["scenario"] == scenario]
        by_policy = {}
        for row in subset:
            by_policy.setdefault(row["policy"], []).append(row)
        means = {
            policy: float(np.mean([row["normalized_cost"] for row in values]))
            for policy, values in by_policy.items()
        }
        deployable_rules = [
            key for key in ("objective_greedy", "forecast_greedy", "deadline_greedy")
            if key in means
        ]
        best_rule = min(deployable_rules, key=means.get)
        tape_cost = {
            policy: {
                (row["tape"], row["policy_seed"]): row["normalized_cost"]
                for row in values
            }
            for policy, values in by_policy.items()
        }

        def contrast(left, right):
            common = sorted(set(tape_cost[left]) & set(tape_cost[right]))
            values = [tape_cost[left][key] - tape_cost[right][key] for key in common]
            return float(np.mean(values)), _paired_se(values)

        def optional_contrast(left, right):
            if left not in tape_cost or right not in tape_cost:
                return math.nan, math.nan
            return contrast(left, right)

        adaptation, adaptation_se = optional_contrast(
            "objective_greedy", "regime_oracle_greedy"
        )
        foresight, foresight_se = optional_contrast(
            "objective_greedy", "belief_rollout"
        )
        ceiling, ceiling_se = optional_contrast(
            "objective_greedy", "hindsight_ceiling"
        )
        rl_share = None
        rl_improvement = math.nan
        rl_ci_low = math.nan
        rl_ci_high = math.nan
        rl_p_value = math.nan
        if "rl" in means:
            reference = (
                "hindsight_ceiling" if "hindsight_ceiling" in means
                else "belief_rollout" if "belief_rollout" in means
                else None
            )
            denominator = (
                means[best_rule] - means[reference]
                if reference is not None else math.nan
            )
            rl_share = (
                (means[best_rule] - means["rl"]) / denominator
                if np.isfinite(denominator) and denominator > 0.0 else None
            )
            best_by_tape = {
                int(row["tape"]): float(row["normalized_cost"])
                for row in by_policy[best_rule]
            }
            seed_effects = []
            for policy_seed in sorted({
                int(row["policy_seed"]) for row in by_policy["rl"]
            }):
                rl_by_tape = {
                    int(row["tape"]): float(row["normalized_cost"])
                    for row in by_policy["rl"]
                    if int(row["policy_seed"]) == policy_seed
                }
                common = sorted(set(best_by_tape) & set(rl_by_tape))
                if common:
                    seed_effects.append(float(np.mean([
                        best_by_tape[tape] - rl_by_tape[tape]
                        for tape in common
                    ])))
            if seed_effects:
                rl_improvement = float(np.mean(seed_effects))
                rl_ci_low, rl_ci_high = _bootstrap_seed_ci(
                    seed_effects, seed=0x53_000 + len(summaries)
                )
                rl_p_value = _sign_flip_pvalue(
                    seed_effects, seed=0x53_100 + len(summaries)
                )
        def by_tape(policy):
            return {
                int(row["tape"]): float(row["normalized_cost"])
                for row in by_policy.get(policy, ())
            }

        headroom_by_tape = []
        objective_by_tape = by_tape("objective_greedy")
        oracle_by_tape = by_tape("regime_oracle_greedy")
        belief_by_tape = by_tape("belief_rollout")
        for tape in sorted(set(objective_by_tape) & set(belief_by_tape)):
            adaptation_term = (
                objective_by_tape[tape] - oracle_by_tape[tape]
                if tape in oracle_by_tape else 0.0
            )
            headroom_by_tape.append(
                adaptation_term + (objective_by_tape[tape] - belief_by_tape[tape])
            )
        headroom = float(np.mean(headroom_by_tape)) if headroom_by_tape else math.nan
        headroom_se = _paired_se(headroom_by_tape)
        first = subset[0]
        summaries.append({
            "scenario": scenario,
            "family": first.get("family", scenario),
            "dose_key": first.get("dose_key", ""),
            "dose": first.get("dose", math.nan),
            "in_training_support": bool(first.get("in_training_support", True)),
            "out_of_support_keys": first.get("out_of_support_keys", "[]"),
            "best_tuned_rule": best_rule,
            "adaptation_value": adaptation,
            "adaptation_paired_se": adaptation_se,
            "foresight_value": foresight,
            "foresight_paired_se": foresight_se,
            "ceiling_value": ceiling,
            "ceiling_paired_se": ceiling_se,
            "adaptation_plus_foresight": headroom,
            "adaptation_plus_foresight_se": headroom_se,
            "gate_1_pass": bool(
                np.isfinite(headroom) and headroom >= 3.0 * headroom_se
            ),
            "rl_vs_best_improvement": rl_improvement,
            "rl_vs_best_seed_bootstrap_95_low": rl_ci_low,
            "rl_vs_best_seed_bootstrap_95_high": rl_ci_high,
            "rl_vs_best_p_value": rl_p_value,
            "rl_share": rl_share,
            **{f"cost_{key}": value for key, value in means.items()},
        })
    _holm_adjust(summaries)
    return summaries


def _dose_response(rows, summaries):
    """Per stress family: paired slope of (best tuned rule - policy) on dose.

    Doses within a family share tapes, so each tape contributes one slope and
    scenario difficulty cancels. Positive slope: the policy's advantage over
    the best deployable rule grows as the mechanism is turned up.
    """
    best_rule = {row["scenario"]: row["best_tuned_rule"] for row in summaries}
    results = []
    families = sorted({
        row["family"] for row in rows if row.get("dose_key")
    })
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        cost = {}
        for row in family_rows:
            key = (row["policy"], int(row["tape"]), float(row["dose"]))
            cost.setdefault(key, []).append(float(row["normalized_cost"]))
        cost = {key: float(np.mean(values)) for key, values in cost.items()}
        doses = sorted({float(row["dose"]) for row in family_rows})
        scenario_for_dose = {
            float(row["dose"]): row["scenario"] for row in family_rows
        }
        for policy in sorted({row["policy"] for row in family_rows}):
            if policy in {"objective_greedy", "forecast_greedy", "deadline_greedy"}:
                continue
            gaps = {}
            for (name, tape, dose), value in cost.items():
                if name != policy:
                    continue
                rule = best_rule[scenario_for_dose[dose]]
                reference = cost.get((rule, tape, dose))
                if reference is not None:
                    gaps.setdefault(tape, {})[dose] = reference - value
            slope, slope_se, tapes = paired_dose_slope(gaps)
            per_dose = {
                dose: [by_dose[dose] for by_dose in gaps.values() if dose in by_dose]
                for dose in doses
            }
            results.append({
                "family": family,
                "policy": policy,
                "dose_key": family_rows[0]["dose_key"],
                "slope_per_dose_unit": slope,
                "slope_paired_se": slope_se,
                "tapes": tapes,
                "monotone_trend_2se": bool(
                    np.isfinite(slope) and np.isfinite(slope_se)
                    and slope > 2.0 * slope_se
                ),
                "gap_by_dose": json.dumps({
                    str(dose): (
                        float(np.mean(values)) if values else None
                    )
                    for dose, values in per_dose.items()
                }),
            })
    return results


def _write_markdown(path, summaries, dose_response=()):
    policies = sorted({
        key.removeprefix("cost_") for row in summaries
        for key in row if key.startswith("cost_")
    })
    columns = [
        "scenario", "training support", *policies, "adaptation", "foresight", "ceiling",
        "Gate 1", "RL vs rule (seed bootstrap 95% CI)", "Holm p", "RL share",
    ]
    lines = ["# v53r value decomposition", "", "| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in summaries:
        values = [row["scenario"]]
        values.append(
            "in" if row.get("in_training_support", True)
            else "OUT " + ",".join(json.loads(row.get("out_of_support_keys", "[]")))
        )
        values.extend(f"{row.get('cost_' + policy, math.nan):.5f}" for policy in policies)
        values.extend((
            f"{row['adaptation_value']:.5f} ± {row['adaptation_paired_se']:.5f}",
            f"{row['foresight_value']:.5f} ± {row['foresight_paired_se']:.5f}",
            f"{row['ceiling_value']:.5f} ± {row['ceiling_paired_se']:.5f}",
            (
                "pass" if row["gate_1_pass"] else "fail"
            ) + (
                f" ({row['adaptation_plus_foresight']:.5f} ± "
                f"{row['adaptation_plus_foresight_se']:.5f})"
            ),
            (
                "—"
                if not np.isfinite(row["rl_vs_best_improvement"])
                else (
                    f"{row['rl_vs_best_improvement']:.5f} "
                    f"[{row['rl_vs_best_seed_bootstrap_95_low']:.5f}, "
                    f"{row['rl_vs_best_seed_bootstrap_95_high']:.5f}]"
                )
            ),
            (
                "—" if not np.isfinite(row["rl_vs_best_holm_p"])
                else f"{row['rl_vs_best_holm_p']:.5g}"
            ),
            "—" if row["rl_share"] is None else f"{row['rl_share']:.3f}",
        ))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend((
        "",
        "Lower cost is better. `hindsight_ceiling` is a clairvoyant bound, not a deployable policy.",
        "Training support `OUT` marks scenarios whose listed drivers never occur in the "
        "policy-training curriculum; results there measure extrapolation, not learning.",
    ))
    if dose_response:
        lines.extend((
            "",
            "## Stress-family dose response",
            "",
            "Gap = best deployable tuned rule minus policy (positive favours the policy).",
            "Slope is the paired per-tape OLS slope on dose; tapes are common across doses.",
            "",
            "| family | dose key | policy | gap by dose | slope ± paired SE | trend > 2 SE |",
            "|---|---|---|---|---|---|",
        ))
        for row in dose_response:
            lines.append(
                f"| {row['family']} | {row['dose_key']} | {row['policy']} | "
                f"{row['gap_by_dose']} | {row['slope_per_dose_unit']:.6f} ± "
                f"{row['slope_paired_se']:.6f} | "
                f"{'yes' if row['monotone_trend_2se'] else 'no'} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/regime_value_decomposition.json")
    parser.add_argument("--machine", default="local")
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--rl-runs", nargs="*", default=[])
    parser.add_argument("--tapes", type=int, default=None)
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    spec = _load(ROOT / args.config)
    output = Path(args.output) if args.output else (
        ROOT / "runs" / f"value_decomposition_{date.today().isoformat()}"
    )
    output.mkdir(parents=True, exist_ok=True)
    source_files = [
        ROOT / "DecisionInterface.py", ROOT / "RLBridge.py",
        ROOT / "CounterfactualBranch.py", ROOT / "RolloutPolicyIteration.py",
        ROOT / "Core.py", ROOT / "HazardDatabase.py", ROOT / "StressScenarios.py",
        ROOT / args.config,
    ]
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ("git", "status", "--porcelain"), cwd=ROOT, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    manifest = {
        "schema_version": 1,
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "command": sys.argv,
        "information_contract": "shared_public_regional_observation",
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "spec": spec,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = _run_panel(args, spec, output)
    summaries = _summarize(rows)
    dose_response = _dose_response(rows, summaries)
    _write_csv(output / "episodes.csv", rows)
    _write_csv(output / "summary.csv", summaries)
    _write_csv(output / "dose_response.csv", dose_response)
    _write_markdown(output / "table.md", summaries, dose_response)
    _write_event_studies(output, rows)
    print(output / "table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
