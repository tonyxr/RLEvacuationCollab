#!/usr/bin/env python3
"""Build a chapter-ready Markdown results report from completed artifacts.

The generator is fail-closed: it will not convert a running launch, incomplete
sample matrix, or failed convergence audit into a scientific result.  It
combines the city-simulation evidence with the already separated controlled
mechanism and computational-scaling tiers, while preserving their scope.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LAUNCH = PROJECT_ROOT / "runs" / "full_e0_e6_confirmatory_staged_v2_20260914"
DEFAULT_CONTROLLED = PROJECT_ROOT / "runs" / "or_journal_full_20260909"
DEFAULT_SENSITIVITY = PROJECT_ROOT / "runs" / "extreme_sensitivity_full_20260909"
LEARNED_STRATEGIES = {"rl", "rl_precommit"}
STRATEGY_LABELS = {
    "rl": "Sequential RL",
    "risk_reduction": "Future risk-time reduction",
    "heuristic": "Active-population heuristic",
    "hazard_weighted": "Hazard-weighted demand",
    "accessibility_deficit": "Accessibility deficit",
    "random": "Random feasible region",
    "static_greedy": "Static demand-greedy",
    "rl_precommit": "RL precommitment",
    "initial_only": "Initial-only/round-robin",
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric result: {value!r}")
    return result


def _fmt(value, digits: int = 4) -> str:
    number = _finite(value)
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    return f"{number:.{digits}f}"


def _pct(value, digits: int = 2) -> str:
    return f"{100.0 * _finite(value):.{digits}f}%"


def _require_fields(rows: Sequence[Mapping], fields: Sequence[str], label: str) -> None:
    if not rows:
        raise RuntimeError(f"{label} is empty")
    missing = sorted(set(fields).difference(rows[0]))
    if missing:
        raise RuntimeError(f"{label} lacks required fields: {missing}")


def _expected_evaluation_rows(manifest: Mapping) -> int:
    city_count = len(manifest["city_ids"])
    scenarios = int(manifest["eval_replications_per_city"])
    policies = int(manifest["policy_replicates"])
    strategies = [str(value) for value in manifest["strategies"]]
    multiplier = sum(policies if strategy in LEARNED_STRATEGIES else 1 for strategy in strategies)
    return city_count * scenarios * multiplier


def audit_core_launch(launch_dir: Path) -> dict:
    manifest_path = launch_dir / "experiment_manifest.json"
    training_path = launch_dir / "training_episode_summary.csv"
    evaluation_path = launch_dir / "evaluation_episode_summary.csv"
    convergence_path = launch_dir / "training_convergence_diagnostics.json"
    paired_path = launch_dir / "paired_comparison_by_city.csv"
    required = (manifest_path, training_path, evaluation_path, convergence_path, paired_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Core launch is missing required artifacts: {missing}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Core launch status is {manifest.get('status')!r}, not 'complete'")
    training = _read_csv(training_path)
    evaluation = _read_csv(evaluation_path)
    paired = _read_csv(paired_path)
    convergence = _read_json(convergence_path)
    expected_training = (
        int(manifest["policy_replicates"])
        * int(manifest["train_episodes_per_city"])
        * len(manifest["city_ids"])
    )
    expected_evaluation = _expected_evaluation_rows(manifest)
    checks = {
        "training_rows": len(training) == expected_training,
        "evaluation_rows": len(evaluation) == expected_evaluation,
        "all_policies_converged": bool(convergence.get("all_policies_converged")),
        "population_accounting": all(
            int(float(row["safe_completed"]))
            + int(float(row["casualty"]))
            + int(float(row["unfinished"]))
            == int(float(row["initial_population"]))
            for row in evaluation
        ),
        "finite_primary_metric": all(
            math.isfinite(float(row["objective_episode_return"])) for row in evaluation
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Core completion audit failed: {checks}")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "training": training,
        "training_path": training_path,
        "evaluation": evaluation,
        "evaluation_path": evaluation_path,
        "paired": paired,
        "paired_path": paired_path,
        "convergence": convergence,
        "convergence_path": convergence_path,
        "expected_training": expected_training,
        "expected_evaluation": expected_evaluation,
        "checks": checks,
    }


def equal_city_strategy_summary(rows: Sequence[Mapping]) -> list[dict]:
    """Return equal-city macro descriptive means for every evaluated policy."""
    cities = sorted({str(row["city_id"]) for row in rows})
    strategies = sorted({str(row["deployment_strategy"]) for row in rows})
    output = []
    for strategy in strategies:
        city_means = []
        for city in cities:
            subset = [
                row
                for row in rows
                if str(row["city_id"]) == city
                and str(row["deployment_strategy"]) == strategy
            ]
            if not subset:
                raise RuntimeError(f"Missing {strategy} evaluation rows for {city}")
            population = np.asarray(
                [_finite(row["initial_population"]) for row in subset], dtype=float
            )
            safe = np.asarray(
                [_finite(row["safe_completed"]) for row in subset], dtype=float
            )
            casualty = np.asarray(
                [_finite(row["casualty"]) for row in subset], dtype=float
            )
            unfinished = np.asarray(
                [_finite(row["unfinished"]) for row in subset], dtype=float
            )
            city_means.append(
                {
                    "objective": float(np.mean([_finite(row["objective_episode_return"]) for row in subset])),
                    "safe": float(np.mean(safe / population)),
                    "casualty": float(np.mean(casualty / population)),
                    "unfinished": float(np.mean(unfinished / population)),
                    "rmts": float(np.mean([_finite(row["restricted_mean_time_to_safety"]) for row in subset])),
                    "risk": float(np.mean([_finite(row["normalized_risk_weighted_person_time"]) for row in subset])),
                }
            )
        output.append(
            {
                "strategy": strategy,
                **{
                    key: float(np.mean([row[key] for row in city_means]))
                    for key in city_means[0]
                },
            }
        )
    return sorted(output, key=lambda row: row["objective"], reverse=True)


def _macro_paired_rows(rows: Sequence[Mapping]) -> list[dict]:
    macro = [row for row in rows if str(row.get("scope")) == "macro_all_cities"]
    if not macro:
        raise RuntimeError("Paired analysis contains no macro_all_cities rows")
    order = {
        name: index
        for index, name in enumerate(
            (
                "episode_return",
                "safe_completed",
                "casualty",
                "unfinished",
                "restricted_mean_time_to_safety",
                "normalized_risk_weighted_person_time",
            )
        )
    }
    return sorted(macro, key=lambda row: order.get(str(row["metric"]), 99))


def _simpson_status(paired_rows: Sequence[Mapping]) -> tuple[str, str]:
    objective = [row for row in paired_rows if str(row["metric"]) == "episode_return"]
    macro = [row for row in objective if str(row["scope"]) == "macro_all_cities"]
    cities = [row for row in objective if str(row["scope"]) == "city"]
    if len(macro) != 1 or not cities:
        return "CAUTION", "City-stratified direction could not be fully checked."
    macro_sign = np.sign(_finite(macro[0]["mean_rl_improvement"]))
    city_signs = {int(np.sign(_finite(row["mean_rl_improvement"]))) for row in cities}
    reversal = macro_sign != 0 and city_signs == {-int(macro_sign)}
    if reversal:
        return "RED_FLAG", "The aggregate objective direction reverses in every city."
    mixed = len(city_signs.difference({0})) > 1
    return (
        "CAUTION" if mixed else "NOTE",
        "City-specific directions are heterogeneous but do not show a complete reversal."
        if mixed
        else "Macro and city-stratified objective directions show no complete reversal.",
    )


def fallacy_scan(paired_rows: Sequence[Mapping]) -> list[tuple[str, str, str]]:
    simpson_severity, simpson_detail = _simpson_status(paired_rows)
    return [
        ("Simpson's paradox", simpson_severity, simpson_detail),
        ("Ecological fallacy", "CAUTION", "The estimand is system/city-level; no individual-level behavioral inference is licensed."),
        ("Berkson's paradox", "NOTE", "Cities are fixed design sites rather than an outcome-selected sample; external generalization remains bounded."),
        ("Collider bias", "NOTE", "The paired policy contrast uses no post-treatment covariate adjustment."),
        ("Base-rate neglect", "NOTE", "No classifier sensitivity/specificity claim is made; casualty and completion base rates are reported directly."),
        ("Regression to the mean", "NOTE", "This is not an extreme-score-selected pre/post design."),
        ("Survivorship bias", "NOTE", "Every completed episode retains casualties and unfinished evacuees; the design-cell completion audit is reported."),
        ("Look-elsewhere effect", "NOTE", "One primary objective contrast is confirmatory; operational decompositions are labeled secondary/descriptive."),
        ("Garden of forking paths", "CAUTION", "The executable design is frozen in-repository but was not independently time-stamped before all development work."),
        ("Correlation is not causation", "NOTE", "Common-random-number simulation contrasts support within-model policy effects, not real-world causal effectiveness."),
        ("Reverse causality", "NOTE", "Policies are assigned before simulated outcomes; real-world feedback claims are outside scope."),
    ]


def _controlled_summary(controlled_dir: Path) -> list[dict]:
    manifest = _read_json(controlled_dir / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("audit", {}).get("status") != "passed":
        raise RuntimeError("Controlled mechanism/scaling experiment is not complete and audited")
    return _read_csv(controlled_dir / "extreme_paired_comparison.csv")


def _latency_summary(controlled_dir: Path) -> dict:
    models = _read_json(controlled_dir / "latency_scaling_models.json")["models"]
    result = {}
    for row in models:
        result[(row["benchmark_family"], row["method"])] = row
    return result


def _sensitivity_audit(sensitivity_dir: Path) -> dict:
    manifest = _read_json(sensitivity_dir / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("audit", {}).get("status") != "passed":
        raise RuntimeError("Extreme-sensitivity experiment is not complete and audited")
    return manifest


def build_chapter(
    launch_dir: Path,
    controlled_dir: Path,
    sensitivity_dir: Path,
) -> tuple[str, dict]:
    core = audit_core_launch(launch_dir)
    manifest = core["manifest"]
    macro = _macro_paired_rows(core["paired"])
    strategies = equal_city_strategy_summary(core["evaluation"])
    controlled = _controlled_summary(controlled_dir)
    latency = _latency_summary(controlled_dir)
    sensitivity = _sensitivity_audit(sensitivity_dir)
    reproduction_path = launch_dir / "reproducibility_sample" / "reproducibility_report.json"
    reproduction = _read_json(reproduction_path) if reproduction_path.is_file() else None
    verification_status = "ANALYZED"
    reproducibility_verdict = (
        reproduction.get("verdict", "CANNOT_VERIFY") if reproduction else "CANNOT_VERIFY"
    )
    fallacies = fallacy_scan(core["paired"])
    confidence = "CAUTION" if any(row[1] in {"CAUTION", "RED_FLAG"} for row in fallacies) else "SOLID"

    objective = next(row for row in macro if row["metric"] == "episode_return")
    objective_supported = (
        str(objective.get("inferentially_eligible", "")).lower() == "true"
        and _finite(objective["bootstrap_95_ci_low"]) > 0.0
    )
    conclusion = (
        "The prespecified primary analysis supports RL superiority over the active-population heuristic within the modeled five-city testbed."
        if objective_supported
        else "The prespecified primary analysis does not establish RL superiority over the active-population heuristic within the modeled five-city testbed."
    )

    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {datetime.now(timezone.utc).isoformat()}",
        f"- Verification Status: {verification_status}",
        "- Version Label: full_results_chapter_v1",
        f"- Core Launch: {launch_dir.resolve()}",
        "",
        "# Experiment Results",
        "",
        "## 1. Experiment completion and analysis population",
        "",
        (
            f"The confirmatory city experiment completed {core['expected_training']:,} staged PPO training episodes "
            f"and {core['expected_evaluation']:,} held-out policy-evaluation episodes across "
            f"{len(manifest['city_ids'])} fixed cities, {manifest['policy_replicates']} independently initialized "
            f"policy seeds, and {manifest['eval_replications_per_city']} held-out scenarios per city. "
            "All evaluation rows satisfied exact population accounting and had finite primary objective values."
        ),
        "",
        "The simulator used weighted cohorts capped at 20 persons per moving agent. Results therefore describe the declared cohort approximation and should not be relabeled as exact individual microsimulation.",
        "",
        "## 2. Training convergence",
        "",
        "| Policy seed | Episodes | Equal-city blocks | Tail trend (SD) | Tail shift (SD) | KL violation rate | Converged |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in core["convergence"]["policies"]:
        lines.append(
            "| {policy_replication} | {episodes} | {equal_city_blocks} | {trend_slope:.4f} | {half_shift:.4f} | {max_tail_approximate_kl:.4f} | {passed} |".format(
                policy_replication=int(row["policy_replication"]),
                episodes=int(row["episodes"]),
                equal_city_blocks=int(row.get("equal_city_blocks", 0)),
                trend_slope=_finite(row.get("tail_trend_span_standard_deviations", 0.0)),
                half_shift=_finite(row.get("adjacent_window_shift_standard_deviations", 0.0)),
                max_tail_approximate_kl=_finite(row.get("tail_kl_violation_rate", 0.0)),
                passed="yes" if row.get("converged") else "no",
            )
        )
    lines.extend(
        [
            "",
            "The convergence test was applied to equal-city block means, and its final window fell entirely inside the stationary nominal stage of the curriculum.",
            "",
            "## 3. Primary RL efficacy",
            "",
            (
                f"{conclusion} The equal-city macro improvement in objective return was "
                f"{_fmt(objective['mean_rl_improvement'])} (hierarchical-bootstrap 95% CI "
                f"[{_fmt(objective['bootstrap_95_ci_low'])}, {_fmt(objective['bootstrap_95_ci_high'])}]; "
                f"two-sided policy-seed randomization p={_fmt(objective['two_sided_randomization_p'])}; "
                f"paired dz={_fmt(objective['paired_effect_size_dz'])})."
            ),
            "",
            "| Outcome (benefit-oriented difference) | RL mean | Heuristic mean | RL improvement | 95% CI | p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    metric_labels = {
        "episode_return": "Objective return",
        "safe_completed": "Safe completions",
        "casualty": "Casualties avoided",
        "unfinished": "Unfinished avoided",
        "restricted_mean_time_to_safety": "RMTS reduction (min)",
        "normalized_risk_weighted_person_time": "Normalized risk-time reduction",
    }
    for row in macro:
        lines.append(
            f"| {metric_labels.get(row['metric'], row['metric'])} | {_fmt(row['rl_mean'])} | "
            f"{_fmt(row['heuristic_mean'])} | {_fmt(row['mean_rl_improvement'])} | "
            f"[{_fmt(row['bootstrap_95_ci_low'])}, {_fmt(row['bootstrap_95_ci_high'])}] | "
            f"{_fmt(row['two_sided_randomization_p'])} |"
        )
    lines.extend(
        [
            "",
            "Positive values in the improvement column always favor RL; lower-is-better outcomes were sign-reversed by the analysis code. Only the objective-return contrast is the single prespecified confirmatory primary test. The remaining rows diagnose the operational tradeoff and are secondary.",
            "",
            "## 4. Absolute benchmark performance",
            "",
            "| Policy | Objective return | Safe fraction | Casualty fraction | Unfinished fraction | RMTS (min) | Normalized risk-time |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in strategies:
        lines.append(
            f"| {STRATEGY_LABELS.get(row['strategy'], row['strategy'])} | {_fmt(row['objective'])} | "
            f"{_pct(row['safe'])} | {_pct(row['casualty'])} | {_pct(row['unfinished'])} | "
            f"{_fmt(row['rmts'])} | {_fmt(row['risk'])} |"
        )
    lines.extend(
        [
            "",
            "Absolute values are equal-city descriptive means. Static and precommitment policies are timing controls; their action timing differs intentionally from online policies, while the reported objective remains action-count invariant.",
            "",
            "## 5. City heterogeneity",
            "",
            "| City | Objective improvement | 95% CI | RL win rate |",
            "|---|---:|---:|---:|",
        ]
    )
    city_objective = [
        row
        for row in core["paired"]
        if row["scope"] == "city" and row["metric"] == "episode_return"
    ]
    for row in city_objective:
        lines.append(
            f"| {row['city_id']} | {_fmt(row['mean_rl_improvement'])} | "
            f"[{_fmt(row['bootstrap_95_ci_low'])}, {_fmt(row['bootstrap_95_ci_high'])}] | "
            f"{_pct(row['rl_win_rate'])} |"
        )
    lines.extend(
        [
            "",
            "Cities are fixed study sites and are equally weighted; the intervals do not treat five cities as a random sample of all cities.",
            "",
            "## 6. Controlled mechanism evidence",
            "",
            "The controlled ring experiment is a contextual decision test, not a city evacuation outcome trial.",
            "",
            "| Variant | RL utility improvement | 95% policy-seed CI | Exact sign p | Win rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in controlled:
        lines.append(
            f"| {str(row['variant']).replace('_', ' ').title()} | {_fmt(row['mean_rl_utility_improvement'], 5)} | "
            f"[{_fmt(row['cluster_normal_95_ci_low'], 5)}, {_fmt(row['cluster_normal_95_ci_high'], 5)}] | "
            f"{_fmt(row['two_sided_exact_sign_randomization_p'], 5)} | {_pct(row['rl_utility_win_rate'])} |"
        )
    hierarchical = latency[("candidate_action_space", "hierarchical_rl")]
    flat = latency[("candidate_action_space", "flat_candidate_actor")]
    lines.extend(
        [
            "",
            "## 7. Computational scalability",
            "",
            (
                "At a fixed 8×8 regional representation, the fitted median-latency exponent with candidate count was "
                f"{_fmt(hierarchical['log_log_slope'], 4)} for the exact-site contextual actor and "
                f"{_fmt(flat['log_log_slope'], 4)} for the context-free candidate MLP. These measurements cover policy scoring and exact feasible-candidate selection only; they are not total simulation runtime."
            ),
            "",
            (
                f"The independent sensitivity tier also passed its registered count/hash audit: "
                f"{sensitivity['audit']['observed_counts']['training_update_rows']:,} training updates and "
                f"{sensitivity['audit']['observed_counts']['evaluation_rows']:,} controlled policy evaluations."
            ),
            "",
            "## 8. Statistical validation and limitations",
            "",
            f"- Overall confidence: {confidence}",
            "- Fallacy-scan coverage: 11/11",
            f"- Reproducibility verdict: {reproducibility_verdict}",
            "- Primary uncertainty unit: independently initialized policy seed, with scenario resampling nested within each fixed city",
            "- External-validity boundary: five named OSM-based simulated cities under one calibrated behavioral model",
            "- Model-validity boundary: weighted cohorts, stylized hazard transitions, and candidate shelters inferred from OSM tags",
            "",
            "| Fallacy | Severity | Audit finding |",
            "|---|---|---|",
        ]
    )
    for name, severity, detail in fallacies:
        lines.append(f"| {name} | {severity} | {detail} |")
    lines.extend(
        [
            "",
            "## 9. Reproducibility and artifact inventory",
            "",
            f"- Core manifest SHA-256: `{_sha256(core['manifest_path'])}`",
            f"- Training table SHA-256: `{_sha256(core['training_path'])}`",
            f"- Evaluation table SHA-256: `{_sha256(core['evaluation_path'])}`",
            f"- Paired-analysis table SHA-256: `{_sha256(core['paired_path'])}`",
            f"- Reproducibility sample: `{reproduction_path}`" if reproduction else "- Reproducibility sample: not available",
            f"- Controlled mechanism/scaling manifest: `{controlled_dir / 'manifest.json'}`",
            f"- Sensitivity manifest: `{sensitivity_dir / 'manifest.json'}`",
            "",
            "A sampled exact re-run cannot substitute for a second full independent training-and-evaluation campaign. Accordingly, the Material Passport remains ANALYZED rather than VERIFIED for the complete chapter.",
            "",
            "## 10. Result statement",
            "",
            conclusion,
            "Interpretation must retain the operational decomposition: an objective advantage is not automatically equivalent to simultaneous improvement in survival, completion, and timeliness. The corresponding secondary outcome table identifies which mechanism drove the aggregate reward difference.",
        ]
    )
    audit = {
        "schema_version": 1,
        "status": "passed",
        "verification_status": verification_status,
        "overall_confidence": confidence,
        "primary_superiority_supported": objective_supported,
        "fallacy_scan_coverage": "11/11",
        "reproducibility_verdict": reproducibility_verdict,
        "core_checks": core["checks"],
        "source_hashes": {
            "manifest": _sha256(core["manifest_path"]),
            "training": _sha256(core["training_path"]),
            "evaluation": _sha256(core["evaluation_path"]),
            "paired": _sha256(core["paired_path"]),
        },
    }
    return "\n".join(lines) + "\n", audit


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-dir", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--controlled-dir", type=Path, default=DEFAULT_CONTROLLED)
    parser.add_argument("--sensitivity-dir", type=Path, default=DEFAULT_SENSITIVITY)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    launch_dir = args.launch_dir.resolve()
    output = args.output or (launch_dir / "EXPERIMENT_RESULTS_CHAPTER.md")
    chapter, audit = build_chapter(
        launch_dir,
        args.controlled_dir.resolve(),
        args.sensitivity_dir.resolve(),
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(chapter, encoding="utf-8")
    os.replace(temporary, output)
    audit_path = output.with_name(output.stem + "_validation.json")
    audit["chapter_path"] = str(output)
    audit["chapter_sha256"] = _sha256(output)
    audit["generated_utc"] = datetime.now(timezone.utc).isoformat()
    temporary_audit = audit_path.with_suffix(audit_path.suffix + ".tmp")
    temporary_audit.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary_audit, audit_path)
    print(output)
    print(audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
