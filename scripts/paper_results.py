#!/usr/bin/env python3
"""Every results table and figure of the v65 paper from sealed panel outputs.

Inputs are the collected panels of one campaign
(``<campaign>/panels/<panel_id>/evaluation_episode_summary.csv``), the
training shards, and the optional city build profile.  Each exhibit is an
independent function; a missing input or column marks that exhibit
``missing-input`` and the others still run, so the same command is used on
the micro-smoke (``--check``) and on the final campaign.

    python scripts/paper_results.py --campaign runs/paper_v65 \
        --training runs/paper_v65_seed1 runs/paper_v65_seed2 runs/paper_v65_seed3 \
        --out runs/paper_v65/paper_results

``--check`` exits non-zero if any exhibit fails for a reason other than a
panel that was deliberately not run (listed with ``--expect-missing``).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")

# Color follows the entity in every figure (validated categorical slots 1-6;
# contrast relief is provided by legends, markers and the CSV tables).
COLORS = {
    "rl": "#2a78d6",
    "objective_greedy": "#eb6834",
    "belief_rollout": "#1baf7a",
    "rl_early_search": "#eda100",
    "static_greedy": "#e87ba4",
    "heuristic": "#008300",
}
MUTED = "#8a8985"
MARKERS = {"rl": "o", "objective_greedy": "s", "belief_rollout": "^", "rl_early_search": "D",
           "static_greedy": "v", "heuristic": "P"}
LABELS = {
    "rl": "RL (residual NMCC)",
    "rl_early_search": "RL + early search",
    "rl_forecast_ablated": "RL, forecast ablated",
    "rl_precommit": "RL, precommitted (bound)",
    "objective_greedy": "Front-aware greedy",
    "objective_greedy_legacy": "Overrun-blind greedy",
    "deadline_greedy_tuned": "Deadline greedy (tuned)",
    "heuristic": "Practice heuristic",
    "random": "Random",
    "initial_only": "Initial-only",
    "static_greedy": "Static greedy",
    "belief_rollout": "Belief-rollout planner",
    "hindsight_rollout": "Hindsight ceiling",
    "forecast_greedy": "Forecast greedy",
    "regime_oracle_greedy": "Regime-oracle greedy",
    "hazard_weighted": "Hazard-weighted demand",
    "accessibility_deficit": "Accessibility deficit",
    "mip_rolling_horizon": "Rolling-horizon MIP",
}
COST = "objective_normalized_cost"
RL_ARMS = ("rl", "rl_forecast_ablated", "rl_precommit", "rl_early_search")
CITY_LABEL = {"state_college_pa": "State College", "spokane_wa": "Spokane",
              "seattle_wa": "Seattle", "chicago_il": "Chicago", "malibu_ca": "Malibu"}

# Columns each panel must carry for the exhibits that read it.
BASE_COLUMNS = ["arm", "city_id", "city_scenario_replication", "policy_replication", COST,
                "episode_artifacts_dir", "deployment_strategy", "episode_wall_time_s"]
REQUIRED_COLUMNS = {
    "main": BASE_COLUMNS + [
        "unfinished", "active_panicked_at_horizon", "expected_casualties",
        "panic_eligible_first_exposures", "clearance_t90_minutes", "clearance_time_minutes",
        "overrun_shelters", "forced_reevacuated", "deferred_decisions",
        "mean_pedestrian_distance_m", "congested_person_minute_share",
        "unassigned_rational_after_final_installation", "mean_end_to_end_deployment_latency_ms",
        "captured_planner_matrices", "initial_population",
    ],
    "gate1_calm": BASE_COLUMNS,
    "gate1_compound": BASE_COLUMNS,
    "capacity_sweep": BASE_COLUMNS + ["label_interval_min", "label_base_arm"],
    "kappa_sweep": BASE_COLUMNS + ["label_kappa", "label_base_arm"],
    "zero_shot": BASE_COLUMNS,
    "learning_curve": BASE_COLUMNS + ["label_iteration"],
    "sim_checks": BASE_COLUMNS + ["captured_link_density"],
    "plugin_check": BASE_COLUMNS,
}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
class MissingInput(Exception):
    """An exhibit's input panel/file does not exist (not an error in --check)."""


def number(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result


class Inputs:
    def __init__(self, campaign: Path, training: list[Path], out: Path, city_profile: Path | None):
        self.campaign = campaign
        self.training = training
        self.out = out
        self.city_profile = city_profile
        self._panels: dict[str, list[dict]] = {}

    def panel_path(self, panel_id: str) -> Path:
        return self.campaign / "panels" / panel_id / "evaluation_episode_summary.csv"

    def has(self, panel_id: str) -> bool:
        return self.panel_path(panel_id).is_file()

    def panel(self, panel_id: str) -> list[dict]:
        if panel_id not in self._panels:
            path = self.panel_path(panel_id)
            if not path.is_file():
                raise MissingInput(f"panel {panel_id!r} not collected ({path})")
            with path.open(newline="", encoding="utf-8") as handle:
                self._panels[panel_id] = list(csv.DictReader(handle))
        return self._panels[panel_id]

    def panel_manifest(self, panel_id: str) -> dict:
        path = self.campaign / "panels" / panel_id / "panel_manifest.json"
        if not path.is_file():
            raise MissingInput(str(path))
        return json.loads(path.read_text())


def audit_columns(inputs: Inputs) -> list[dict]:
    report = []
    for panel_id, columns in REQUIRED_COLUMNS.items():
        if not inputs.has(panel_id):
            report.append({"panel": panel_id, "status": "not-collected", "missing": []})
            continue
        rows = inputs.panel(panel_id)
        present = set(rows[0]) if rows else set()
        missing = [c for c in columns if c not in present]
        empty = [c for c in columns if c in present and all(r.get(c, "") == "" for r in rows)
                 and not c.startswith("captured_")]
        report.append({"panel": panel_id, "status": "ok" if not missing and not empty else "FAIL",
                       "rows": len(rows), "missing": missing, "empty": empty})
    return report


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def tape_values(rows: list[dict], arm: str, metric: str = COST, *, policies=None) -> dict:
    """{(city, rep): {policy: value}} for one arm."""
    table: dict[tuple, dict] = {}
    for row in rows:
        if row["arm"] != arm:
            continue
        policy = int(float(row.get("policy_replication") or 0))
        if policies is not None and policy not in policies:
            continue
        key = (row["city_id"], int(float(row["city_scenario_replication"])))
        table.setdefault(key, {})[policy] = number(row.get(metric))
    return table


def tape_mean(rows, arm, metric=COST) -> dict:
    return {k: float(np.mean(list(v.values()))) for k, v in tape_values(rows, arm, metric).items()}


def paired(rows, left: str, right: str, metric: str = COST) -> dict:
    """Per-city arrays of left - right on shared tapes (seed-averaged)."""
    a, b = tape_mean(rows, left, metric), tape_mean(rows, right, metric)
    by_city: dict[str, list] = {}
    for key in sorted(set(a) & set(b)):
        by_city.setdefault(key[0], []).append(a[key] - b[key])
    return {city: np.asarray(values, dtype=float) for city, values in by_city.items()}


def se(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else math.nan


def macro(diffs: dict) -> tuple[float, float]:
    if not diffs:
        return math.nan, math.nan
    means = [float(np.mean(v)) for v in diffs.values()]
    ses = [se(v) for v in diffs.values()]
    return float(np.mean(means)), float(math.sqrt(sum(s * s for s in ses)) / len(ses))


def seed_tape_bootstrap(rows, comparator: str, learned: str, metric=COST, draws=2000, seed=6500):
    """CI of the equal-city macro (comparator - learned) resampling tapes and seeds."""
    rng = np.random.default_rng(seed)
    comp = tape_mean(rows, comparator, metric)
    learn = tape_values(rows, learned, metric)
    cities = sorted({k[0] for k in comp} & {k[0] for k in learn})
    blocks = {}
    for city in cities:
        keys = sorted(k for k in comp if k[0] == city and k in learn)
        policies = sorted({p for k in keys for p in learn[k]})
        matrix = np.array([[comp[k] - learn[k].get(p, np.nan) for k in keys] for p in policies])
        blocks[city] = matrix
    if not blocks:
        return math.nan, math.nan
    estimates = np.empty(draws)
    for draw in range(draws):
        city_means = []
        for matrix in blocks.values():
            s = rng.integers(0, matrix.shape[0], matrix.shape[0])
            t = rng.integers(0, matrix.shape[1], matrix.shape[1])
            city_means.append(np.nanmean(matrix[np.ix_(s, t)]))
        estimates[draw] = np.mean(city_means)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def per_seed_effects(rows, comparator: str, learned: str, metric=COST) -> dict:
    comp = tape_mean(rows, comparator, metric)
    learn = tape_values(rows, learned, metric)
    effects: dict[int, list] = {}
    cities = sorted({k[0] for k in learn})
    for policy in sorted({p for v in learn.values() for p in v}):
        city_means = []
        for city in cities:
            diffs = [comp[k] - learn[k][policy] for k in learn
                     if k[0] == city and k in comp and policy in learn[k]]
            if diffs:
                city_means.append(float(np.mean(diffs)))
        effects[policy] = float(np.mean(city_means)) if city_means else math.nan
    return effects


def holm(pvalues: dict) -> dict:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted, running, m = {}, 0.0, len(ordered)
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    for value in np.unique(values):  # average ranks over ties
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x, y) -> float:
    """Spearman rank correlation (average ranks for ties), numpy only."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3:
        return math.nan
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def sign_flip_p(values: np.ndarray, draws=100_000, seed=0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    observed = abs(values.mean())
    if values.size <= 16:
        signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * values.size)).reshape(values.size, -1).T
    else:
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(draws, values.size))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null >= observed - 1e-15) + 1) / (len(null) + 1))


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict], columns: list[str], floatfmt="{:.4f}") -> None:
    def fmt(value):
        if isinstance(value, float):
            return "—" if not math.isfinite(value) else floatfmt.format(value)
        return str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(row.get(c, "")) for c in columns) + " |" for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def figure(width=7.0, height=4.2, ncols=1, nrows=1, **kw):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
        "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
        "lines.linewidth": 1.5, "lines.markersize": 5, "axes.axisbelow": True,
    })
    kw.setdefault("layout", "constrained")
    return plt.subplots(nrows, ncols, figsize=(width, height), **kw)


def save(fig, out: Path, name: str) -> list[str]:
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "pdf"):
        path = out / f"{name}.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def color(arm: str) -> str:
    return COLORS.get(arm, MUTED)


def progress_series(row: dict, column: str) -> np.ndarray | None:
    pattern = os.path.join(row.get("episode_artifacts_dir", ""), "*", "rep_*", "progress.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    with open(matches[0], newline="", encoding="utf-8") as handle:
        values = [(int(float(r["timestep"])), number(r.get(column))) for r in csv.DictReader(handle)]
    if not values:
        return None
    horizon = max(t for t, _ in values)
    series = np.full(horizon + 1, np.nan)
    for t, v in values:
        series[t] = v
    return series


# ---------------------------------------------------------------------------
# exhibits
# ---------------------------------------------------------------------------
def ex_table7_main(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    rows = inputs.panel("main")
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    table = []
    for city in sorted({r["city_id"] for r in rows}):
        city_rows = [r for r in rows if r["city_id"] == city]
        for arm in arms:
            costs = tape_mean(city_rows, arm)
            if not costs:
                continue
            values = np.array(list(costs.values()))
            entry = {"city": CITY_LABEL.get(city, city), "arm": arm, "label": LABELS.get(arm, arm),
                     "tapes": len(values), "cost_mean": float(values.mean()), "cost_se": se(values)}
            if arm != "rl":
                diff = paired(city_rows, arm, "rl").get(city, np.array([]))
                entry.update({"minus_rl_mean": float(diff.mean()) if diff.size else math.nan,
                              "minus_rl_se": se(diff), "rl_wins": int(np.sum(diff > 0)),
                              "pairs": int(diff.size)})
            for metric in ("unfinished", "expected_casualties", "panic_eligible_first_exposures",
                           "clearance_t90_minutes", "clearance_time_minutes", "overrun_shelters"):
                metric_values = np.array(list(tape_mean(city_rows, arm, metric).values()))
                entry[metric] = float(np.nanmean(metric_values)) if metric_values.size else math.nan
            table.append(entry)
    write_csv(out / "table7_main_results.csv", table)
    write_markdown_table(out / "table7_main_results.md", table,
                         ["city", "label", "tapes", "cost_mean", "cost_se", "minus_rl_mean",
                          "minus_rl_se", "rl_wins", "unfinished", "expected_casualties",
                          "panic_eligible_first_exposures", "clearance_t90_minutes"])
    return [str(out / "table7_main_results.csv")]


def ex_hypotheses_main(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    rows = inputs.panel("main")
    arms = set(r["arm"] for r in rows)
    result = {}
    # H1 primary
    diffs = paired(rows, "objective_greedy", "rl")
    mean, macro_se = macro(diffs)
    low, high = seed_tape_bootstrap(rows, "objective_greedy", "rl")
    seeds = per_seed_effects(rows, "objective_greedy", "rl")
    wins = {c: int(np.sum(v > 0)) for c, v in diffs.items()}
    n = {c: int(v.size) for c, v in diffs.items()}
    result["H1"] = {
        "estimand": "equal-city macro mean of (front-aware greedy - RL), seed-averaged",
        "estimate": mean, "se": macro_se, "seed_tape_bootstrap_95": [low, high],
        "per_seed": seeds, "wins": wins, "pairs": n,
        "pass": bool(mean >= 2 * macro_se and all(v >= 0 for v in seeds.values())
                     and all(wins[c] >= 0.6 * n[c] for c in wins)),
        "criterion": ">= 2 SE, >= 0 on every seed, RL wins >= 60% of pairs per city",
    }
    # H2 mechanism: rational unfinished and first exposures by t = 60
    rational = {}
    for arm in ("objective_greedy", "rl"):
        unfinished = tape_mean(rows, arm, "unfinished")
        panicked = tape_mean(rows, arm, "active_panicked_at_horizon")
        rational[arm] = {k: unfinished[k] - panicked.get(k, 0.0) for k in unfinished}
    rational_diffs: dict[str, list] = {}
    for key in sorted(set(rational["objective_greedy"]) & set(rational["rl"])):
        rational_diffs.setdefault(key[0], []).append(rational["objective_greedy"][key] - rational["rl"][key])
    m2, s2 = macro({c: np.array(v) for c, v in rational_diffs.items()})
    exposure_60 = {}
    for arm in ("objective_greedy", "rl"):
        values = {}
        for row in rows:
            if row["arm"] != arm:
                continue
            series = progress_series(row, "panic_eligible_first_exposures")
            if series is None or series.size <= 60:
                continue
            key = (row["city_id"], int(float(row["city_scenario_replication"])))
            values.setdefault(key, []).append(float(series[60]))
        exposure_60[arm] = {k: float(np.mean(v)) for k, v in values.items()}
    exposure_diffs: dict[str, list] = {}
    for key in sorted(set(exposure_60["objective_greedy"]) & set(exposure_60["rl"])):
        exposure_diffs.setdefault(key[0], []).append(
            exposure_60["objective_greedy"][key] - exposure_60["rl"][key])
    m3, s3 = macro({c: np.array(v) for c, v in exposure_diffs.items()})
    result["H2"] = {
        "rational_unfinished_greedy_minus_rl": [m2, s2],
        "first_exposures_t60_greedy_minus_rl": [m3, s3],
        "pass": bool(m2 >= 2 * s2 and m3 >= 2 * s3) if all(map(math.isfinite, (m2, s2, m3, s3))) else None,
        "criterion": "both >= 2 SE in RL's favour",
    }
    # H3 planner share, H4 early search
    if "belief_rollout" in arms:
        greedy_minus_rl = macro(paired(rows, "objective_greedy", "rl"))[0]
        greedy_minus_planner = macro(paired(rows, "objective_greedy", "belief_rollout"))[0]
        share = greedy_minus_rl / greedy_minus_planner if greedy_minus_planner > 0 else math.nan
        result["H3"] = {"planner_share": share,
                        "planner_minus_rl": list(macro(paired(rows, "belief_rollout", "rl"))),
                        "pass": bool(share >= 0.5) if math.isfinite(share) else None,
                        "criterion": "(greedy - RL)/(greedy - planner) >= 0.5"}
        if "rl_early_search" in arms:
            rl_minus_search = macro(paired(rows, "rl", "rl_early_search"))
            search_minus_planner = macro(paired(rows, "rl_early_search", "belief_rollout"))
            rl_minus_planner = macro(paired(rows, "rl", "belief_rollout"))[0]
            wall = {arm: float(np.nanmedian([number(r["episode_wall_time_s"]) for r in rows if r["arm"] == arm]))
                    for arm in ("rl_early_search", "belief_rollout")}
            ratio = wall["rl_early_search"] / wall["belief_rollout"]
            result["H4"] = {
                "rl_minus_search": list(rl_minus_search),
                "search_minus_planner": list(search_minus_planner),
                "episode_wall_ratio_search_to_planner": ratio,
                "pass": bool(rl_minus_search[0] >= -rl_minus_search[1]
                             and search_minus_planner[0] <= 0.5 * rl_minus_planner
                             and ratio <= 0.2),
                "criterion": "RL - search >= -1 SE; search - planner <= half of RL - planner; time ratio <= 0.2",
            }
    if "objective_greedy_legacy" in arms:
        lost = {arm: float(np.nanmean(list(tape_mean(rows, arm, "overrun_shelters").values())))
                for arm in ("objective_greedy_legacy", "objective_greedy", "rl")}
        result["H5"] = {"overrun_shelters_per_episode": lost,
                        "legacy_minus_frontaware": list(macro(paired(rows, "objective_greedy_legacy", "objective_greedy"))),
                        "legacy_minus_rl": list(macro(paired(rows, "objective_greedy_legacy", "rl"))),
                        "pass": None, "criterion": "descriptive"}
    if "rl_forecast_ablated" in arms:
        result["H6"] = {"ablated_minus_rl": list(macro(paired(rows, "rl_forecast_ablated", "rl"))),
                        "pass": None, "criterion": "reported either way"}
    if "random" in arms:
        random_cost = tape_mean(rows, "random")
        gain = {}
        greedy, learned = tape_mean(rows, "objective_greedy"), tape_mean(rows, "rl")
        for key in set(random_cost) & set(greedy) & set(learned):
            gain[key] = greedy[key] - learned[key]
        keys = sorted(gain)
        x = np.array([random_cost[k] for k in keys]); y = np.array([gain[k] for k in keys])
        rho = spearman(x, y)
        rng = np.random.default_rng(8)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(keys), len(keys))
            if len(set(idx)) > 2:
                boots.append(spearman(x[idx], y[idx]))
        result["H8"] = {"spearman_random_cost_vs_gain": rho,
                        "bootstrap_95": [float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))] if boots else None,
                        "pass": None, "criterion": "sign and CI reported"}
        fig, ax = figure(4.6, 3.6)
        for city, marker in zip(sorted({k[0] for k in keys}), ("o", "s", "^")):
            sel = [i for i, k in enumerate(keys) if k[0] == city]
            ax.scatter(x[sel], y[sel], s=28, marker=marker, color=COLORS["rl"], alpha=0.8,
                       edgecolor="white", linewidth=0.8, label=CITY_LABEL.get(city, city))
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set(xlabel="Scenario severity: random-policy cost J", ylabel="Greedy − RL (cost J)",
               title=f"Severity vs gain (Spearman ρ = {rho:.2f})")
        ax.legend()
        save(fig, out, "fig12_severity_vs_gain")
    summary["main_hypotheses"] = result
    (out / "hypotheses_main.json").write_text(json.dumps(result, indent=2, default=float))
    return [str(out / "hypotheses_main.json")]


def ex_fig5_forest(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    rows = inputs.panel("main")
    comparators = [a for a in dict.fromkeys(r["arm"] for r in rows) if a not in ("rl",)]
    cities = sorted({r["city_id"] for r in rows})
    table = []
    for arm in comparators:
        diffs = paired(rows, arm, "rl")
        for city in cities:
            if city in diffs:
                v = diffs[city]
                table.append({"arm": arm, "scope": city, "mean": float(v.mean()), "se": se(v),
                              "low": float(v.mean() - 1.96 * se(v)), "high": float(v.mean() + 1.96 * se(v)),
                              "rl_wins": int(np.sum(v > 0)), "pairs": int(v.size)})
        m, s = macro(diffs)
        low, high = seed_tape_bootstrap(rows, arm, "rl") if arm not in RL_ARMS else (m - 1.96 * s, m + 1.96 * s)
        table.append({"arm": arm, "scope": "macro", "mean": m, "se": s, "low": low, "high": high})
    write_csv(out / "fig5_forest.csv", table)
    fig, ax = figure(6.4, 0.42 * len(comparators) + 1.2)
    scopes = cities + ["macro"]
    offsets = np.linspace(-0.25, 0.25, len(scopes))
    for i, arm in enumerate(comparators):
        for offset, scope in zip(offsets, scopes):
            entry = next((t for t in table if t["arm"] == arm and t["scope"] == scope), None)
            if entry is None or not math.isfinite(entry["mean"]):
                continue
            is_macro = scope == "macro"
            ax.errorbar(entry["mean"], i + offset, xerr=[[entry["mean"] - entry["low"]], [entry["high"] - entry["mean"]]],
                        fmt="D" if is_macro else "o", color="#0b0b0b" if is_macro else color("rl"),
                        markersize=5 if is_macro else 4, elinewidth=1.2, capsize=0)
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.set_yticks(range(len(comparators)), [LABELS.get(a, a) for a in comparators])
    ax.invert_yaxis()
    ax.set(xlabel="Comparator − RL, normalized cost J (positive: RL better)",
           title="Paired differences on held-out CRN pairs (dots: cities; diamond: macro, seed×tape CI)")
    return save(fig, out, "fig5_forest")


def ex_fig6_mechanism(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    rows = inputs.panel("main")
    arms = [a for a in ("objective_greedy", "rl", "rl_early_search", "belief_rollout")
            if any(r["arm"] == a for r in rows)]
    fig, axes = figure(10.0, 3.8, ncols=2, gridspec_kw={"width_ratios": [1.1, 1.0]})
    records = []
    for arm in arms:
        curves = [s[1:] for s in (progress_series(r, "panic_eligible_first_exposures")
                                  for r in rows if r["arm"] == arm) if s is not None]
        if not curves:
            continue
        length = min(len(c) for c in curves)
        population = np.nanmean([number(r.get("initial_population")) for r in rows if r["arm"] == arm])
        mean_curve = np.nanmean([c[:length] for c in curves], axis=0) / max(population, 1.0)
        axes[0].plot(np.arange(1, length + 1), mean_curve, color=color(arm), label=LABELS.get(arm, arm),
                     marker=MARKERS.get(arm), markevery=max(1, length // 8), markersize=4)
        for minute in range(0, length, 5):
            records.append({"arm": arm, "minute": minute + 1, "first_exposure_share": float(mean_curve[minute])})
    axes[0].set(xlabel="Minute", ylabel="Cumulative first exposures / population",
                title="(a) Evacuees first exposed to danger ≥ 3")
    axes[0].legend(loc="upper left", fontsize=7)
    stranded = []
    for arm in [a for a in dict.fromkeys(r["arm"] for r in rows)]:
        unfinished = tape_mean(rows, arm, "unfinished")
        panicked = tape_mean(rows, arm, "active_panicked_at_horizon")
        values = [unfinished[k] - panicked.get(k, 0.0) for k in unfinished]
        stranded.append((arm, float(np.mean(values)), se(np.array(values))))
    stranded.sort(key=lambda item: item[1])
    axes[1].barh(range(len(stranded)), [s[1] for s in stranded], xerr=[s[2] for s in stranded],
                 color=[color(s[0]) for s in stranded], height=0.6, error_kw={"elinewidth": 1})
    axes[1].set_yticks(range(len(stranded)), [LABELS.get(s[0], s[0]) for s in stranded])
    axes[1].set(xlabel="Rational evacuees not safe at the horizon", title="(b) Stranding")
    write_csv(out / "fig6_mechanism_curves.csv", records)
    write_csv(out / "fig6_stranding.csv", [{"arm": a, "mean": m, "se": s} for a, m, s in stranded])
    return save(fig, out, "fig6_mechanism")


def ex_table8_behavior(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    rows = inputs.panel("main")
    metrics = ("overrun_shelters", "forced_reevacuated", "deferred_decisions", "unfinished",
               "active_panicked_at_horizon", "expected_casualties", "mean_pedestrian_distance_m",
               "congested_person_minute_share", "unassigned_rational_after_final_installation")
    table = []
    for arm in dict.fromkeys(r["arm"] for r in rows):
        entry = {"arm": arm, "label": LABELS.get(arm, arm)}
        for metric in metrics:
            values = np.array(list(tape_mean(rows, arm, metric).values()))
            entry[metric] = float(np.nanmean(values)) if values.size else math.nan
        table.append(entry)
    write_csv(out / "table8_behavior.csv", table)
    write_markdown_table(out / "table8_behavior.md", table, ["label", *metrics], "{:.2f}")
    return [str(out / "table8_behavior.csv")]


def ex_fig11_planner_share(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    rows = inputs.panel("main")
    greedy_minus_planner = macro(paired(rows, "objective_greedy", "belief_rollout"))[0]
    points = []
    for arm in ("objective_greedy", "rl", "rl_early_search", "belief_rollout"):
        if not any(r["arm"] == arm for r in rows):
            continue
        gain = 0.0 if arm == "objective_greedy" else macro(paired(rows, "objective_greedy", arm))[0]
        latency = float(np.nanmedian([number(r.get("mean_end_to_end_deployment_latency_ms")) for r in rows if r["arm"] == arm]))
        wall = float(np.nanmedian([number(r["episode_wall_time_s"]) for r in rows if r["arm"] == arm]))
        points.append({"arm": arm, "share_of_planner_gain": gain / greedy_minus_planner if greedy_minus_planner else math.nan,
                       "median_decision_latency_ms": latency, "median_episode_wall_s": wall})
    write_csv(out / "fig11_planner_share.csv", points)
    fig, ax = figure(4.8, 3.6)
    for p in points:
        ax.scatter(p["median_decision_latency_ms"] / 1000.0, p["share_of_planner_gain"], s=48,
                   color=color(p["arm"]), marker=MARKERS.get(p["arm"], "o"), edgecolor="white", zorder=3)
        ax.annotate(LABELS.get(p["arm"], p["arm"]), (p["median_decision_latency_ms"] / 1000.0, p["share_of_planner_gain"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color="#52514e")
    ax.set_xscale("log")
    ax.axhline(1.0, color="#52514e", linewidth=0.8, linestyle=":")
    ax.set(xlabel="Median decision latency (s, log scale)", ylabel="Share of planner gain over greedy",
           title="Planner share vs decision latency")
    return save(fig, out, "fig11_planner_share_vs_latency")


def ex_fig7_capacity(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    rows = inputs.panel("capacity_sweep")
    intervals = sorted({int(float(r["label_interval_min"])) for r in rows})
    table = []
    fig, ax = figure(4.8, 3.6)
    for base, style in (("static_greedy", "-"), ("initial_only", "--")):
        means, errors = [], []
        for interval in intervals:
            subset = [r for r in rows if int(float(r["label_interval_min"])) == interval]
            diffs = paired(subset, f"{base}_i{interval:02d}", f"objective_greedy_i{interval:02d}")
            m, s = macro(diffs)
            means.append(m); errors.append(s)
            table.append({"interval_min": interval, "precommitted": base, "value_of_adaptivity": m, "se": s,
                          **{f"city_{c}": float(v.mean()) for c, v in diffs.items()}})
        ax.errorbar(intervals, means, yerr=errors, color=color(base), marker=MARKERS.get(base, "o"),
                    linestyle=style, capsize=0, label=f"{LABELS.get(base, base)} − sequential greedy")
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set(xlabel="Minutes between capacity tokens", ylabel="Value of adaptivity (cost J)",
           title="Sequential placement pays when capacity arrives over time")
    ax.legend()
    write_csv(out / "fig7_capacity_sweep.csv", table)
    summary["H9"] = {"table": table, "criterion": "value of adaptivity increases with the interval (descriptive)"}
    return save(fig, out, "fig7_capacity_sweep")


def _gate1_family(rows: list[dict]) -> dict:
    arms = set(r["arm"] for r in rows)
    result = {}
    for name, left, right in (("adaptation", "objective_greedy", "regime_oracle_greedy"),
                              ("foresight", "objective_greedy", "belief_rollout"),
                              ("ceiling", "objective_greedy", "hindsight_rollout")):
        if left in arms and right in arms:
            result[name] = list(macro(paired(rows, left, right)))
    greedy, oracle, planner = (tape_mean(rows, a) for a in ("objective_greedy", "regime_oracle_greedy", "belief_rollout"))
    headroom = [(greedy[k] - oracle.get(k, greedy[k])) + (greedy[k] - planner[k]) for k in sorted(set(greedy) & set(planner))]
    headroom = np.array(headroom)
    result["adaptation_plus_foresight"] = [float(headroom.mean()) if headroom.size else math.nan, se(headroom)]
    result["gate_pass"] = bool(headroom.size and headroom.mean() >= 3 * se(headroom))
    deployable = [a for a in ("objective_greedy", "forecast_greedy", "deadline_greedy_tuned") if a in arms]
    means = {a: float(np.mean(list(tape_mean(rows, a).values()))) for a in deployable}
    best = min(means, key=means.get)
    seeds = per_seed_effects(rows, best, "rl") if "rl" in arms else {}
    result["best_deployable_rule"] = best
    result["rl_vs_best_per_seed"] = seeds
    result["rl_vs_best_mean"] = float(np.mean(list(seeds.values()))) if seeds else math.nan
    result["rl_vs_best_p"] = sign_flip_p(np.array(list(seeds.values()))) if seeds else math.nan
    if "hindsight_rollout" in arms or "belief_rollout" in arms:
        reference = "hindsight_rollout" if "hindsight_rollout" in arms else "belief_rollout"
        denominator = macro(paired(rows, best, reference))[0]
        gain = macro(paired(rows, best, "rl"))[0] if "rl" in arms else math.nan
        result["rl_share_of_reference"] = gain / denominator if denominator and denominator > 0 else math.nan
    return result


def ex_fig8_gate1(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    families = {}
    for family in ("calm", "compound"):
        panel_id = f"gate1_{family}"
        if inputs.has(panel_id):
            families[family] = _gate1_family(inputs.panel(panel_id))
    if not families:
        raise MissingInput("no gate1 panel collected")
    adjusted = holm({f: v["rl_vs_best_p"] for f, v in families.items() if math.isfinite(v.get("rl_vs_best_p", math.nan))})
    for family, value in families.items():
        value["rl_vs_best_holm_p"] = adjusted.get(family, math.nan)
    (out / "gate1_summary.json").write_text(json.dumps(families, indent=2, default=float))
    summary["H7"] = families
    fig, ax = figure(5.6, 3.6)
    components = ["adaptation", "foresight", "ceiling"]
    width = 0.8 / len(components)
    shades = {"adaptation": COLORS["objective_greedy"], "foresight": COLORS["belief_rollout"], "ceiling": MUTED}
    for j, component in enumerate(components):
        xs, ys, es = [], [], []
        for i, family in enumerate(families):
            if component in families[family]:
                xs.append(i + (j - 1) * width); ys.append(families[family][component][0]); es.append(families[family][component][1])
        ax.bar(xs, ys, width=width * 0.9, yerr=es, color=shades[component], label=component.capitalize(),
               error_kw={"elinewidth": 1})
    ax.set_xticks(range(len(families)), [f.capitalize() + (" ✓" if families[f]["gate_pass"] else "") for f in families])
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set(ylabel="Greedy − reference (cost J)", title="Gate 1 value decomposition (✓: ≥ 3 SE)")
    ax.legend()
    return save(fig, out, "fig8_gate1")


def ex_fig9_momentum(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    rows = [r for r in inputs.panel("main") if r.get("captured_planner_matrices") and r["arm"] == "belief_rollout"]
    if not rows:
        raise MissingInput("no planner matrices captured in the main panel")
    states = []
    for row in rows:
        path = Path(row["captured_planner_matrices"])
        if not path.is_file():
            continue
        for record in json.loads(path.read_text()):
            if record.get("strategy") != "belief_rollout":
                continue
            matrix = np.asarray(record["matrix"], dtype=float)
            keep = [j for j, cell in enumerate(record["cells"]) if cell is not None]  # drop defer
            matrix = matrix[:, keep]
            if matrix.shape[0] < 2 or matrix.shape[1] < 2 or not np.all(np.isfinite(matrix)):
                continue
            grand = matrix.mean()
            tape = matrix.mean(axis=1, keepdims=True) - grand
            action = matrix.mean(axis=0, keepdims=True) - grand
            residual = matrix - grand - tape - action
            ss = {"tape": float(matrix.shape[1] * np.sum(tape ** 2)),
                  "action": float(matrix.shape[0] * np.sum(action ** 2)),
                  "interaction": float(np.sum(residual ** 2))}
            total = sum(ss.values())
            if total <= 0:
                continue
            states.append({"city_id": row["city_id"], "replication": row["city_scenario_replication"],
                           "decision_index": int(record["decision_index"]),
                           "futures": matrix.shape[0], "candidates": matrix.shape[1],
                           **{f"share_{k}": v / total for k, v in ss.items()}})
    if not states:
        raise MissingInput("planner matrices present but empty")
    write_csv(out / "fig9_momentum_states.csv", states)
    decisions = sorted({s["decision_index"] for s in states})
    summary_rows = []
    for d in decisions:
        sel = [s for s in states if s["decision_index"] == d]
        summary_rows.append({"decision_index": d, "states": len(sel),
                             **{f"median_share_{k}": float(np.median([s[f"share_{k}"] for s in sel]))
                                for k in ("tape", "action", "interaction")}})
    pooled = {k: float(np.median([s[f"share_{k}"] for s in states])) for k in ("tape", "action", "interaction")}
    summary["H10"] = {"pooled_median_shares": pooled, "states": len(states),
                      "pass": bool(pooled["tape"] > pooled["action"]),
                      "criterion": "tape (momentum) share exceeds action share (descriptive)"}
    write_csv(out / "fig9_momentum_by_decision.csv", summary_rows)
    fig, ax = figure(5.6, 3.4)
    bottom = np.zeros(len(decisions))
    for key, shade in (("tape", MUTED), ("action", COLORS["rl"]), ("interaction", COLORS["belief_rollout"])):
        values = np.array([r[f"median_share_{key}"] for r in summary_rows])
        ax.bar(decisions, values, bottom=bottom, color=shade, width=0.7, edgecolor="white", linewidth=1.0,
               label={"tape": "Exogenous future (momentum)", "action": "Placement", "interaction": "Interaction"}[key])
        bottom += values
    ax.set(xlabel="Decision index", ylabel="Median share of graded-cost variance", ylim=(0, 1.0),
           title="System momentum dominates the learning signal")
    ax.legend(loc="lower right", fontsize=7)
    return save(fig, out, "fig9_momentum_variance")


def ex_fig10_learning(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    if not inputs.training:
        raise MissingInput("no training shards given")
    fig, axes = figure(9.0, 3.4, ncols=2)
    records = []
    for index, shard in enumerate(inputs.training):
        for diagnostics in sorted(shard.glob("policies/policy_*/learner_diagnostics.csv")):
            policy = int(diagnostics.parent.name.split("_")[1])
            with diagnostics.open(newline="") as handle:
                updates = [r for r in csv.DictReader(handle) if number(r.get("optimizer_updated")) == 1.0
                           and number(r.get("rollout_pi_actor_enabled")) == 1.0]
            if not updates:
                continue
            iterations = np.arange(1, len(updates) + 1)
            before = np.array([number(r.get("rollout_pi_weighted_pairwise_accuracy_before")) for r in updates])
            axes[0].plot(iterations, before, color=color("rl"), alpha=0.35 + 0.2 * index,
                         marker="o", markersize=3, label=f"Seed {policy}")
            for it, b, r in zip(iterations, before, updates):
                records.append({"policy": policy, "iteration": int(it), "pairwise_accuracy_before": b,
                                "pairwise_accuracy_after": number(r.get("rollout_pi_weighted_pairwise_accuracy_after")),
                                "top1_before": number(r.get("rollout_pi_branched_top1_before")),
                                "target_kl_after": number(r.get("rollout_pi_target_kl_after"))})
    axes[0].set(xlabel="Policy iteration", ylabel="Out-of-sample pairwise accuracy",
                title="(a) Ranking accuracy before each update")
    axes[0].legend()
    if inputs.has("learning_curve"):
        rows = inputs.panel("learning_curve")
        greedy = tape_mean(rows, "objective_greedy")
        iterations = sorted({int(float(r["label_iteration"])) for r in rows if r.get("label_iteration")})
        for policy in sorted({int(float(r["policy_replication"])) for r in rows if r["arm"].startswith("rl_iter")}):
            ys = []
            for it in iterations:
                learned = tape_values(rows, f"rl_iter{it:03d}", policies={policy})
                diffs = [greedy[k] - v[policy] for k, v in learned.items() if k in greedy]
                ys.append(float(np.mean(diffs)) if diffs else math.nan)
            axes[1].plot(iterations, ys, color=color("rl"), marker="o", alpha=0.6, label=f"Seed {policy}")
            for it, y in zip(iterations, ys):
                records.append({"policy": policy, "iteration": it, "fixed_panel_greedy_minus_rl": y})
        axes[1].axhline(0, color="#52514e", linewidth=0.8)
        axes[1].set(xlabel="Policy iteration (checkpoint)", ylabel="Greedy − RL on a fixed panel (J)",
                    title="(b) Fixed-panel improvement")
        axes[1].legend()
    write_csv(out / "fig10_learning.csv", records)
    return save(fig, out, "fig10_learning_curves")


def ex_fig13_zero_shot(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    rows = []
    for panel_id in ("zero_shot", "zero_shot_malibu"):
        if inputs.has(panel_id):
            rows += inputs.panel(panel_id)
    if not rows:
        raise MissingInput("no zero-shot panel collected")
    table = []
    cities = sorted({r["city_id"] for r in rows})
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    for city in cities:
        subset = [r for r in rows if r["city_id"] == city]
        for arm in arms:
            values = np.array(list(tape_mean(subset, arm).values()))
            if values.size:
                diff = paired(subset, "objective_greedy", arm).get(city, np.array([])) if arm != "objective_greedy" else np.array([])
                table.append({"city": city, "arm": arm, "cost_mean": float(values.mean()), "cost_se": se(values),
                              "greedy_minus_arm": float(diff.mean()) if diff.size else math.nan,
                              "greedy_minus_arm_se": se(diff) if diff.size else math.nan})
    write_csv(out / "fig13_zero_shot.csv", table)
    fig, ax = figure(6.0, 3.4)
    width = 0.8 / max(1, len(arms))
    for j, arm in enumerate(arms):
        xs = [i + (j - (len(arms) - 1) / 2) * width for i, c in enumerate(cities)]
        ys = [next((t["cost_mean"] for t in table if t["city"] == c and t["arm"] == arm), math.nan) for c in cities]
        es = [next((t["cost_se"] for t in table if t["city"] == c and t["arm"] == arm), math.nan) for c in cities]
        ax.bar(xs, ys, width=width * 0.9, yerr=es, color=color(arm), label=LABELS.get(arm, arm),
               error_kw={"elinewidth": 1})
    ax.set_xticks(range(len(cities)), [CITY_LABEL.get(c, c) for c in cities])
    ax.set(ylabel="Normalized cost J", title="Zero-shot transfer (no retraining; descriptive)")
    ax.legend(fontsize=7, ncols=2)
    return save(fig, out, "fig13_zero_shot")


def ex_kappa(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    rows = inputs.panel("kappa_sweep")
    table = []
    for kappa in sorted({float(r["label_kappa"]) for r in rows}):
        subset = [r for r in rows if float(r["label_kappa"]) == kappa]
        k = int(kappa)
        for base in dict.fromkeys(r["label_base_arm"] for r in subset):
            arm = f"{base}_k{k}"
            values = np.array(list(tape_mean(subset, arm).values()))
            entry = {"kappa": kappa, "arm": base, "cost_mean": float(values.mean()) if values.size else math.nan}
            if base != "rl" and any(r["arm"] == f"rl_k{k}" for r in subset):
                entry["minus_rl"], entry["minus_rl_se"] = macro(paired(subset, arm, f"rl_k{k}"))
            table.append(entry)
    write_csv(out / "kappa_sweep.csv", table)
    write_markdown_table(out / "kappa_sweep.md", table, ["kappa", "arm", "cost_mean", "minus_rl", "minus_rl_se"])
    return [str(out / "kappa_sweep.csv")]


def ex_table9_compute(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    table = []
    for panel_id in ("main", "zero_shot"):
        if not inputs.has(panel_id):
            continue
        rows = inputs.panel(panel_id)
        for city in sorted({r["city_id"] for r in rows}):
            for arm in dict.fromkeys(r["arm"] for r in rows):
                sel = [r for r in rows if r["city_id"] == city and r["arm"] == arm]
                if not sel:
                    continue
                latency = [number(r.get("mean_end_to_end_deployment_latency_ms")) for r in sel]
                table.append({"panel": panel_id, "city": city, "arm": arm, "episodes": len(sel),
                              "median_episode_wall_s": float(np.nanmedian([number(r["episode_wall_time_s"]) for r in sel])),
                              "median_decision_latency_ms": float(np.nanmedian(latency)),
                              "p95_decision_latency_ms": float(np.nanquantile(latency, 0.95))})
    for shard in inputs.training:
        path = shard / "training_episode_summary.csv"
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for stage in sorted({r.get("training_stage_id", "") for r in rows}):
            sel = [number(r["episode_wall_time_s"]) for r in rows if r.get("training_stage_id", "") == stage]
            table.append({"panel": f"training:{shard.name}", "city": "both", "arm": f"stage {stage}",
                          "episodes": len(sel), "median_episode_wall_s": float(np.nanmedian(sel)),
                          "total_hours": float(np.nansum(sel) / 3600.0)})
    if not table:
        raise MissingInput("no panel or training rows")
    write_csv(out / "table9_compute.csv", table)
    write_markdown_table(out / "table9_compute.md", table,
                         ["panel", "city", "arm", "episodes", "median_episode_wall_s", "median_decision_latency_ms",
                          "p95_decision_latency_ms", "total_hours"], "{:.1f}")
    return [str(out / "table9_compute.csv")]


def ex_table1_cities(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    if inputs.city_profile is None or not inputs.city_profile.is_file():
        raise MissingInput("city build profile not produced (scripts/city_build_profile.py)")
    payload = json.loads(inputs.city_profile.read_text())
    table = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in payload["cities"]]
    write_csv(out / "table1_cities.csv", table)
    write_markdown_table(out / "table1_cities.md", table, list(table[0]), "{:.1f}")
    return [str(out / "table1_cities.csv")]


def ex_fig3_fundamental_diagram(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    rows = [r for r in inputs.panel("sim_checks") if r.get("captured_link_density")]
    if not rows:
        raise MissingInput("no link-density captures")
    totals, record = None, None
    for row in rows:
        path = Path(row["captured_link_density"])
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        counts = np.asarray(record["person_substeps"], dtype=float)
        totals = counts if totals is None else totals + counts
    if totals is None:
        raise MissingInput("link-density files missing")
    edges = np.asarray(record["density_edges"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    jam, shape, floor = record["jam_density"], record["shape"], record["minimum_speed_ratio"]
    free_speed = float(np.nanmedian([number(r.get("free_flow_speed_m_per_second")) for r in rows]))
    if not math.isfinite(free_speed):
        free_speed = 1.34  # Weidmann (1993) free-flow speed, m/s
    rho = np.linspace(0.01, jam, 400)
    ratio = np.clip(1.0 - np.exp(-shape * (1.0 / rho - 1.0 / jam)), floor, 1.0)
    share = totals / totals.sum() if totals.sum() > 0 else totals
    fig, axes = figure(10.5, 3.2, ncols=3)
    axes[0].plot(rho, ratio, color="#0b0b0b")
    axes[0].set(xlabel="Link density (ped/m²)", ylabel="Speed / free-flow speed", title="(a) Weidmann speed–density")
    axes[1].plot(rho, rho * ratio * free_speed, color="#0b0b0b")
    axes[1].set(xlabel="Link density (ped/m²)", ylabel="Specific flow (ped/m/s)", title="(b) Flow–density")
    axes[2].bar(centers, share, width=0.09, color=COLORS["rl"], edgecolor="white", linewidth=0.5)
    axes[2].set(xlabel="Link density (ped/m²)", ylabel="Share of person-substeps",
                title="(c) Densities the simulation visits")
    for axis in axes[:2]:
        axis.axvline(float(edges[np.searchsorted(np.cumsum(share), 0.99)]) if share.sum() else jam,
                     color=MUTED, linestyle=":", linewidth=1)
    write_csv(out / "fig3_link_density_distribution.csv",
              [{"density_low": float(edges[i]), "density_high": float(edges[i + 1]), "share_person_substeps": float(share[i])}
               for i in range(len(share))])
    summary["D1"] = {"p99_density_ped_per_m2": float(edges[np.searchsorted(np.cumsum(share), 0.99)]) if share.sum() else math.nan,
                     "share_above_2_ped_per_m2": float(share[centers > 2.0].sum()), "free_speed_m_per_s": free_speed}
    return save(fig, out, "fig3_fundamental_diagram")


def ex_fig4_cumulative_safe(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    rows = inputs.panel("sim_checks")
    fig, ax = figure(4.8, 3.4)
    records = []
    for city, marker in zip(sorted({r["city_id"] for r in rows}), ("o", "s", "^")):
        curves = []
        for row in [r for r in rows if r["city_id"] == city]:
            evacuated = progress_series(row, "evacuated")
            sheltered = progress_series(row, "current_sheltered_population")
            population = number(row.get("initial_population"))
            if evacuated is None or not math.isfinite(population) or population <= 0:
                continue
            safe = evacuated if sheltered is None else np.fmax(evacuated, 0) + np.nan_to_num(sheltered)
            curves.append(safe / population)
        if not curves:
            continue
        length = min(len(c) for c in curves)
        stack = np.array([c[:length] for c in curves])
        mean = np.nanmean(stack, axis=0)
        ax.plot(np.arange(length), mean, color=COLORS["rl"] if marker == "o" else COLORS["objective_greedy"],
                marker=marker, markevery=max(1, length // 8), markersize=4, label=CITY_LABEL.get(city, city))
        for minute in range(0, length, 5):
            records.append({"city": city, "minute": minute, "safe_share_mean": float(mean[minute])})
    ax.set(xlabel="Minute since warning", ylabel="Share safe (sheltered or evacuated)",
           title="Cumulative safe share (all evacuees depart at t = 0)")
    ax.legend()
    write_csv(out / "fig4_cumulative_safe.csv", records)
    return save(fig, out, "fig4_cumulative_safe")


def ex_plugin_identity(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    rows = inputs.panel("plugin_check")
    a = tape_mean(rows, "objective_greedy")
    b = tape_mean(rows, "plugin_public_score_greedy")
    same = bool(a) and all(a[k] == b.get(k) for k in a)
    summary["plugin_identity"] = {"pairs": len(a), "bitwise_equal_cost": same}
    if not same:
        raise RuntimeError("plug-in path does not reproduce objective_greedy on real maps")
    return []


def ex_tuning(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    tuned = []
    for panel_id in ("main", "gate1_calm", "gate1_compound"):
        try:
            manifest = inputs.panel_manifest(panel_id)
        except MissingInput:
            continue
        for entry in manifest.get("tuned_values", []):
            tuned.append({"panel": panel_id, "tuning_panel": entry["tuning_panel"],
                          "parameter": entry["parameter"], "selected_value": entry["selected_value"]})
    if not tuned:
        raise MissingInput("no tuned values recorded")
    write_csv(out / "tuned_parameters.csv", tuned)
    return [str(out / "tuned_parameters.csv")]


def ex_g2_seed1(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    """Gate G2: seed 1 against the front-aware greedy rule on the 20 main pairs per city."""
    rows = inputs.panel("g2_seed1")
    diffs = paired(rows, "objective_greedy", "rl")
    per_city = {c: {"greedy_minus_rl": float(v.mean()), "se": se(v), "rl_wins": int(np.sum(v > 0)),
                    "pairs": int(v.size)} for c, v in diffs.items()}
    m, s = macro(diffs)
    verdict = bool(per_city and all(entry["greedy_minus_rl"] > 0 for entry in per_city.values()))
    result = {"per_city": per_city, "macro": m, "macro_se": s, "g2_pass": verdict,
              "macro_at_least_2_se": bool(m >= 2 * s),
              "rule": "G2 (readiness plan): seed 1 beats front-aware greedy in both cities; "
                      "macro >= 2 SE is reported as the strength of the evidence"}
    summary["G2"] = result
    (out / "g2_seed1.json").write_text(json.dumps(result, indent=2, default=float))
    return [str(out / "g2_seed1.json")]


def ex_manifest_tables(inputs: Inputs, out: Path, summary: dict) -> list[str]:
    """Tables 2, 3 and 5 inputs and the provenance digests, from the training manifest."""
    manifests = [shard / "experiment_manifest.json" for shard in inputs.training]
    manifests = [m for m in manifests if m.is_file()]
    if not manifests:
        raise MissingInput("no training manifest")
    manifest = json.loads(manifests[0].read_text())
    features = []
    for layer in ("cell", "candidate", "global", "momentum"):
        for name in manifest.get(f"{layer}_observation_features", []):
            features.append({"layer": layer, "feature": name})
    write_csv(out / "table3_features.csv", features)
    curriculum = manifest.get("training_curriculum") or {}
    stage = (curriculum.get("stages") or [{}])[-1]
    variant = (stage.get("variants") or [{}])[0]
    parameters = [{"parameter": k, "value": json.dumps(v)} for k, v in sorted(variant.get("overrides", {}).items())]
    write_csv(out / "table2_table5_parameters.csv", parameters)
    city_overrides = manifest.get("effective_overrides_by_city", {})
    provenance = {
        "git": manifest.get("git"), "launch_seed": manifest.get("launch_seed"),
        "curriculum_id": curriculum.get("curriculum_id"), "curriculum_sha256": curriculum.get("source_sha256"),
        "city_profile_sha256": manifest.get("city_profile_sha256"),
        "dynamics_fingerprint_sha256": __import__("hashlib").sha256(
            json.dumps(manifest.get("policy_dynamics_source_sha256"), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "feature_counts": {layer: len(manifest.get(f"{layer}_observation_features", []))
                           for layer in ("cell", "candidate", "global", "momentum")},
        "cities": sorted(city_overrides),
        "shards": [str(m.parent) for m in manifests],
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    summary["provenance"] = provenance
    return [str(out / "table3_features.csv"), str(out / "provenance.json")]


EXHIBITS: list[tuple[str, Callable]] = [
    ("g2_seed1_gate", ex_g2_seed1),
    ("manifest_tables_2_3_5", ex_manifest_tables),
    ("table7_main_results", ex_table7_main),
    ("hypotheses_main_H1_H6_H8_fig12", ex_hypotheses_main),
    ("fig5_forest", ex_fig5_forest),
    ("fig6_mechanism", ex_fig6_mechanism),
    ("table8_behavior", ex_table8_behavior),
    ("fig11_planner_share", ex_fig11_planner_share),
    ("fig7_capacity_sweep_H9", ex_fig7_capacity),
    ("fig8_gate1_H7", ex_fig8_gate1),
    ("fig9_momentum_H10", ex_fig9_momentum),
    ("fig10_learning_curves", ex_fig10_learning),
    ("fig13_zero_shot", ex_fig13_zero_shot),
    ("kappa_sweep", ex_kappa),
    ("table9_compute", ex_table9_compute),
    ("table1_cities", ex_table1_cities),
    ("fig3_fundamental_diagram_D1", ex_fig3_fundamental_diagram),
    ("fig4_cumulative_safe_D2", ex_fig4_cumulative_safe),
    ("plugin_identity", ex_plugin_identity),
    ("tuned_parameters", ex_tuning),
]


def write_summary(out: Path, summary: dict, status: list[dict], audit: list[dict]) -> Path:
    lines = ["# v65 results summary", "", "Generated by `scripts/paper_results.py`. Every number below is",
             "recomputed from the collected panels; do not edit by hand.", "", "## Exhibit status", "",
             "| exhibit | status | detail |", "|---|---|---|"]
    lines += [f"| {s['exhibit']} | {s['status']} | {s.get('detail', '')[:140]} |" for s in status]
    lines += ["", "## Column audit", "", "| panel | status | rows | missing | empty |", "|---|---|---|---|---|"]
    lines += [f"| {a['panel']} | {a['status']} | {a.get('rows', '')} | {', '.join(a['missing'])} | {', '.join(a.get('empty', []))} |"
              for a in audit]
    lines += ["", "## Hypotheses", "", "```json", json.dumps(summary, indent=2, default=float)[:20000], "```"]
    path = out / "results_summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--training", nargs="*", default=[])
    parser.add_argument("--out")
    parser.add_argument("--city-profile")
    parser.add_argument("--only", help="comma-separated exhibit names")
    parser.add_argument("--check", action="store_true", help="exit non-zero on any failed exhibit or column audit")
    parser.add_argument("--expect-missing", default="", help="comma-separated exhibits allowed to be missing in --check")
    args = parser.parse_args(list(argv) if argv is not None else None)
    campaign = Path(args.campaign).resolve()
    out = Path(args.out).resolve() if args.out else campaign / "paper_results"
    out.mkdir(parents=True, exist_ok=True)
    city_profile = Path(args.city_profile).resolve() if args.city_profile else campaign / "city_build_profile.json"
    inputs = Inputs(campaign, [Path(p).resolve() for p in args.training], out, city_profile)
    audit = audit_columns(inputs)
    summary: dict = {}
    status = []
    only = set(args.only.split(",")) if args.only else None
    for name, function in EXHIBITS:
        if only and name not in only:
            continue
        try:
            outputs = function(inputs, out, summary)
            status.append({"exhibit": name, "status": "ok", "detail": ", ".join(Path(p).name for p in outputs)})
        except MissingInput as exc:
            status.append({"exhibit": name, "status": "missing-input", "detail": str(exc)})
        except Exception as exc:  # report and continue with the other exhibits
            status.append({"exhibit": name, "status": "FAILED", "detail": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()})
    (out / "exhibit_status.json").write_text(json.dumps({"status": status, "audit": audit}, indent=2))
    path = write_summary(out, summary, status, audit)
    for entry in status:
        print(f"{entry['status']:>14s}  {entry['exhibit']}  {entry.get('detail', '')[:100]}")
    for entry in audit:
        if entry["status"] == "FAIL":
            print(f"{'AUDIT FAIL':>14s}  {entry['panel']}: missing={entry['missing']} empty={entry.get('empty')}")
    print(f"summary: {path}")
    if args.check:
        allowed = set(filter(None, args.expect_missing.split(",")))
        bad = [s for s in status if s["status"] == "FAILED"
               or (s["status"] == "missing-input" and s["exhibit"] not in allowed)]
        bad_audit = [a for a in audit if a["status"] == "FAIL"]
        for entry in bad:
            if entry.get("traceback"):
                print(entry["traceback"])
        return 1 if bad or bad_audit else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
