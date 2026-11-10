#!/usr/bin/env python3
"""Tables and figure for the initial-site-loss corner case (panel ``site_loss``).

    python scripts/site_loss_results.py --campaign runs/paper_v65_site_loss \
        --out runs/paper_v65_site_loss/site_loss_results [--check]

Reads ``<campaign>/panels/site_loss/evaluation_episode_summary.csv`` (written
by ``panel_eval.py collect``).

Treatment delivery (manipulation check SL0)
-------------------------------------------
The treatment of a stressed cell is the *incremental* loss of its targeted
initial shelters relative to the base episode of the same tape, policy and
seed. Each targeted shelter of each stressed episode is classified from the
``shelter_overrun_events`` column of the stressed and of the paired base
episode:

* ``delivered``: the target is overrun exactly at the designed minute
  (``scenario_expected_overrun_minute``);
* ``pre-empted``: the target is overrun before the designed minute, at the same
  minute as in the paired base episode. The fire of the tape already destroys
  that shelter, so the spot fire adds no loss there. Initial shelters and the
  fire are identical under every policy, so pre-emption is a property of the
  tape; the check requires it to be identical across all policies and seeds,
  which keeps every within-cell policy comparison a common-random-number one;
* ``missed``: anything else (never overrun, overrun late, or overrun early
  without the same overrun in the base episode).

The check passes when no target is missed and pre-emption is identical across
policies on every tape. ``--check`` exits 1 otherwise.

Outputs
-------
* ``manipulation_check.csv``: per cell and policy, the shares of targets
  delivered, pre-empted and missed, the delivered failure minutes, the
  incremental sites lost per episode, the people released at the targeted
  shelters (occupants and reservation holders) and the occupants forced to
  re-evacuate at delivered targets; base rows give the initial-shelter
  overruns of the unstressed episodes.
* ``target_delivery.csv``: one row per stressed episode and target.
* ``tape_delivery.csv``: one row per cell, city, tape and target, with the
  status shared by all policies and seeds (or ``inconsistent``).
* ``site_loss_penalty.csv``: the paired site-loss penalty
  ``metric(stressed) - metric(base)`` per policy, cell and city, plus the
  equal-city macro mean, for two samples: ``all_tapes`` (every tape; primary,
  as pre-registered) and ``full_dose`` (only tapes on which every target of
  the cell is delivered; the clean dose for SL4). RL is averaged over its
  seeds on each tape before the standard error is taken across tapes.
* ``penalty_contrasts.csv``: penalty(A) - penalty(B) on the same tape, for both
  samples. A positive value means B absorbs the loss better. The contrasts are:
  - adaptive vs precommitted: static_greedy - objective_greedy and
    static_greedy - rl;
  - front-aware vs overrun-blind: objective_greedy_legacy - objective_greedy;
  - learning: objective_greedy - rl;
  - practice: heuristic - rl.
* ``stressed_levels.csv``: J(A) - J(rl) within each cell.
* ``site_loss_penalty.png`` / ``.pdf`` (all tapes), and ``site_loss_summary.md``.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("objective_normalized_cost", "unfinished", "safe_completed", "expected_casualties")
CONTRASTS = (
    ("static_greedy", "objective_greedy", "adaptivity (precommitted - front-aware greedy)"),
    ("static_greedy", "rl", "adaptivity (precommitted - RL)"),
    ("objective_greedy_legacy", "objective_greedy", "front awareness (overrun-blind - front-aware greedy)"),
    ("objective_greedy", "rl", "learning (front-aware greedy - RL)"),
    ("heuristic", "rl", "practice (heuristic - RL)"),
)
POLICY_ORDER = ("rl", "objective_greedy", "objective_greedy_legacy", "static_greedy", "heuristic")
CELL_ORDER = ("t1_m05", "t1_m35", "t2_m05", "t2_m35")
CELL_LABEL = {
    "t1_m05": "1 site, ignition min 5 (fails min 34)",
    "t1_m35": "1 site, ignition min 35 (fails min 64)",
    "t2_m05": "2 sites, ignition min 5 (fail min 34)",
    "t2_m35": "2 sites, ignition min 35 (fail min 64)",
}
SAMPLES = ("all_tapes", "full_dose")
STATUSES = ("delivered", "pre-empted", "missed")
EPISODE_KEY = ["city_id", "tape", "seed", "label_policy"]


def _parse(value):
    """A JSON or Python-literal cell (lists and dicts are written either way)."""
    if isinstance(value, (list, dict)):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == "":
        return None
    text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def _overruns(value) -> dict[int, tuple[int, int]]:
    """First overrun minute and occupants forced out, per shelter id."""
    out: dict[int, tuple[int, int]] = {}
    for event in _parse(value) or []:
        shelter, time = int(event["shelter_id"]), int(round(float(event["time"])))
        if shelter not in out or time < out[shelter][0]:
            out[shelter] = (time, int(event.get("forced_reevacuated_population") or 0))
    return out


def load(campaign: Path) -> pd.DataFrame:
    path = campaign / "panels" / "site_loss" / "evaluation_episode_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: run and collect the site_loss panel first")
    frame = pd.read_csv(path, low_memory=False).copy()
    required = {"label_policy", "label_cell", "city_id", "city_scenario_replication", "policy_replication",
                "shelter_overrun_events", "scenario_targeted_shelter_ids", "scenario_expected_overrun_minute",
                *METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    keys = pd.DataFrame({"tape": frame["city_scenario_replication"].astype(int),
                         "seed": frame["policy_replication"].astype(int)}, index=frame.index)
    return pd.concat([frame.drop(columns=["tape", "seed"], errors="ignore"), keys], axis=1)


# ---------------------------------------------------------------------------
# treatment delivery
# ---------------------------------------------------------------------------
def target_delivery(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per stressed episode and targeted shelter."""
    base = {tuple(row[k] for k in EPISODE_KEY): _overruns(row["shelter_overrun_events"])
            for _, row in frame[frame["label_cell"] == "base"].iterrows()}
    rows = []
    for _, row in frame[frame["label_cell"] != "base"].iterrows():
        key = tuple(row[k] for k in EPISODE_KEY)
        expected = int(round(float(row["scenario_expected_overrun_minute"])))
        stressed = _overruns(row["shelter_overrun_events"])
        paired = base.get(key)
        for target in [int(t) for t in (_parse(row["scenario_targeted_shelter_ids"]) or [])]:
            failure = stressed.get(target, (None, 0))[0]
            base_failure = None if paired is None else paired.get(target, (None, 0))[0]
            if failure is not None and failure == expected:
                status = "delivered"
            elif (failure is not None and base_failure is not None
                  and base_failure < expected and failure == base_failure):
                status = "pre-empted"
            else:
                status = "missed"
            rows.append({
                "cell": row["label_cell"], "policy": row["label_policy"], "city_id": row["city_id"],
                "tape": int(row["tape"]), "seed": int(row["seed"]), "job_key": row.get("job_key"),
                "target": target, "expected_failure_minute": expected, "failure_minute": failure,
                "base_failure_minute": base_failure, "base_paired": paired is not None, "status": status,
                "occupants_reevacuated": stressed.get(target, (None, 0))[1] if status == "delivered" else 0,
            })
    return pd.DataFrame(rows, columns=["cell", "policy", "city_id", "tape", "seed", "job_key", "target",
                                       "expected_failure_minute", "failure_minute", "base_failure_minute",
                                       "base_paired", "status", "occupants_reevacuated"])


def tape_delivery(targets: pd.DataFrame) -> pd.DataFrame:
    """Per cell, city, tape and target: the status shared by every policy and seed."""
    rows = []
    for (cell, city, tape, target), group in targets.groupby(["cell", "city_id", "tape", "target"]):
        outcomes = {(s, None if pd.isna(m) else int(m)) for s, m in zip(group["status"], group["failure_minute"])}
        status, minute = next(iter(outcomes)) if len(outcomes) == 1 else ("inconsistent", None)
        rows.append({"cell": cell, "city_id": city, "tape": int(tape), "target": int(target),
                     "status": status, "failure_minute": minute, "episodes": len(group),
                     "statuses_seen": "; ".join(f"{s}@{m}" for s, m in sorted(outcomes, key=str))})
    return pd.DataFrame(rows, columns=["cell", "city_id", "tape", "target", "status", "failure_minute",
                                       "episodes", "statuses_seen"])


def annotate(frame: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Add per-episode delivery counts and the full-dose flag to the stressed rows."""
    frame = frame.copy()
    key = ["cell", "city_id", "tape", "seed", "policy"]
    counts = (targets.assign(**{s: targets["status"] == s for s in STATUSES})
              .groupby(key)[list(STATUSES)].sum().reset_index())
    counts = counts.rename(columns={"cell": "label_cell", "policy": "label_policy", "pre-empted": "preempted"})
    frame = frame.merge(counts, on=["label_cell", "city_id", "tape", "seed", "label_policy"], how="left")
    stressed = frame["label_cell"] != "base"
    for column in ("delivered", "preempted", "missed"):
        frame[column] = frame[column].fillna(0).astype(int)
    frame["full_dose"] = np.where(stressed, (frame["preempted"] == 0) & (frame["missed"] == 0), True)
    return frame


def manipulation_check(frame: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stressed = frame[frame["label_cell"] != "base"]
    for (cell, policy), group in stressed.groupby(["label_cell", "label_policy"]):
        mine = targets[(targets["cell"] == cell) & (targets["policy"] == policy)]
        delivered = mine[mine["status"] == "delivered"]
        rows.append({
            "cell": cell, "policy": policy, "episodes": len(group), "targets": len(mine),
            "share_delivered": float((mine["status"] == "delivered").mean()) if len(mine) else float("nan"),
            "share_preempted": float((mine["status"] == "pre-empted").mean()) if len(mine) else float("nan"),
            "share_missed": float((mine["status"] == "missed").mean()) if len(mine) else float("nan"),
            "expected_failure_minute": float(mine["expected_failure_minute"].iloc[0]) if len(mine) else float("nan"),
            "delivered_minute_min": float(delivered["failure_minute"].min()) if len(delivered) else float("nan"),
            "delivered_minute_max": float(delivered["failure_minute"].max()) if len(delivered) else float("nan"),
            "incremental_sites_lost_mean": float(group["delivered"].mean()),
            "full_dose_episodes": int(group["full_dose"].sum()),
            "released_people_mean": float(pd.to_numeric(group.get("scenario_released_population_at_targets"),
                                                        errors="coerce").mean()),
            "occupants_reevacuated_mean": float(mine.groupby(["city_id", "tape", "seed"])["occupants_reevacuated"]
                                                .sum().mean()) if len(mine) else float("nan"),
            "other_overruns_mean": float(pd.to_numeric(group.get("scenario_other_overruns"), errors="coerce").mean()),
        })
    initial = sorted({int(t) for t in targets["target"]}) if len(targets) else []
    for policy, group in frame[frame["label_cell"] == "base"].groupby("label_policy"):
        lost = [sum(1 for s in _overruns(v) if s in initial) for v in group["shelter_overrun_events"]]
        rows.append({"cell": "base", "policy": policy, "episodes": len(group),
                     "initial_shelters_overrun_mean": float(np.mean(lost)) if lost else float("nan"),
                     "other_overruns_mean": float(pd.to_numeric(group.get("overrun_shelters"),
                                                                errors="coerce").mean())})
    return pd.DataFrame(rows)


def check_passes(check: pd.DataFrame, tapes: pd.DataFrame) -> tuple[bool, list[str]]:
    problems = []
    stressed = check[check["cell"] != "base"]
    if stressed.empty:
        problems.append("no stressed episodes")
    missed = stressed[stressed["share_missed"] > 0]
    for _, row in missed.iterrows():
        problems.append(f"{row['cell']} {row['policy']}: {row['share_missed']:.0%} of targets missed")
    inconsistent = tapes[tapes["status"] == "inconsistent"]
    for _, row in inconsistent.iterrows():
        problems.append(f"{row['cell']} {row['city_id']} tape {row['tape']} target {row['target']}: "
                        f"status differs across policies ({row['statuses_seen']})")
    return not problems, problems


# ---------------------------------------------------------------------------
# penalties
# ---------------------------------------------------------------------------
def _per_tape(frame: pd.DataFrame, policy: str, cell: str, metric: str) -> pd.Series:
    """Metric per (city, tape), averaged over seeds (RL has three)."""
    rows = frame[(frame["label_policy"] == policy) & (frame["label_cell"] == cell)]
    return rows.groupby(["city_id", "tape"])[metric].mean()


def _penalty(frame: pd.DataFrame, policy: str, cell: str, metric: str, sample: str = "all_tapes") -> pd.Series:
    """Seed-averaged paired penalty per (city, tape); pairing is per seed."""
    rows = frame[frame["label_policy"] == policy]
    stressed = rows[rows["label_cell"] == cell]
    if sample == "full_dose":
        stressed = stressed[stressed["full_dose"].astype(bool)]
    stressed = stressed.set_index(["city_id", "tape", "seed"])[metric]
    base = rows[rows["label_cell"] == "base"].set_index(["city_id", "tape", "seed"])[metric]
    paired = (stressed - base).dropna()
    return paired.groupby(level=["city_id", "tape"]).mean()


def _summarize(values: pd.Series) -> dict:
    out = {}
    means, ses = [], []
    for city, group in values.groupby(level="city_id"):
        array = group.to_numpy(dtype=float)
        mean = float(array.mean())
        se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else float("nan")
        out[city] = (mean, se, len(array))
        means.append(mean)
        ses.append(se)
    if means:
        out["macro"] = (float(np.mean(means)), float(math.sqrt(sum(s * s for s in ses)) / len(ses)),
                        int(sum(v[2] for v in out.values())))
    return out


def penalty_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample in SAMPLES:
        for metric in METRICS:
            for policy in POLICY_ORDER:
                for cell in CELL_ORDER:
                    values = _penalty(frame, policy, cell, metric, sample)
                    for scope, (mean, se, n) in _summarize(values).items():
                        rows.append({"sample": sample, "metric": metric, "policy": policy, "cell": cell,
                                     "scope": scope, "penalty_mean": mean, "penalty_se": se, "tapes": n,
                                     "t": mean / se if se and se > 0 else float("nan")})
    return pd.DataFrame(rows)


def contrast_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample in SAMPLES:
        for metric in METRICS:
            for left, right, label in CONTRASTS:
                for cell in CELL_ORDER:
                    difference = (_penalty(frame, left, cell, metric, sample)
                                  - _penalty(frame, right, cell, metric, sample)).dropna()
                    for scope, (mean, se, n) in _summarize(difference).items():
                        rows.append({"sample": sample, "metric": metric, "contrast": label, "left": left,
                                     "right": right, "cell": cell, "scope": scope, "difference_mean": mean,
                                     "difference_se": se, "tapes": n,
                                     "t": mean / se if se and se > 0 else float("nan")})
    return pd.DataFrame(rows)


def level_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for cell in ("base", *CELL_ORDER):
            reference = _per_tape(frame, "rl", cell, metric)
            for policy in POLICY_ORDER[1:]:
                difference = (_per_tape(frame, policy, cell, metric) - reference).dropna()
                for scope, (mean, se, n) in _summarize(difference).items():
                    rows.append({"metric": metric, "cell": cell, "policy": policy, "scope": scope,
                                 "minus_rl_mean": mean, "minus_rl_se": se, "tapes": n})
    return pd.DataFrame(rows)


def figure(penalties: pd.DataFrame, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = penalties[(penalties["sample"] == "all_tapes") & (penalties["metric"] == "objective_normalized_cost")
                     & (penalties["scope"] == "macro")]
    fig, axes = plt.subplots(1, len(CELL_ORDER), figsize=(13, 3.4), sharey=True)
    colors = {"rl": "#1f4e79", "objective_greedy": "#5b8db8", "objective_greedy_legacy": "#a6bddb",
              "static_greedy": "#c0504d", "heuristic": "#8c8c8c"}
    for axis, cell in zip(axes, CELL_ORDER):
        rows = data[data["cell"] == cell].set_index("policy").reindex(POLICY_ORDER)
        y = np.arange(len(POLICY_ORDER))
        axis.barh(y, rows["penalty_mean"], xerr=1.96 * rows["penalty_se"],
                  color=[colors[p] for p in POLICY_ORDER], height=0.65, capsize=2)
        axis.set_title(CELL_LABEL[cell], fontsize=9)
        axis.axvline(0, color="black", linewidth=0.6)
        axis.set_yticks(y, [p.replace("_", " ") for p in POLICY_ORDER], fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("site-loss penalty in J (stressed - base)", fontsize=8)
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out / f"site_loss_penalty.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _macro_table(penalties: pd.DataFrame, sample: str) -> list[str]:
    macro = penalties[(penalties["sample"] == sample) & (penalties["metric"] == "objective_normalized_cost")
                      & (penalties["scope"] == "macro")]
    lines = ["| cell | tapes | " + " | ".join(POLICY_ORDER) + " |", "|---|---|" + "---|" * len(POLICY_ORDER)]
    for cell in CELL_ORDER:
        cells, tapes = [], ""
        for policy in POLICY_ORDER:
            row = macro[(macro["cell"] == cell) & (macro["policy"] == policy)]
            if not row.empty:
                tapes = str(int(row["tapes"].iloc[0]))
            cells.append("" if row.empty else f"{row['penalty_mean'].iloc[0]:.4f} ({row['penalty_se'].iloc[0]:.4f})")
        lines.append(f"| {CELL_LABEL[cell]} | {tapes} | " + " | ".join(cells) + " |")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 unless every target is delivered or consistently pre-empted")
    args = parser.parse_args(argv)
    out = args.out or args.campaign / "site_loss_results"
    out.mkdir(parents=True, exist_ok=True)
    frame = load(args.campaign)
    targets = target_delivery(frame)
    tapes = tape_delivery(targets)
    frame = annotate(frame, targets)
    check = manipulation_check(frame, targets)
    passed, problems = check_passes(check, tapes)
    penalties = penalty_table(frame)
    contrasts = contrast_table(frame)
    levels = level_table(frame)
    check.to_csv(out / "manipulation_check.csv", index=False)
    targets.to_csv(out / "target_delivery.csv", index=False)
    tapes.to_csv(out / "tape_delivery.csv", index=False)
    penalties.to_csv(out / "site_loss_penalty.csv", index=False)
    contrasts.to_csv(out / "penalty_contrasts.csv", index=False)
    levels.to_csv(out / "stressed_levels.csv", index=False)
    figures = figure(penalties, out)

    preempted = tapes[tapes["status"] == "pre-empted"]
    shares = targets["status"].value_counts().reindex(STATUSES, fill_value=0)
    lines = ["# Initial-site-loss corner case", "",
             f"Manipulation check: {'PASSED' if passed else 'FAILED'}. Of {len(targets)} targeted shelters "
             f"in stressed episodes, {shares['delivered']} were lost at the designed minute (delivered), "
             f"{shares['pre-empted']} had already been lost at the same minute in the paired base episode "
             f"(pre-empted by the tape's own fire), and {shares['missed']} were missed.", ""]
    lines += [f"- {p}" for p in problems]
    if len(preempted):
        lines += ["", "Pre-empted targets (identical under every policy and seed):", ""]
        for _, row in preempted.iterrows():
            lines.append(f"- {row['cell']}, {row['city_id']} tape {row['tape']}: initial shelter {row['target']} "
                         f"overrun at minute {row['failure_minute']} in base and stressed episodes")
    lines += ["", "## Site-loss penalty in normalized cost J, all tapes (primary)", "",
              *_macro_table(penalties, "all_tapes"),
              "", "## Site-loss penalty on full-dose tapes (every target delivered)", "",
              *_macro_table(penalties, "full_dose"),
              "", "Macro mean (SE) over tapes; RL averaged over seeds on each tape. "
              "Full-dose tapes exclude only tapes on which a target was pre-empted.",
              "", f"Figures: {', '.join(figures)}"]
    (out / "site_loss_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"out": str(out), "manipulation_check_passed": passed,
                      "targets": {k: int(v) for k, v in shares.items()},
                      "preempted_tapes": int(preempted[["cell", "city_id", "tape"]].drop_duplicates().shape[0])}))
    return 0 if (passed or not args.check) else 1


if __name__ == "__main__":
    sys.exit(main())
