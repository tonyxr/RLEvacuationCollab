#!/usr/bin/env python3
"""Site-loss corner case with forward-looking evacuation orders (panels ``site_loss_recourse``
and, if collected, ``site_loss_recourse_rollout``).

    python scripts/site_loss_recourse_results.py --campaign runs/paper_v65_site_loss \
        --out runs/paper_v65_site_loss/recourse_results [--check]

Every arm carries ``label_recourse``: ``off`` (no orders; the stored site_loss
episodes), ``fL30`` / ``fL15`` (forecast order rule, lead 30 / 15 minutes) and,
in the rollout panel, ``roll`` (simulator-graded stopping rule).

Outputs
-------
* ``target_status.csv``: each targeted shelter of each stressed episode is
  ``ordered`` (closed by an order below the overrun level, no later than the
  designed failure minute), ``delivered`` (overrun exactly at the designed
  minute), ``pre-empted`` (overrun earlier, at the same minute as in the paired
  base episode of the same arm family) or ``missed``.
* ``zero_order_identity.csv``: every recourse episode in which no order was
  executed must equal its ``off`` twin on every compared column (the controller
  is then passive: forecasts and rollouts must leave no trace).
* ``recourse_value.csv``: value of recourse ``V = metric(off) - metric(mode)``,
  paired on tape and seed, for the cost J, its five components, expected
  casualties, unfinished and safe evacuees, and the people forced out of
  failing shelters. Positive V for J means the orders help.
* ``recourse_penalty.csv`` / ``recourse_contrasts.csv``: the site-loss penalty
  ``J(cell) - J(base)`` within each mode, and its policy contrasts (Holm across
  the four cells within each contrast and mode), plus ``contrast_change``:
  the contrast under the mode minus the contrast under ``off``.
* ``recourse_orders.csv``: orders per episode, timing relative to the ignition,
  danger at the order, people moved, and false alarms (ordered regions that
  never reach the overrun level), per mode, cell and policy.
* ``rollout_comparison.csv`` (rollout panel only): roll vs fL30 vs off.
* ``recourse_value.png`` and ``recourse_summary.md``.

``--check`` exits 1 when a target is missed, a zero-order episode differs from
its off twin, or an expected mode is absent.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import identity_check as IC  # noqa: E402

POLICY_ORDER = ("rl", "objective_greedy", "objective_greedy_legacy", "static_greedy", "heuristic")
CELL_ORDER = ("t1_m05", "t1_m35", "t2_m05", "t2_m35")
CELLS = ("base", *CELL_ORDER)
MODES = ("fL30", "fL15")
DESIGNED = {"t1_m05": (5, 34), "t2_m05": (5, 34), "t1_m35": (35, 64), "t2_m35": (35, 64)}
CONTRASTS = (
    ("static_greedy", "objective_greedy", "SL1a adaptivity: static - front-aware greedy"),
    ("static_greedy", "rl", "SL1b adaptivity: static - RL"),
    ("objective_greedy_legacy", "objective_greedy", "SL2 front awareness: overrun-blind - front-aware greedy"),
    ("objective_greedy", "rl", "SL3 learning: front-aware greedy - RL"),
    ("heuristic", "rl", "practice: heuristic - RL"),
)
KEY = ["city_id", "tape", "seed", "label_policy"]
VALUE_METRICS = ("objective_normalized_cost", "c_move", "c_term_plain", "c_exposure", "c_term_exposure",
                 "c_casualty", "expected_casualties", "unfinished", "safe_completed", "forced_reevacuated")
OVERRUN_LEVEL = 4


def _parse(value):
    if isinstance(value, (list, dict)):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == "":
        return None
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return ast.literal_eval(str(value))


def _closures(value) -> dict:
    """shelter id -> (minute, danger) of its closure."""
    out = {}
    for event in _parse(value) or []:
        out[int(event["shelter_id"])] = (int(event["time"]), int(event["danger_level"]))
    return out


def load(campaign: Path, panel: str) -> pd.DataFrame | None:
    path = campaign / "panels" / panel / "evaluation_episode_summary.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path, low_memory=False).copy()
    required = {"label_policy", "label_cell", "label_recourse", "city_id", "city_scenario_replication",
                "policy_replication", "shelter_overrun_events", "objective_normalized_cost"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    nh = frame["initial_population"].astype(float) * 180.0
    extra = pd.DataFrame({
        "tape": frame["city_scenario_replication"].astype(int),
        "seed": frame["policy_replication"].astype(int),
        "c_move": frame["active_person_time_weight"] * frame["objective_active_person_time"] / nh,
        "c_term_plain": frame["active_person_time_weight"] * frame["realized_terminal_plain_time"] / nh,
        "c_exposure": frame["hazard_exposure_time_weight"] * frame["objective_hazard_exposure_person_time"] / nh,
        "c_term_exposure": frame["hazard_exposure_time_weight"] * frame["realized_terminal_hazard_exposure_time"] / nh,
        "c_casualty": frame["objective_expected_casualty_person_time"] / nh,
    }, index=frame.index)
    return pd.concat([frame.drop(columns=[c for c in extra.columns if c in frame.columns]), extra], axis=1)


# ---------------------------------------------------------------------------
def target_status(frame: pd.DataFrame) -> pd.DataFrame:
    base = {(row["label_recourse"], *[row[k] for k in KEY]): _closures(row["shelter_overrun_events"])
            for _, row in frame[frame["label_cell"] == "base"].iterrows()}
    rows = []
    for _, row in frame[frame["label_cell"] != "base"].iterrows():
        designed = DESIGNED[row["label_cell"]][1]
        closures = _closures(row["shelter_overrun_events"])
        paired = base.get((row["label_recourse"], *[row[k] for k in KEY]))
        for target in [int(t) for t in (_parse(row.get("scenario_targeted_shelter_ids")) or [])]:
            minute, danger = closures.get(target, (None, None))
            base_minute = None if paired is None else paired.get(target, (None, None))[0]
            if minute is not None and danger < OVERRUN_LEVEL and minute <= designed:
                status = "ordered"
            elif minute == designed and danger is not None and danger >= OVERRUN_LEVEL:
                status = "delivered"
            elif minute is not None and minute < designed and base_minute == minute:
                status = "pre-empted"
            else:
                status = "missed"
            rows.append({"mode": row["label_recourse"], "cell": row["label_cell"], "policy": row["label_policy"],
                         "city_id": row["city_id"], "tape": int(row["tape"]), "seed": int(row["seed"]),
                         "job_key": row.get("job_key"), "target": target, "closure_minute": minute,
                         "closure_danger": danger, "designed_minute": designed, "status": status})
    return pd.DataFrame(rows)


def zero_order_identity(frame: pd.DataFrame) -> pd.DataFrame:
    off = frame[frame["label_recourse"] == "off"].set_index(["label_cell", *KEY])
    rows = []
    passive = frame[(frame["label_recourse"] != "off")
                    & (pd.to_numeric(frame.get("recourse_closed_shelters"), errors="coerce").fillna(0) == 0)]
    columns = [c for c in frame.columns if not IC.EXCLUDE.search(c)
               and not c.startswith(("recourse_", "scenario_", "c_")) and c not in ("tape", "seed", "city_id", "map_provenance")]
    for _, row in passive.iterrows():
        key = (row["label_cell"], *[row[k] for k in KEY])
        if key not in off.index:
            rows.append({"mode": row["label_recourse"], "cell": row["label_cell"], "policy": row["label_policy"],
                         "city_id": row["city_id"], "tape": int(row["tape"]), "seed": int(row["seed"]),
                         "differing_columns": -1, "first_difference": "no off twin"})
            continue
        twin = off.loc[key]
        if isinstance(twin, pd.DataFrame):
            twin = twin.iloc[0]
        diffs = [c for c in columns if not IC._same("" if pd.isna(row[c]) else row[c], "" if pd.isna(twin[c]) else twin[c])]
        rows.append({"mode": row["label_recourse"], "cell": row["label_cell"], "policy": row["label_policy"],
                     "city_id": row["city_id"], "tape": int(row["tape"]), "seed": int(row["seed"]),
                     "differing_columns": len(diffs), "first_difference": diffs[0] if diffs else ""})
    return pd.DataFrame(rows, columns=["mode", "cell", "policy", "city_id", "tape", "seed", "differing_columns",
                                       "first_difference"])


# ---------------------------------------------------------------------------
def _summarize(values: pd.Series) -> dict:
    out = {}
    for city, group in values.groupby(level="city_id"):
        array = group.to_numpy(dtype=float)
        se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else float("nan")
        out[city] = (float(array.mean()), se, len(array))
    if out:
        means = [v[0] for v in out.values()]
        ses = [v[1] for v in out.values()]
        out["macro"] = (float(np.mean(means)), float(math.sqrt(sum(s * s for s in ses)) / len(ses)),
                        int(sum(v[2] for v in out.values())))
    return out


def _series(frame, mode, cell, policy, metric):
    rows = frame[(frame["label_recourse"] == mode) & (frame["label_cell"] == cell) & (frame["label_policy"] == policy)]
    return rows.set_index(["city_id", "tape", "seed"])[metric]


def _seed_mean(values: pd.Series) -> pd.Series:
    return values.dropna().groupby(level=["city_id", "tape"]).mean()


def value_table(frame: pd.DataFrame, modes) -> pd.DataFrame:
    rows = []
    for mode in modes:
        for metric in VALUE_METRICS:
            if metric not in frame:
                continue
            for cell in CELLS:
                for policy in POLICY_ORDER:
                    off = _series(frame, "off", cell, policy, metric)
                    on = _series(frame, mode, cell, policy, metric)
                    if on.empty or off.empty:
                        continue
                    for scope, (mean, se, n) in _summarize(_seed_mean(off - on)).items():
                        rows.append({"mode": mode, "metric": metric, "cell": cell, "policy": policy, "scope": scope,
                                     "value_mean": mean, "value_se": se, "tapes": n,
                                     "t": mean / se if se and se > 0 else float("nan")})
    return pd.DataFrame(rows)


def _penalty(frame, mode, cell, policy, metric="objective_normalized_cost") -> pd.Series:
    return _seed_mean(_series(frame, mode, cell, policy, metric) - _series(frame, mode, "base", policy, metric))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta function (Numerical Recipes 6.4)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def t_two_sided_p(t_value: float, df: float) -> float:
    """Two-sided p-value of Student's t (no SciPy dependency)."""
    if not math.isfinite(t_value):
        return float("nan")
    if t_value == 0.0:
        return 1.0
    x = df / (df + t_value * t_value)
    a, b = df / 2.0, 0.5
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return min(1.0, front * _betacf(a, b, x) / a)
    return min(1.0, 1.0 - front * _betacf(b, a, 1.0 - x) / b)


def _holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted, running = [float("nan")] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def penalty_and_contrasts(frame: pd.DataFrame, modes) -> tuple[pd.DataFrame, pd.DataFrame]:
    penalties, contrasts = [], []
    for mode in ("off", *modes):
        for cell in CELL_ORDER:
            for policy in POLICY_ORDER:
                values = _penalty(frame, mode, cell, policy)
                for scope, (mean, se, n) in _summarize(values).items():
                    penalties.append({"mode": mode, "cell": cell, "policy": policy, "scope": scope,
                                      "penalty_mean": mean, "penalty_se": se, "tapes": n})
        for left, right, label in CONTRASTS:
            block = []
            for cell in CELL_ORDER:
                diff = (_penalty(frame, mode, cell, left) - _penalty(frame, mode, cell, right)).dropna()
                summary = _summarize(diff)
                if "macro" not in summary:
                    continue
                mean, se, n = summary["macro"]
                change = None
                if mode != "off":
                    off = (_penalty(frame, "off", cell, left) - _penalty(frame, "off", cell, right)).dropna()
                    change = _summarize((diff - off).dropna()).get("macro")
                p = t_two_sided_p(abs(mean / se), max(1, n - 2)) if se and se > 0 else float("nan")
                block.append({"mode": mode, "contrast": label, "left": left, "right": right, "cell": cell,
                              "difference_mean": mean, "difference_se": se, "tapes": n, "p": p,
                              "change_vs_off_mean": None if change is None else change[0],
                              "change_vs_off_se": None if change is None else change[1],
                              **{f"{c}_mean": v[0] for c, v in summary.items() if c != "macro"}})
            for row, holm in zip(block, _holm([r["p"] for r in block])):
                row["p_holm"] = holm
            contrasts.extend(block)
    return pd.DataFrame(penalties), pd.DataFrame(contrasts)


def order_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    on = frame[frame["label_recourse"] != "off"]
    for (mode, cell, policy), group in on.groupby(["label_recourse", "label_cell", "label_policy"]):
        minutes = [m for v in group.get("recourse_order_minutes", []) for m in (_parse(v) or [])]
        dangers = [d for v in group.get("recourse_order_dangers", []) for d in (_parse(v) or [])]
        ignition = DESIGNED.get(cell, (None, None))[0]
        rows.append({
            "mode": mode, "cell": cell, "policy": policy, "episodes": len(group),
            "closed_by_order_per_episode": float(pd.to_numeric(group["recourse_closed_shelters"]).mean()),
            "false_alarm_regions_per_episode": float(pd.to_numeric(group["recourse_false_alarm_regions"]).mean()),
            "order_minute_mean": float(np.mean(minutes)) if minutes else float("nan"),
            "minutes_after_ignition_mean": float(np.mean(minutes) - ignition) if minutes and ignition else float("nan"),
            "danger_at_order_mean": float(np.mean(dangers)) if dangers else float("nan"),
            "occupants_evacuated_mean": float(pd.to_numeric(group["recourse_occupants_evacuated"]).mean()),
            "reservations_released_mean": float(pd.to_numeric(group["recourse_reservations_released"]).mean()),
            "controller_wall_s_mean": float(pd.to_numeric(group["recourse_wall_s"]).mean()),
            "evaluations_per_episode": float(pd.to_numeric(group["recourse_evaluations"]).mean()),
        })
    return pd.DataFrame(rows)


def figure(values: pd.DataFrame, out: Path, modes) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = values[(values["metric"] == "objective_normalized_cost") & (values["scope"] == "macro")]
    fig, axes = plt.subplots(1, len(CELLS), figsize=(15, 3.4), sharey=True)
    colors = {"fL30": "#1f4e79", "fL15": "#7fa7cf", "roll": "#c0504d"}
    y = np.arange(len(POLICY_ORDER))
    for axis, cell in zip(axes, CELLS):
        for k, mode in enumerate(modes):
            rows = data[(data["cell"] == cell) & (data["mode"] == mode)].set_index("policy").reindex(POLICY_ORDER)
            axis.barh(y + (k - (len(modes) - 1) / 2) * 0.35, rows["value_mean"], xerr=1.96 * rows["value_se"],
                      height=0.33, color=colors.get(mode, "#888888"), capsize=2, label=mode)
        axis.axvline(0, color="black", linewidth=0.6)
        axis.set_title(cell, fontsize=9)
        axis.set_yticks(y, [p.replace("_", " ") for p in POLICY_ORDER], fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("value of recourse in J (off - on)", fontsize=8)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out / f"recourse_value.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _macro_lines(table: pd.DataFrame, value: str, se: str, modes) -> list[str]:
    lines = ["| mode | cell | " + " | ".join(POLICY_ORDER) + " |", "|---|---|" + "---|" * len(POLICY_ORDER)]
    for mode in modes:
        for cell in CELLS if value == "value_mean" else CELL_ORDER:
            cells = []
            for policy in POLICY_ORDER:
                row = table[(table["mode"] == mode) & (table["cell"] == cell) & (table["policy"] == policy)
                            & (table["scope"] == "macro")]
                if "metric" in table:
                    row = row[row["metric"] == "objective_normalized_cost"]
                cells.append("" if row.empty else f"{row[value].iloc[0]:+.4f} ({row[se].iloc[0]:.4f})")
            lines.append(f"| {mode} | {cell} | " + " | ".join(cells) + " |")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--panel", default="site_loss_recourse")
    parser.add_argument("--rollout-panel", default="site_loss_recourse_rollout")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    out = args.out or args.campaign / "recourse_results"
    out.mkdir(parents=True, exist_ok=True)
    frame = load(args.campaign, args.panel)
    if frame is None:
        raise FileNotFoundError(f"{args.campaign}/panels/{args.panel}: run and collect the panel first")
    modes = tuple(m for m in MODES if (frame["label_recourse"] == m).any())
    problems = []
    if "off" not in set(frame["label_recourse"]):
        problems.append("mode off is absent")
    for mode in MODES:
        if mode not in modes:
            problems.append(f"mode {mode} is absent")

    status = target_status(frame)
    identity = zero_order_identity(frame)
    values = value_table(frame, modes)
    penalties, contrasts = penalty_and_contrasts(frame, modes)
    orders = order_table(frame)
    status.to_csv(out / "target_status.csv", index=False)
    identity.to_csv(out / "zero_order_identity.csv", index=False)
    values.to_csv(out / "recourse_value.csv", index=False)
    penalties.to_csv(out / "recourse_penalty.csv", index=False)
    contrasts.to_csv(out / "recourse_contrasts.csv", index=False)
    orders.to_csv(out / "recourse_orders.csv", index=False)
    figures = figure(values, out, modes)

    missed = status[status["status"] == "missed"]
    if len(missed):
        problems.append(f"{len(missed)} targeted shelter(s) missed")
    bad_identity = identity[identity["differing_columns"] != 0]
    if len(bad_identity):
        problems.append(f"{len(bad_identity)} zero-order episode(s) differ from their off twin")

    rollout = load(args.campaign, args.rollout_panel)
    if rollout is not None:
        roll_modes = tuple(m for m in ("fL30", "roll") if (rollout["label_recourse"] == m).any())
        roll_values = value_table(rollout, roll_modes)
        roll_values.to_csv(out / "rollout_comparison.csv", index=False)
        roll_identity = zero_order_identity(rollout)
        if len(roll_identity[roll_identity["differing_columns"] != 0]):
            problems.append("zero-order episodes of the rollout panel differ from their off twin")
        roll_status = target_status(rollout)
        if len(roll_status[roll_status["status"] == "missed"]):
            problems.append("targeted shelters missed in the rollout panel")

    counts = status.groupby(["mode", "status"]).size().unstack(fill_value=0)
    lines = ["# Site-loss corner case with forward-looking evacuation orders", "",
             f"Checks: {'PASSED' if not problems else 'FAILED'}", *[f"- {p}" for p in problems], "",
             "Targeted shelters by mode and status:", "", "```", counts.to_string(), "```", "",
             f"Zero-order episodes compared with their off twin: {len(identity)}, differing: {len(bad_identity)}", "",
             "## Value of recourse in J (off - on), macro mean (SE); positive = the orders help", "",
             *_macro_lines(values, "value_mean", "value_se", modes), "",
             "## Site-loss penalty J(cell) - J(base) by mode", "",
             *_macro_lines(penalties, "penalty_mean", "penalty_se", ("off", *modes)), "",
             f"Figures: {', '.join(figures)}"]
    (out / "recourse_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"out": str(out), "passed": not problems, "problems": problems}))
    return 0 if (not problems or not args.check) else 1


if __name__ == "__main__":
    sys.exit(main())
