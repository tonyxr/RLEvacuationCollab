#!/usr/bin/env python3
"""Crash-safe, parallel, content-addressed evaluation panels (v65.1).

One evaluator for every held-out block of the paper: the main panel, Gate 1
families, sweeps, zero-shot cities and the fixed-panel learning curve.

* **Environment.**  Every job's configuration is rebuilt from the training
  manifest exactly as ``multicity_backtest.py --eval-only`` builds it
  (city profile -> recorded ``--override`` values -> curriculum learner
  overrides -> final-stage variant cycle), then the panel's own overrides and
  the arm's overrides are applied on top.  Scenario and policy seeds use the
  multicity streams (500 + scale rank, 600 + scale rank) unless the spec says
  otherwise.  The city-profile file, curriculum file and dynamics-source
  fingerprint are checked against the manifest; any mismatch stops the run.
* **Store.**  Each finished episode is written atomically to
  ``<campaign>/episode_store/<k[:2]>/<k>.json`` where ``k`` hashes everything
  that determines the episode (city, seeds, strategy, checkpoint bytes, full
  override dict, dynamics fingerprint, plug-in source).  Identical jobs in
  different panels run once.  Re-running a command resumes; nothing is lost
  on a crash except the episodes in flight.
* **Execution.**  Worker processes are started one at a time (macOS libomp
  start-up race), keep the static map cache warm, and are replaced if they
  crash or exceed the timeout.  Plain and heavy (branching) strategies have
  separate worker groups.

Commands::

    python scripts/panel_eval.py plan    --spec config/panels/main.json
    python scripts/panel_eval.py run     --spec config/panels/main.json --plain-workers 6 --heavy-workers 1
    python scripts/panel_eval.py status  --spec config/panels/main.json
    python scripts/panel_eval.py collect --spec config/panels/main.json

Overrides for smoke tests and re-use of a production spec::

    --campaign runs/smoke_campaign --training runs/a runs/b --replications 1:2

* **Exogenous scenario hooks.**  A spec (or one arm) may name
  ``"scenario": {"hook": "scenario_hooks.module:Class", "params": {...}}``.
  The hook edits the exogenous event schedule of each episode after
  initialization and before the first transition, through the simulator's
  own event list; it is applied identically for every strategy on a tape,
  and its parameters and the digest of its package enter the episode key.
  Jobs without a hook keep exactly the keys they had before hooks existed.
  An arm with ``"scenario": null`` is the unstressed (paired base) arm.
* **Recourse controllers.**  A spec (or one arm) may name
  ``"recourse": {"controller": "recourse.module:Class", "params": {...}}``:
  a decision rule that can order the evacuation of shelters about to be
  overrun, used together with the arm's placement policy.  It is wired into
  the live episode by ``recourse.runtime`` (no frozen file is edited); its
  parameters and the digest of the ``recourse`` package enter the key of the
  jobs that use it, and every other job keeps its key.  ``"recourse": null``
  is the arm without recourse.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVALUATOR_VERSION = "panel_eval/v1"
DEFAULT_SCENARIO_STREAM = 500
DEFAULT_POLICY_STREAM = 600
HEAVY_STRATEGIES = frozenset({"belief_rollout", "hindsight_rollout", "rl_early_search"})
LEARNED_STRATEGIES = frozenset({"rl", "rl_forecast_ablated", "rl_early_search", "rl_precommit"})
ACCEPTED_TRAINING_STATUS = frozenset(
    {"training_shard_complete", "complete", "training_not_converged"}
)
# Fields every training source of one panel must share (assembler contract
# plus the two fields the assembler does not check).
SOURCE_CONTRACT_FIELDS = (
    "launch_seed",
    "city_profile_sha256",
    "city_ids",
    "shared_grid",
    "policy_replicates",
    "ppo_rollout_episodes",
    "learning_rate",
    "effective_overrides_by_city",
    "training_curriculum",
    "train_episodes_per_city",
    "total_train_episodes_per_policy",
    "training_city_schedule",
    "common_overrides",
    "policy_dynamics_source_sha256",
)
SPEC_FIELDS = {
    "schema_version",
    "panel_id",
    "description",
    "campaign_dir",
    "training",
    "cities",
    "replications",
    "seed_streams",
    "overrides",
    "arms",
    "capture",
    "tuning",
    "labels",
    "timeouts_s",
    "scenario",
    "recourse",
}
ARM_FIELDS = {
    "arm",
    "labels",
    "strategy",
    "policies",
    "checkpoint",
    "overrides",
    "replications",
    "cities",
    "heavy",
    "optional",
    "note",
    "scenario",
    "recourse",
}


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    """Episode rows may hold numpy scalars/arrays; store them the way
    ``backtest._write_csv`` would print them (str for reduced-precision floats)."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(str(value))
        if isinstance(value, np.ndarray):
            return [_json_default(v) if isinstance(v, np.generic) else v for v in value.tolist()]
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True, default=_json_default)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, allow_nan=True, default=_json_default)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value: str | os.PathLike) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def parse_replications(value: Any) -> tuple[int, ...]:
    """Accept [first, last] (inclusive), "first:last", an int count, or a list."""
    if isinstance(value, int):
        return tuple(range(1, value + 1))
    if isinstance(value, str):
        first, _, last = value.partition(":")
        return tuple(range(int(first), int(last or first) + 1))
    values = [int(item) for item in value]
    if len(values) == 2 and values[0] <= values[1]:
        return tuple(range(values[0], values[1] + 1))
    if len(values) == 1:
        return (values[0],)
    raise ValueError(f"replications must be [first, last], [single], 'a:b' or a count, got {value!r}")


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------
def load_spec(path: Path, cli: argparse.Namespace | None = None) -> dict:
    spec = read_json(path)
    unknown = sorted(set(spec).difference(SPEC_FIELDS))
    if unknown:
        raise ValueError(f"{path}: unknown spec fields {unknown}")
    for required in ("schema_version", "panel_id", "campaign_dir", "training", "cities", "replications", "arms"):
        if required not in spec:
            raise ValueError(f"{path}: missing spec field {required!r}")
    if int(spec["schema_version"]) != 1:
        raise ValueError("Only panel spec schema_version 1 is supported")
    panel_id = str(spec["panel_id"])
    if not panel_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in panel_id):
        raise ValueError("panel_id must use lower-case letters, digits, '_' or '-'")
    names = set()
    for arm in spec["arms"]:
        bad = sorted(set(arm).difference(ARM_FIELDS))
        if bad:
            raise ValueError(f"arm {arm.get('arm')!r}: unknown fields {bad}")
        name = str(arm["arm"])
        if name in names:
            raise ValueError(f"duplicate arm {name!r}")
        names.add(name)
        if arm.get("scenario") is not None:
            scenario_hook(arm["scenario"])  # validates hook name and params
        if arm.get("recourse") is not None:
            recourse_controller(arm["recourse"])  # validates controller name and params
    if spec.get("scenario") is not None:
        scenario_hook(spec["scenario"])
    if spec.get("recourse") is not None:
        recourse_controller(spec["recourse"])
    spec = copy.deepcopy(spec)
    spec["_spec_path"] = str(path)
    spec["_spec_sha256"] = sha256_file(path)
    if cli is not None:
        if getattr(cli, "campaign", None):
            spec["campaign_dir"] = cli.campaign
        if getattr(cli, "training", None):
            spec["training"] = {"sources": list(cli.training)}
        if getattr(cli, "replications", None):
            spec["replications"] = cli.replications
            for arm in spec["arms"]:
                if "replications" in arm:
                    limited = [r for r in parse_replications(arm["replications"])
                               if r in parse_replications(cli.replications)]
                    limited = limited or [parse_replications(cli.replications)[0]]
                    arm["replications"] = f"{limited[0]}:{limited[-1]}"
        if getattr(cli, "cities", None):
            spec["cities"] = [c for c in cli.cities.split(",") if c]
        if getattr(cli, "policies", None):
            available = {int(p) for p in str(cli.policies).split(",") if p}
            for arm in spec["arms"]:
                if "policies" in arm:
                    kept = [p for p in arm["policies"] if int(p) in available]
                    if not kept:
                        raise ValueError(f"arm {arm['arm']!r} has no policy in --policies {sorted(available)}")
                    arm["policies"] = kept
        spec["_cli_overrides"] = {
            key: getattr(cli, key)
            for key in ("campaign", "training", "replications", "cities", "policies")
            if getattr(cli, key, None)
        }
    return spec


# ---------------------------------------------------------------------------
# training source and environment
# ---------------------------------------------------------------------------
@dataclass
class TrainingSource:
    manifest: dict
    source_dirs: list[Path]
    policy_dirs: dict[int, Path]
    statuses: dict[str, str]

    @property
    def fingerprint(self) -> dict:
        return dict(self.manifest["policy_dynamics_source_sha256"])


def load_training_source(paths: Sequence[str], *, allow_running: bool = False) -> TrainingSource:
    if not paths:
        raise ValueError("At least one training source directory is required")
    reference = None
    reference_dir = None
    policy_dirs: dict[int, Path] = {}
    statuses = {}
    resolved_dirs = []
    for raw in paths:
        directory = resolve_path(raw).resolve()
        manifest_path = directory / "experiment_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = read_json(manifest_path)
        status = str(manifest.get("status"))
        statuses[str(directory)] = status
        if status not in ACCEPTED_TRAINING_STATUS and not (allow_running and status == "running"):
            raise ValueError(
                f"Training source {directory} has status {status!r}; a panel with "
                "learned arms needs a finished shard or an assembled launch"
            )
        contract = {name: manifest.get(name) for name in SOURCE_CONTRACT_FIELDS}
        if reference is None:
            reference, reference_dir = manifest, directory
        else:
            reference_contract = {name: reference.get(name) for name in SOURCE_CONTRACT_FIELDS}
            if contract != reference_contract:
                mismatched = sorted(
                    name for name in SOURCE_CONTRACT_FIELDS
                    if contract[name] != reference_contract[name]
                )
                raise ValueError(
                    f"Training sources disagree on {mismatched}: {reference_dir} vs {directory}"
                )
        indices = manifest.get("training_policy_indices") or list(
            range(1, int(manifest.get("policy_replicates", 1)) + 1)
        )
        for index in indices:
            index = int(index)
            if index in policy_dirs:
                raise ValueError(f"Policy {index} appears in more than one training source")
            policy_dirs[index] = directory / "policies" / f"policy_{index:03d}"
        resolved_dirs.append(directory)
    return TrainingSource(reference, resolved_dirs, policy_dirs, statuses)


@dataclass
class EvaluationEnvironment:
    launch_seed: int
    suite: Any
    curriculum: Any
    user_overrides: dict
    hazard_mode: str
    rollout_episodes: int
    learning_rate: float
    fingerprint_sha256: str
    # v66: files whose source differs from the training manifest, recorded
    # only when the panel was run with --allow-dynamics-change.
    dynamics_change: Optional[list] = None

    def city(self, city_id: str):
        return self.suite.select([city_id])[0]

    def base_overrides(self, city) -> dict:
        """Exactly ``evaluation_overrides_by_city`` of multicity_backtest.main."""
        from multicity_backtest import _city_overrides

        overrides = _city_overrides(self.suite, city, self.user_overrides, self.hazard_mode)
        overrides["ppoRolloutEpisodes"] = int(self.rollout_episodes)
        overrides["learningRate"] = float(self.learning_rate)
        variants = tuple(self.curriculum.stages[-1].variants)
        interface = {} if not variants else dict(variants[0].overrides)
        return {**overrides, **self.curriculum.learner_overrides, **interface}

    def variant_for(self, replication: int):
        variants = tuple(self.curriculum.stages[-1].variants)
        cycle = tuple(v for v in variants for _ in range(int(v.weight)))
        return None if not cycle else cycle[(int(replication) - 1) % len(cycle)]


def build_environment(source: TrainingSource, *, check_code: bool = True) -> EvaluationEnvironment:
    from CityProfiles import load_city_suite
    from multicity_backtest import POLICY_DYNAMICS_SOURCES, PROJECT_ROOT, _city_overrides
    from PolicyCache import source_fingerprint
    from TrainingCurriculum import load_training_curriculum

    manifest = source.manifest
    snapshot_path = source.source_dirs[0] / "city_profile_snapshot.json"
    profile_path = (
        read_json(snapshot_path).get("source_path") if snapshot_path.is_file() else None
    )
    suite = load_city_suite(profile_path or str(ROOT / "config" / "city_profiles.json"))
    if suite.source_sha256 != manifest["city_profile_sha256"]:
        raise ValueError(
            "city_profiles.json changed since training "
            f"({suite.source_sha256[:12]} != {manifest['city_profile_sha256'][:12]})"
        )
    recorded = manifest["training_curriculum"]
    if recorded is None:
        raise ValueError("Panels require a curriculum-trained launch")
    curriculum = load_training_curriculum(recorded["source_path"])
    if curriculum.as_dict() != recorded:
        raise ValueError(
            f"Curriculum file {recorded['source_path']} changed since training"
        )
    if check_code:
        current = source_fingerprint(PROJECT_ROOT, POLICY_DYNAMICS_SOURCES)
        recorded_fp = manifest["policy_dynamics_source_sha256"]
        changed = sorted(
            name for name in set(current) | set(recorded_fp)
            if current.get(name) != recorded_fp.get(name)
        )
        if changed:
            raise ValueError(
                "Dynamics sources differ from the training code: "
                f"{changed}. Restore the frozen code (git checkout the training "
                "commit) before evaluating."
            )
    user_overrides = dict(manifest.get("common_overrides") or {})
    effective = manifest["effective_overrides_by_city"]
    first = next(iter(effective.values()))
    hazard_mode = str(first.get("hazardEvolutionMode", "stochastic"))
    env = EvaluationEnvironment(
        launch_seed=int(manifest["launch_seed"]),
        suite=suite,
        curriculum=curriculum,
        user_overrides=user_overrides,
        hazard_mode=hazard_mode,
        rollout_episodes=int(manifest["ppo_rollout_episodes"]),
        learning_rate=float(manifest["learning_rate"]),
        fingerprint_sha256=sha256_text(canonical_json(manifest["policy_dynamics_source_sha256"])),
    )
    # Reproduce the recorded per-city training overrides exactly.
    for city_id, recorded_overrides in effective.items():
        rebuilt = _city_overrides(suite, env.city(city_id), user_overrides, hazard_mode)
        rebuilt["ppoRolloutEpisodes"] = env.rollout_episodes
        rebuilt["learningRate"] = env.learning_rate
        if canonical_json(rebuilt) != canonical_json(recorded_overrides):
            raise ValueError(f"Rebuilt overrides for {city_id} differ from the training manifest")
    return env


def multicity_seed(launch_seed: int, stream: int, index: int) -> int:
    from multicity_backtest import _seed

    return int(_seed(launch_seed, stream, index))


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------
@dataclass
class Job:
    key: str
    panel_id: str
    arm: str
    strategy: str
    policy_replication: int
    city_id: str
    city_scale_rank: int
    city_scenario_replication: int
    scenario_seed: int
    policy_seed: int
    checkpoint_path: str
    checkpoint_sha256: str | None
    overrides: dict
    evaluation_variant_id: str
    heavy: bool
    optional: bool
    timeout_s: float
    external_source_sha256: str | None = None
    labels: dict = field(default_factory=dict)
    capture: list = field(default_factory=list)
    scenario: dict | None = None
    scenario_source_sha256: str | None = None
    recourse: dict | None = None
    recourse_source_sha256: str | None = None

    def store_paths(self, campaign: Path) -> tuple[Path, Path]:
        base = campaign / "episode_store" / self.key[:2]
        return base / f"{self.key}.json", base / self.key


def _checkpoint_for(source: TrainingSource, policy: int, checkpoint: str | None) -> Path:
    if policy not in source.policy_dirs:
        raise ValueError(f"Policy {policy} is not in the training sources")
    directory = source.policy_dirs[policy]
    if not checkpoint or checkpoint == "final":
        return directory / "regional_policy.pt"
    if checkpoint.startswith("iteration:"):
        number = int(checkpoint.split(":", 1)[1])
        return directory / "iteration_checkpoints" / f"regional_policy_iteration_{number:03d}.pt"
    raise ValueError(f"checkpoint must be 'final' or 'iteration:N', got {checkpoint!r}")


def _external_source_sha(overrides: Mapping) -> str | None:
    """Digest of the plug-in's whole top-level package (helpers included)."""
    spec = str(overrides.get("externalPolicy", "") or "")
    if not spec:
        return None
    module_name = spec.partition(":")[0]
    candidate = ROOT / Path(*module_name.split("."))
    if not candidate.with_suffix(".py").is_file() and not (candidate / "__init__.py").is_file():
        raise FileNotFoundError(f"External comparator module not found: {module_name}")
    package = ROOT / module_name.split(".")[0]
    files = sorted(package.rglob("*.py")) if package.is_dir() else [candidate.with_suffix(".py")]
    return sha256_text(canonical_json({str(f.relative_to(ROOT)): sha256_file(f) for f in files}))


def _scenario_block(spec: dict, arm: dict) -> dict | None:
    """The arm's scenario, else the spec's; ``"scenario": null`` in an arm is the base arm."""
    if "scenario" in arm:
        return arm["scenario"]
    return spec.get("scenario")


def scenario_hook(block: Mapping):
    """Instantiate a scenario hook from ``{"hook": "module:Class", "params": {...}}``."""
    import importlib

    if not isinstance(block, Mapping) or set(block) - {"hook", "params"} or "hook" not in block:
        raise ValueError(f"scenario must be {{'hook': 'module:Class', 'params': {{...}}}}, got {block!r}")
    module_name, _, class_name = str(block["hook"]).partition(":")
    if not module_name.startswith("scenario_hooks") or not class_name:
        raise ValueError("scenario hooks live in the scenario_hooks package: 'scenario_hooks.module:Class'")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    hook_class = getattr(importlib.import_module(module_name), class_name)
    return hook_class(**dict(block.get("params") or {}))


def _scenario_source_sha() -> str:
    package = ROOT / "scenario_hooks"
    files = sorted(package.rglob("*.py"))
    return sha256_text(canonical_json({str(f.relative_to(ROOT)): sha256_file(f) for f in files}))


def _recourse_block(spec: dict, arm: dict) -> dict | None:
    """The arm's recourse controller, else the spec's; ``"recourse": null`` in an arm means none."""
    if "recourse" in arm:
        return arm["recourse"]
    return spec.get("recourse")


def recourse_controller(block: Mapping):
    """Instantiate ``{"controller": "recourse.module:Class", "params": {...}}``."""
    import importlib

    if not isinstance(block, Mapping) or set(block) - {"controller", "params"} or "controller" not in block:
        raise ValueError(f"recourse must be {{'controller': 'module:Class', 'params': {{...}}}}, got {block!r}")
    module_name, _, class_name = str(block["controller"]).partition(":")
    if not module_name.startswith("recourse") or not class_name:
        raise ValueError("recourse controllers live in the recourse package: 'recourse.module:Class'")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    controller_class = getattr(importlib.import_module(module_name), class_name)
    return controller_class(**dict(block.get("params") or {}))


def _recourse_source_sha() -> str:
    package = ROOT / "recourse"
    files = sorted(package.rglob("*.py"))
    return sha256_text(canonical_json({str(f.relative_to(ROOT)): sha256_file(f) for f in files}))


_EVALUATION_CODE_SHA: str | None = None
EPISODE_ENTRY_MODULES = ("backtest", "Core", "RLBridge", "CounterfactualBranch", "NMCCBranchWorker")
EPISODE_DATA_FILES = ("RLEvacuationParameter.csv",)


def episode_module_closure() -> list[Path]:
    """Repository modules an episode can execute: the static import closure
    (every ``import``/``from`` anywhere in the file, lazy ones included) of
    the episode entry points, restricted to top-level files of this repo."""
    import ast

    pending = list(EPISODE_ENTRY_MODULES)
    seen: dict[str, Path] = {}
    while pending:
        name = pending.pop()
        path = ROOT / f"{name}.py"
        if name in seen or not path.is_file():
            continue
        seen[name] = path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pending.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                pending.append(node.module.split(".")[0])
    return sorted(seen.values())


def evaluation_code_sha() -> str:
    """Digest of everything an episode executes or reads beyond the trained
    checkpoint and the override dictionary.  It covers modules outside
    POLICY_DYNAMICS_SOURCES (RolloutPolicyIteration.py, NMCCBranchWorker.py,
    backtest.py, ...) but not unrelated analysis scripts, so editing a
    plotting script never invalidates stored episodes."""
    global _EVALUATION_CODE_SHA
    if _EVALUATION_CODE_SHA is None:
        files = episode_module_closure() + [ROOT / name for name in EPISODE_DATA_FILES if (ROOT / name).is_file()]
        _EVALUATION_CODE_SHA = sha256_text(
            canonical_json({f.name: sha256_file(f) for f in files})
        )
    return _EVALUATION_CODE_SHA


def resolve_tuned_values(spec: dict, campaign: Path, cli) -> dict:
    """Resolve {"$tuned": {...}} override values from finished tuning panels."""
    resolved = {}

    def visit(value):
        if isinstance(value, dict) and "$tuned" in value:
            reference = value["$tuned"]
            tune_spec_path = resolve_path(reference["spec"])
            key = canonical_json(reference)
            if key not in resolved:
                tune_spec = load_spec(tune_spec_path, cli)
                resolved[key] = select_tuned_value(tune_spec, cli)
            return resolved[key]["selected_value"]
        return value

    for arm in spec["arms"]:
        overrides = arm.get("overrides", {})
        for name, value in list(overrides.items()):
            overrides[name] = visit(value)
    spec["_tuned"] = list(resolved.values())
    return spec


def select_tuned_value(tune_spec: dict, cli) -> dict:
    """One-SE rule: keep a value only if it beats the baseline by >= 1 paired SE."""
    import numpy as np

    tuning = tune_spec.get("tuning")
    if not tuning:
        raise ValueError(f"{tune_spec['_spec_path']} has no 'tuning' block")
    source = load_training_source(_training_paths(tune_spec), allow_running=True)
    env = build_environment(source)
    jobs = build_jobs(tune_spec, source, env)
    campaign = resolve_path(tune_spec["campaign_dir"])
    costs: dict[str, dict[tuple, float]] = {}
    missing = 0
    for job in jobs:
        record_path, _ = job.store_paths(campaign)
        if not record_path.is_file():
            missing += 1
            continue
        row = read_json(record_path)["row"]
        costs.setdefault(job.arm, {})[(job.city_id, job.city_scenario_replication)] = float(
            row["objective_normalized_cost"]
        )
    if missing:
        raise RuntimeError(
            f"Tuning panel {tune_spec['panel_id']} is incomplete ({missing} episodes missing); "
            "run it before the panel that uses its value"
        )
    values = {str(k): v for k, v in tuning["arm_values"].items()}
    baseline = str(tuning["baseline_arm"])
    base = costs[baseline]
    keys = sorted(base)
    candidates = [(float(np.mean([base[k] for k in keys])), baseline)]
    table = []
    for arm, arm_costs in costs.items():
        improvement = np.array([base[k] - arm_costs[k] for k in keys])
        se = float(improvement.std(ddof=1) / math.sqrt(len(keys))) if len(keys) > 1 else 0.0
        mean_cost = float(np.mean([arm_costs[k] for k in keys]))
        table.append({"arm": arm, "value": values[arm], "mean_cost": mean_cost,
                      "improvement_vs_baseline": float(improvement.mean()), "paired_se": se})
        if arm != baseline and float(improvement.mean()) >= se:
            candidates.append((mean_cost, arm))
    chosen = min(candidates)[1]
    return {
        "tuning_panel": tune_spec["panel_id"],
        "parameter": tuning["parameter"],
        "selected_arm": chosen,
        "selected_value": values[chosen],
        "rule": "minimum mean cost among the baseline and values that improve on it by >= 1 paired SE",
        "table": table,
    }


def _training_paths(spec: dict) -> list[str]:
    training = spec["training"]
    if isinstance(training, dict):
        return list(training["sources"])
    return list(training)


def build_jobs(spec: dict, source: TrainingSource, env: EvaluationEnvironment) -> list[Job]:
    streams = dict(spec.get("seed_streams") or {})
    scenario_stream = int(streams.get("scenario", DEFAULT_SCENARIO_STREAM))
    policy_stream = int(streams.get("policy", DEFAULT_POLICY_STREAM))
    panel_overrides = dict(spec.get("overrides") or {})
    panel_reps = parse_replications(spec["replications"])
    timeouts = {"plain": 3 * 3600.0, "heavy": 8 * 3600.0, **(spec.get("timeouts_s") or {})}
    checkpoint_sha: dict[Path, str] = {}
    jobs: list[Job] = []
    for city_id in spec["cities"]:
        city = env.city(city_id)
        base = env.base_overrides(city)
        for replication in panel_reps:
            variant = env.variant_for(replication)
            for arm in spec["arms"]:
                if arm.get("cities") and city_id not in arm["cities"]:
                    continue
                if "replications" in arm and replication not in parse_replications(arm["replications"]):
                    continue
                strategy = str(arm.get("strategy", arm["arm"]))
                learned = strategy in LEARNED_STRATEGIES
                if learned and not arm.get("policies"):
                    raise ValueError(f"learned arm {arm['arm']!r} must list its policies explicitly")
                policies = [int(p) for p in arm.get("policies", [0])]
                for policy in policies:
                    overrides = dict(base)
                    if variant is not None:
                        overrides.update(variant.overrides)
                    overrides.update(panel_overrides)
                    overrides.update(arm.get("overrides") or {})
                    if any(isinstance(v, dict) and "$tuned" in v for v in overrides.values()):
                        raise RuntimeError("Unresolved $tuned value; call resolve_tuned_values first")
                    if learned:
                        checkpoint = _checkpoint_for(source, policy, arm.get("checkpoint"))
                        if checkpoint not in checkpoint_sha:
                            if not checkpoint.is_file():
                                raise FileNotFoundError(checkpoint)
                            checkpoint_sha[checkpoint] = sha256_file(checkpoint)
                        digest = checkpoint_sha[checkpoint]
                    else:
                        first = min(source.policy_dirs)
                        checkpoint = source.policy_dirs[first] / "regional_policy.pt"
                        digest = None
                    external_sha = _external_source_sha(overrides) if strategy == "external" else None
                    scenario_block = _scenario_block(spec, arm)
                    recourse_block = _recourse_block(spec, arm)
                    recourse = None
                    if recourse_block is not None:
                        controller = recourse_controller(recourse_block)
                        recourse = {"controller": str(recourse_block["controller"]), "params": controller.params()}
                    scenario = None
                    if scenario_block is not None:
                        hook = scenario_hook(scenario_block)
                        scenario = {"hook": str(scenario_block["hook"]), "params": hook.params()}
                    scenario_seed = multicity_seed(env.launch_seed, scenario_stream + city.scale_rank, replication)
                    policy_seed = multicity_seed(env.launch_seed, policy_stream + city.scale_rank, replication)
                    identity = {
                        "evaluator": EVALUATOR_VERSION,
                        "city_id": city_id,
                        "scenario_seed": scenario_seed,
                        "policy_seed": policy_seed,
                        "strategy": strategy,
                        "checkpoint_sha256": digest,
                        "overrides": overrides,
                        "dynamics_sha256": env.fingerprint_sha256,
                        "evaluation_code_sha256": evaluation_code_sha(),
                        "external_source_sha256": external_sha,
                    }
                    scenario_sha = None
                    if scenario is not None:
                        # Only hooked jobs carry this field, so every job
                        # without a hook keeps its pre-hook key.
                        scenario_sha = _scenario_source_sha()
                        identity["scenario"] = {**scenario, "source_sha256": scenario_sha}
                    recourse_sha = None
                    if recourse is not None:
                        # Only jobs with a controller carry this field.
                        recourse_sha = _recourse_source_sha()
                        identity["recourse"] = {**recourse, "source_sha256": recourse_sha}
                    heavy = bool(arm.get("heavy", strategy in HEAVY_STRATEGIES))
                    # Planner matrices are always recorded for branching
                    # strategies (passive, free); other captures are opt-in
                    # and part of the episode identity.
                    requested = sorted(
                        name for name, on in (spec.get("capture") or {}).items()
                        if on and name != "planner_matrices"
                    )
                    if requested:
                        identity["capture"] = requested
                    capture = sorted(set(requested) | ({"planner_matrices"} if heavy else set()))
                    jobs.append(
                        Job(
                            key=sha256_text(canonical_json(identity)),
                            panel_id=str(spec["panel_id"]),
                            arm=str(arm["arm"]),
                            strategy=strategy,
                            policy_replication=policy,
                            city_id=city_id,
                            city_scale_rank=int(city.scale_rank),
                            city_scenario_replication=int(replication),
                            scenario_seed=scenario_seed,
                            policy_seed=policy_seed,
                            checkpoint_path=str(checkpoint),
                            checkpoint_sha256=digest,
                            overrides=overrides,
                            evaluation_variant_id="" if variant is None else variant.variant_id,
                            heavy=heavy,
                            optional=bool(arm.get("optional", False)),
                            timeout_s=float(timeouts["heavy" if heavy else "plain"]),
                            external_source_sha256=external_sha,
                            labels={**dict(spec.get("labels") or {}), **dict(arm.get("labels") or {})},
                            capture=capture,
                            scenario=scenario,
                            scenario_source_sha256=scenario_sha,
                            recourse=recourse,
                            recourse_source_sha256=recourse_sha,
                        )
                    )
    keys = [job.key for job in jobs]
    duplicates = len(keys) - len(set(keys))
    if duplicates:
        # Two arms that resolve to the same episode are allowed (one run).
        pass
    return jobs


# ---------------------------------------------------------------------------
# store with claims (several panel processes may share one campaign)
# ---------------------------------------------------------------------------
class EpisodeStore:
    def __init__(self, campaign: Path):
        self.campaign = campaign
        self.root = campaign / "episode_store"
        self.root.mkdir(parents=True, exist_ok=True)
        self.host = socket.gethostname()

    def done(self, job: Job) -> bool:
        return job.store_paths(self.campaign)[0].is_file()

    def read(self, job: Job) -> dict:
        return read_json(job.store_paths(self.campaign)[0])

    def write(self, job: Job, record: dict) -> None:
        atomic_write_json(job.store_paths(self.campaign)[0], record)

    def _claim_path(self, job: Job) -> Path:
        return job.store_paths(self.campaign)[0].with_suffix(".claim")

    STALE_CLAIM_S = 900.0  # claims are refreshed every <= 10 s while a job runs

    def claim(self, job: Job) -> bool:
        """Claim a job for this process; False if a live process holds it."""
        path = self._claim_path(job)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json({"pid": os.getpid(), "host": self.host, "utc": utc_now()})
        for _ in range(3):
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(payload)
            try:
                os.link(temporary, path)  # atomic: the claim appears complete or not at all
                return True
            except FileExistsError:
                pass
            finally:
                temporary.unlink(missing_ok=True)
            try:
                holder = json.loads(path.read_text() or "{}")
                age = time.time() - path.stat().st_mtime
            except (OSError, json.JSONDecodeError):
                holder, age = {}, self.STALE_CLAIM_S + 1
            if holder.get("pid") == os.getpid() and holder.get("host") == self.host:
                return True
            dead = holder.get("host") == self.host and not _pid_alive(int(holder.get("pid", -1)))
            if dead or age > self.STALE_CLAIM_S:
                path.unlink(missing_ok=True)  # stale: dead holder or no heartbeat
                continue
            return False
        return False

    def heartbeat(self, job: Job) -> None:
        try:
            os.utime(self._claim_path(job))
        except OSError:
            pass

    def release(self, job: Job) -> None:
        self._claim_path(job).unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------
# scenario runtime (exogenous schedule edits, applied once per episode)
# ---------------------------------------------------------------------------
class _ScenarioRuntime:
    def __init__(self):
        self.reset(None)

    def reset(self, block: Mapping | None):
        self.block = None if block is None else dict(block)
        self.hook = None if block is None else scenario_hook(block)
        self.applied = None
        self.audit = None
        self.closures: list[dict] = []
        self.shelter_ds = None

    def begin(self, core) -> None:
        if self.hook is not None and self.applied is None:
            self.applied = self.hook.apply(core)
            # Closures are recorded for this episode's shelter store only, not
            # for the cloned stores that in-process planner branches advance.
            self.shelter_ds = core.shelterDS

    def end(self, core) -> None:
        if self.hook is not None and self.applied is not None and self.audit is None:
            self.audit = self.hook.audit(core, self.applied, list(self.closures))

    def result(self) -> dict | None:
        if self.hook is None:
            return None
        if self.applied is None:
            raise RuntimeError(f"scenario {self.block} was declared but never applied")
        return {"applied": self.applied, "audit": self.audit, "closures": list(self.closures)}


SCENARIO = _ScenarioRuntime()


def _is_live_shelter_store(store) -> bool:
    """True for the live episode's shelter store during its closure phase.

    With the recourse runtime installed (every panel worker), a closure is live
    when it happens in the live closure phase on the live core's store; this
    stays correct after in-process branches restore new store objects. Without
    it (unit tests of the scenario runtime alone) the store is compared with
    the one seen when the episode began.
    """
    try:
        from recourse.runtime import LIVE
    except ImportError:  # pragma: no cover - package missing
        LIVE = None
    if LIVE is not None and LIVE.core is not None:
        return bool(LIVE.in_live_closure) and store is getattr(LIVE.core, "shelterDS", None)
    return store is SCENARIO.shelter_ds


def install_scenario_runtime(core_class) -> None:
    """Wrap the episode loop and the shelter-closure transaction (once per process).

    ``Core.simulationEnumerator`` runs after initialization is complete and
    before the first transition, so a hook sees the initial shelters and the
    drawn exogenous schedule and edits it before anything consumes it.  The
    closure wrapper only records what the unmodified transaction returns.
    """
    import ShelterDatabase as SD

    original_loop = core_class.simulationEnumerator
    if not getattr(original_loop, "_panel_scenario", False):
        def wrapped_loop(self, *args, **kwargs):
            SCENARIO.begin(self)
            try:
                return original_loop(self, *args, **kwargs)
            finally:
                SCENARIO.end(self)

        wrapped_loop._panel_scenario = True
        core_class.simulationEnumerator = wrapped_loop
    original_close = SD.ShelterDS.closeHazardOverrunShelters
    if not getattr(original_close, "_panel_scenario", False):
        def wrapped_close(self, cellTracker, simulation_time=None):
            transitions = original_close(self, cellTracker, simulation_time=simulation_time)
            if SCENARIO.hook is not None and _is_live_shelter_store(self):
                from recourse.transaction import ORDER_IN_PROGRESS

                for shelter, released, danger in transitions:
                    ordered = int(shelter.shelterID) in ORDER_IN_PROGRESS
                    SCENARIO.closures.append({
                        "shelter_id": int(shelter.shelterID),
                        "minute": int(self.currentSimulationTime),
                        "released_population": int(released),
                        "danger": int(ORDER_IN_PROGRESS.get(int(shelter.shelterID), danger)),
                        "ordered": bool(ordered),
                    })
            return transitions

        wrapped_close._panel_scenario = True
        SD.ShelterDS.closeHazardOverrunShelters = wrapped_close


# ---------------------------------------------------------------------------
# capture hooks (record only; return values are untouched)
# ---------------------------------------------------------------------------
class _Capture:
    def __init__(self):
        self.enabled: set[str] = set()
        self.active: set[str] = set()
        self.records: dict[str, list] = {}
        self.strategy = ""

    def reset(self, strategy: str, active: Iterable[str] = ()):
        self.strategy = strategy
        self.active = set(active)
        self.records = {"planner_matrices": [], "link_density": None}


CAPTURE = _Capture()
LINK_DENSITY_EDGES = [round(0.1 * i, 2) for i in range(0, 61)]  # 0..6 ped/m2


def install_capture_hooks(names: Iterable[str]) -> None:
    names = set(names)
    CAPTURE.enabled = names
    if "planner_matrices" in names:
        import numpy as np
        import CounterfactualBranch as CB

        original = CB.CounterfactualBrancher.rollout_policy_costs
        if not getattr(original, "_panel_capture", False):
            def wrapped(self, cell_indices, **kwargs):
                matrix = original(self, cell_indices, **kwargs)
                if "planner_matrices" not in CAPTURE.active:
                    return matrix
                CAPTURE.records["planner_matrices"].append({
                    "strategy": CAPTURE.strategy,
                    "decision_index": int(kwargs.get("decision_index", -1)),
                    "remaining_deployments": int(kwargs.get("remaining_deployments", -1)),
                    "mode": str(kwargs.get("mode", "")),
                    "cells": [None if c is None or int(c) < 0 else int(c) for c in cell_indices],
                    "future_seeds": [int(s) for s in kwargs.get("future_seeds", ())],
                    "matrix": np.asarray(matrix, dtype=float).tolist(),
                })
                return matrix

            wrapped._panel_capture = True
            CB.CounterfactualBrancher.rollout_policy_costs = wrapped
    if "link_density" in names:
        import NetworkCongestion as NC

        original_snapshot = NC.PedestrianCongestionModel.snapshot
        if not getattr(original_snapshot, "_panel_capture", False):
            def wrapped_snapshot(self, pedestrians, edges):
                result = original_snapshot(self, pedestrians, edges)
                if "link_density" not in CAPTURE.active:
                    return result
                record = CAPTURE.records.get("link_density")
                if record is None:
                    record = {
                        "density_edges": LINK_DENSITY_EDGES,
                        "person_substeps": [0] * (len(LINK_DENSITY_EDGES) - 1),
                        "link_substeps": [0] * (len(LINK_DENSITY_EDGES) - 1),
                        "jam_density": float(self.jam_density_ped_per_m2),
                        "shape": float(self.shape),
                        "minimum_speed_ratio": float(self.minimum_speed_ratio),
                        "effective_width_m": float(self.effective_width_m),
                        "substeps": 0,
                    }
                    CAPTURE.records["link_density"] = record
                record["substeps"] += 1
                top = len(LINK_DENSITY_EDGES) - 2
                for state in result[1]:
                    index = min(top, int(float(state.density_ped_per_m2) / 0.1))
                    record["person_substeps"][index] += int(state.occupancy)
                    record["link_substeps"][index] += 1
                return result

            wrapped_snapshot._panel_capture = True
            NC.PedestrianCongestionModel.snapshot = wrapped_snapshot


# ---------------------------------------------------------------------------
# worker process
# ---------------------------------------------------------------------------
def _peak_rss_mb() -> float:
    import resource

    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (1024.0 * 1024.0) if platform.system() == "Darwin" else peak / 1024.0


def execute_job(job: dict, *, capture: Iterable[str] = ()) -> dict:
    """Run one episode exactly as multicity_backtest's evaluation loop does."""
    from backtest import _run_episode

    artifacts = Path(job["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    CAPTURE.reset(job["strategy"], job.get("capture", ()))
    SCENARIO.reset(job.get("scenario"))
    from recourse.runtime import LIVE

    LIVE.reset(None if job.get("recourse") is None else recourse_controller(job["recourse"]))
    row = _run_episode(
        replication=int(job["city_scenario_replication"]),
        machine="local",
        phase=str(artifacts),
        strategy=job["strategy"],
        train_mode=False,
        scenario_seed=int(job["scenario_seed"]),
        policy_seed=int(job["policy_seed"]),
        checkpoint_path=job["checkpoint_path"],
        # Evaluation never writes learner diagnostics; point away from shards.
        diagnostics_path=str(artifacts / "unused_learner_diagnostics.csv"),
        overrides=dict(job["overrides"]),
        visualization_enabled=False,
    )
    captured = {}
    if CAPTURE.records.get("planner_matrices"):
        captured["planner_matrices"] = str(artifacts / "planner_matrices.json")
        atomic_write_json(artifacts / "planner_matrices.json", CAPTURE.records["planner_matrices"])
    if CAPTURE.records.get("link_density"):
        captured["link_density"] = str(artifacts / "link_density.json")
        atomic_write_json(artifacts / "link_density.json", CAPTURE.records["link_density"])
    result = {"row": row, "captured": captured}
    scenario = SCENARIO.result()
    if scenario is not None:
        atomic_write_json(artifacts / "scenario.json", scenario)
        result["scenario"] = scenario
    recourse = LIVE.result()
    if recourse is not None:
        atomic_write_json(artifacts / "recourse.json", recourse)
        result["recourse"] = recourse
    LIVE.reset(None)
    return result


def _worker_main(connection, settings: dict) -> None:  # pragma: no cover - subprocess
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Test seam only: PANEL_EVAL_TEST_EXECUTOR=module:function replaces the
    # simulator so the scheduler can be tested without the map stack.
    test_executor = os.environ.get("PANEL_EVAL_TEST_EXECUTOR", "")
    core_module = None
    try:
        sys.path.insert(0, str(ROOT))
        os.chdir(str(ROOT))
        if test_executor:
            import importlib

            module_name, _, function_name = test_executor.partition(":")
            run = getattr(importlib.import_module(module_name), function_name)
        else:
            import backtest  # noqa: F401  (warm the heavy stack before reporting ready)
            import Core as core_module
            import RLBridge  # noqa: F401
            install_capture_hooks(("planner_matrices", "link_density"))
            install_scenario_runtime(core_module.Core)
            import PedestrianDatabase
            import recourse.runtime

            recourse.runtime.install(core_module.Core, RLBridge.RLBridge, PedestrianDatabase.PedDS)
            run = execute_job
        connection.send(("ready", os.getpid(), None))
    except Exception:
        connection.send(("fatal", os.getpid(), traceback.format_exc()))
        return
    last_city = None
    while True:
        try:
            job = connection.recv()
        except EOFError:
            return
        if job is None:
            return
        if core_module is not None and last_city is not None and job["city_id"] != last_city:
            core_module.Core._prepared_infra_cache.clear()
            core_module.Core._cell_partition_cache.clear()
            osm = sys.modules.get("OSMProcessor")
            if osm is not None:
                for name in ("_location_cache", "_network_feature_cache", "_building_feature_cache"):
                    getattr(osm.OSMProcessor, name, {}).clear()
            gc.collect()
        last_city = job["city_id"]
        started = time.time()
        try:
            result = run(job, capture=settings.get("capture", ()))
            result["wall_s"] = time.time() - started
            result["peak_rss_mb"] = _peak_rss_mb()
            connection.send(("done", job["key"], result))
        except Exception:
            connection.send(("error", job["key"], traceback.format_exc()))


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------
@dataclass
class _Worker:
    group: str
    process: Any
    connection: Any
    job: Job | None = None
    started: float = 0.0


class Scheduler:
    def __init__(self, store: EpisodeStore, panel_dir: Path, *, plain_workers: int,
                 heavy_workers: int, capture: Sequence[str], max_attempts: int = 2,
                 log=print):
        self.store = store
        self.panel_dir = panel_dir
        self.counts = {"plain": int(plain_workers), "heavy": int(heavy_workers)}
        self.capture = list(capture)
        self.max_attempts = int(max_attempts)
        self.log = log
        self.workers: list[_Worker] = []
        self.failures: dict[str, list[str]] = {}

    def _start_worker(self, group: str) -> _Worker:
        """Start one worker; retry start-up failures (macOS libomp races)."""
        import multiprocessing as mp

        context = mp.get_context("spawn")
        problems = []
        for attempt, pause in enumerate((0, 10, 30, 60)):
            time.sleep(pause)
            parent, child = context.Pipe(duplex=True)
            process = context.Process(
                target=_worker_main, args=(child, {"capture": self.capture}), daemon=False
            )
            process.start()
            try:
                if not parent.poll(600):
                    raise RuntimeError("no ready message within 600 s")
                kind, pid, detail = parent.recv()
                if kind != "ready":
                    raise RuntimeError(f"start-up error:\n{detail}")
            except (RuntimeError, EOFError, OSError) as exc:
                problems.append(f"attempt {attempt + 1}: {exc}")
                process.terminate()
                process.join(10)
                self.log(f"[WORKER] start failed ({problems[-1][:200]}); retrying")
                continue
            worker = _Worker(group, process, parent)
            self.workers.append(worker)
            return worker
        raise RuntimeError("Worker could not start:\n" + "\n".join(problems))

    def _stop_worker(self, worker: _Worker, *, kill: bool = False) -> None:
        try:
            if kill:
                worker.process.terminate()
            else:
                worker.connection.send(None)
        except (BrokenPipeError, OSError):
            pass
        worker.process.join(30)
        if worker.process.is_alive():
            worker.process.kill()
            worker.process.join(5)
        self.workers.remove(worker)

    def run(self, jobs: Sequence[Job]) -> dict:
        from multiprocessing.connection import wait

        queues: dict[str, list[Job]] = {"plain": [], "heavy": []}
        seen = set()
        for job in jobs:
            if job.key in seen or self.store.done(job):
                continue
            seen.add(job.key)
            queues["heavy" if job.heavy else "plain"].append(job)
        total = sum(len(q) for q in queues.values())
        deferred: list[Job] = []  # claimed by another live panel process
        attempts: dict[str, int] = {}
        finished = 0
        durations: dict[str, list[float]] = {"plain": [], "heavy": []}
        self.log(f"[PANEL] {total} episodes to run "
                 f"(plain={len(queues['plain'])}, heavy={len(queues['heavy'])}); "
                 f"workers plain={self.counts['plain']} heavy={self.counts['heavy']}")
        if total and any(self.counts[g] <= 0 and queues[g] for g in queues):
            raise ValueError("A worker group with queued episodes has zero workers")
        try:
            while True:
                # 1. assign work to idle workers, starting workers as needed
                for group in ("plain", "heavy"):
                    queue = queues[group]
                    while queue:
                        idle = [w for w in self.workers if w.group == group and w.job is None]
                        if not idle:
                            members = sum(w.group == group for w in self.workers)
                            if members >= self.counts[group]:
                                break
                            idle = [self._start_worker(group)]
                        job = queue.pop(0)
                        if self.store.done(job):
                            continue
                        if not self.store.claim(job):
                            deferred.append(job)
                            continue
                        if self.store.done(job):  # finished elsewhere between the checks
                            self.store.release(job)
                            continue
                        payload = asdict(job)
                        payload["artifacts_dir"] = str(job.store_paths(self.store.campaign)[1])
                        worker = idle[0]
                        try:
                            worker.connection.send(payload)
                        except (BrokenPipeError, OSError):
                            # an idle worker died: replace it and requeue the job
                            self.store.release(job)
                            queue.insert(0, job)
                            self._stop_worker(worker, kill=True)
                            continue
                        worker.job, worker.started = job, time.time()
                    if not queue:
                        for worker in [w for w in self.workers if w.group == group and w.job is None]:
                            self._stop_worker(worker)
                busy = [w for w in self.workers if w.job is not None]
                if not busy:
                    deferred = [job for job in deferred if not self.store.done(job)]
                    if not deferred:
                        break
                    for job in list(deferred):
                        if self.store.claim(job):
                            self.store.release(job)
                            deferred.remove(job)
                            queues["heavy" if job.heavy else "plain"].append(job)
                    if deferred and not any(queues.values()):
                        self.log(f"[WAIT] {len(deferred)} episode(s) are running in another panel process")
                        time.sleep(30)
                    continue
                # 2. wait for a result, a crash or a timeout
                ready = wait([w.connection for w in busy] + [w.process.sentinel for w in busy], timeout=10)
                now = time.time()
                for worker in busy:
                    self.store.heartbeat(worker.job)
                for worker in busy:
                    job = worker.job
                    message = None
                    if worker.connection in ready:
                        try:
                            message = worker.connection.recv()
                        except (EOFError, OSError):
                            message = ("crash", job.key, "worker connection closed")
                    elif worker.process.sentinel in ready or not worker.process.is_alive():
                        message = ("crash", job.key, f"worker exited with code {worker.process.exitcode}")
                    elif now - worker.started > job.timeout_s:
                        message = ("timeout", job.key, f"exceeded {job.timeout_s:.0f} s")
                    if message is None:
                        continue
                    kind, _, detail = message
                    if kind == "done":
                        record = {
                            "job": asdict(job),
                            "row": detail["row"],
                            "captured": detail["captured"],
                            "evaluation": {
                                "evaluator": EVALUATOR_VERSION,
                                "finished_utc": utc_now(),
                                "wall_s": detail["wall_s"],
                                "worker_peak_rss_mb": detail["peak_rss_mb"],
                                "worker_pid": worker.process.pid,
                                "host": self.store.host,
                                "attempt": attempts.get(job.key, 0) + 1,
                            },
                        }
                        if detail.get("scenario") is not None:
                            record["scenario"] = detail["scenario"]
                        if detail.get("recourse") is not None:
                            record["recourse"] = detail["recourse"]
                        self.store.write(job, record)
                        self.store.release(job)
                        finished += 1
                        durations[worker.group].append(float(detail["wall_s"]))
                        worker.job = None
                        left = sum(len(q) for q in queues.values()) + sum(
                            1 for w in self.workers if w.job is not None
                        )
                        self.log(
                            f"[DONE {finished}/{total}] {job.arm} p{job.policy_replication} "
                            f"{job.city_id} r{job.city_scenario_replication} "
                            f"{detail['wall_s']:.0f}s rss={detail['peak_rss_mb']:.0f}MB "
                            f"left={left} eta={self._eta(queues, durations, self.counts)}"
                        )
                        continue
                    attempts[job.key] = attempts.get(job.key, 0) + 1
                    self._append_failure(job, kind, detail, attempts[job.key])
                    self.store.release(job)
                    worker.job = None
                    # A failed episode may leave branch pools or caches behind:
                    # always replace the worker.
                    self._stop_worker(worker, kill=True)
                    if attempts[job.key] < self.max_attempts:
                        queues["heavy" if job.heavy else "plain"].append(job)
                        self.log(f"[RETRY] {job.arm} p{job.policy_replication} {job.city_id} "
                                 f"r{job.city_scenario_replication}: {kind}")
                    else:
                        self.log(f"[FAILED] {job.arm} p{job.policy_replication} {job.city_id} "
                                 f"r{job.city_scenario_replication}: {kind} "
                                 f"(see {self.panel_dir / 'failures.jsonl'})")
        finally:
            for worker in list(self.workers):
                if worker.job is not None:
                    self.store.release(worker.job)
                self._stop_worker(worker, kill=worker.job is not None)
        failed = sorted(k for k, v in attempts.items() if v >= self.max_attempts)
        optional = {job.key for job in jobs if job.optional}
        return {"ran": finished, "failed": failed,
                "failed_required": [k for k in failed if k not in optional]}

    @staticmethod
    def _eta(queues, durations, counts) -> str:
        hours = 0.0
        for group, queue in queues.items():
            if queue and durations[group]:
                median = sorted(durations[group])[len(durations[group]) // 2]
                hours = max(hours, len(queue) * median / max(1, counts[group]) / 3600.0)
        return "?" if hours == 0 else f"{hours:.1f}h"

    def _append_failure(self, job: Job, kind: str, detail: str, attempt: int) -> None:
        path = self.panel_dir / "failures.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json({
                "utc": utc_now(), "key": job.key, "arm": job.arm, "city_id": job.city_id,
                "replication": job.city_scenario_replication, "policy": job.policy_replication,
                "kind": kind, "attempt": attempt, "detail": str(detail)[-4000:],
            }) + "\n")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def prepare(cli) -> tuple[dict, TrainingSource, EvaluationEnvironment, list[Job], Path, Path]:
    spec = load_spec(resolve_path(cli.spec), cli)
    learned = any(
        str(arm.get("strategy", arm["arm"])) in LEARNED_STRATEGIES for arm in spec["arms"]
    )
    source = load_training_source(_training_paths(spec), allow_running=not learned)
    allow_change = bool(getattr(cli, "allow_dynamics_change", False))
    env = build_environment(source, check_code=not allow_change)
    if allow_change:
        # v66: evaluate frozen checkpoints under changed simulator code.  The
        # job identity must then carry the CURRENT source fingerprint, or the
        # episode store would hand back episodes simulated by the old code.
        from multicity_backtest import POLICY_DYNAMICS_SOURCES, PROJECT_ROOT
        from PolicyCache import source_fingerprint

        current = source_fingerprint(PROJECT_ROOT, POLICY_DYNAMICS_SOURCES)
        recorded = dict(source.manifest["policy_dynamics_source_sha256"])
        env.dynamics_change = sorted(
            name for name in set(current) | set(recorded)
            if current.get(name) != recorded.get(name)
        )
        env.fingerprint_sha256 = sha256_text(
            canonical_json({"recorded": recorded, "current": current})
        )
    campaign = resolve_path(spec["campaign_dir"])
    if allow_change and campaign.resolve() == (ROOT / "runs" / "paper_v65").resolve():
        raise ValueError(
            "--allow-dynamics-change must write to a new campaign directory, "
            "never to runs/paper_v65"
        )
    spec = resolve_tuned_values(spec, campaign, cli)
    jobs = build_jobs(spec, source, env)
    panel_dir = campaign / "panels" / spec["panel_id"]
    return spec, source, env, jobs, campaign, panel_dir


def _panel_manifest(spec, source, env, jobs, status: str) -> dict:
    return {
        "schema_version": 1,
        "evaluator": EVALUATOR_VERSION,
        "status": status,
        "updated_utc": utc_now(),
        "panel_id": spec["panel_id"],
        "spec_path": spec["_spec_path"],
        "spec_sha256": spec["_spec_sha256"],
        "cli_overrides": spec.get("_cli_overrides", {}),
        "tuned_values": spec.get("_tuned", []),
        "training_sources": [str(path) for path in source.source_dirs],
        "training_statuses": source.statuses,
        "launch_seed": env.launch_seed,
        "dynamics_fingerprint_sha256": env.fingerprint_sha256,
        "dynamics_change": env.dynamics_change,
        "curriculum_sha256": env.curriculum.source_sha256,
        "city_profile_sha256": env.suite.source_sha256,
        "jobs": [
            {"key": j.key, "arm": j.arm, "strategy": j.strategy, "policy": j.policy_replication,
             "city_id": j.city_id, "replication": j.city_scenario_replication,
             "scenario_seed": j.scenario_seed, "heavy": j.heavy, "optional": j.optional}
            for j in jobs
        ],
        "python": sys.version,
        "host": socket.gethostname(),
    }


def _check_manifest_spec(panel_dir: Path, spec: dict, allow_change: bool) -> None:
    path = panel_dir / "panel_manifest.json"
    if not path.is_file():
        return
    previous = read_json(path)
    if previous.get("spec_sha256") != spec["_spec_sha256"] and not allow_change:
        raise ValueError(
            f"{spec['_spec_path']} changed since this panel started. Use a new panel_id, "
            "or pass --allow-spec-change if the change is intended (finished episodes are kept)."
        )


def command_plan(cli) -> int:
    spec, source, env, jobs, campaign, panel_dir = prepare(cli)
    store = EpisodeStore(campaign)
    unique = {}
    for job in jobs:
        unique.setdefault(job.key, job)
    done = sum(store.done(job) for job in unique.values())
    by_arm: dict[str, list[int]] = {}
    for job in jobs:
        entry = by_arm.setdefault(job.arm, [0, 0, 0])
        entry[0] += 1
        entry[1] += int(store.done(job))
        entry[2] += int(job.heavy)
    print(f"panel {spec['panel_id']}: {len(jobs)} jobs, {len(unique)} unique episodes, "
          f"{done} already in the store ({campaign / 'episode_store'})")
    print(f"launch_seed={env.launch_seed} cities={spec['cities']} "
          f"replications={parse_replications(spec['replications'])[0]}..{parse_replications(spec['replications'])[-1]}")
    for tuned in spec.get("_tuned", []):
        print(f"tuned {tuned['parameter']} from {tuned['tuning_panel']}: {tuned['selected_value']}")
    print(f"{'arm':32s} {'jobs':>6s} {'done':>6s} {'heavy':>6s}")
    for arm, (count, finished, heavy) in by_arm.items():
        print(f"{arm:32s} {count:6d} {finished:6d} {heavy:6d}")
    if cli.show_job:
        print(json.dumps(asdict(jobs[0]), indent=1, sort_keys=True))
    return 0


def command_run(cli) -> int:
    spec, source, env, jobs, campaign, panel_dir = prepare(cli)
    _check_manifest_spec(panel_dir, spec, cli.allow_spec_change)
    panel_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(panel_dir / "panel_manifest.json", _panel_manifest(spec, source, env, jobs, "running"))
    if cli.pin_threads:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
    capture = [name for name, enabled in (spec.get("capture") or {}).items() if enabled]
    log_path = panel_dir / "panel_eval.log"

    def log(message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    selected = jobs
    if cli.only_arms:
        wanted = set(cli.only_arms.split(","))
        selected = [job for job in jobs if job.arm in wanted]
    scheduler = Scheduler(
        EpisodeStore(campaign), panel_dir,
        plain_workers=cli.plain_workers, heavy_workers=cli.heavy_workers,
        capture=capture, max_attempts=cli.max_attempts, log=log,
    )
    result = scheduler.run(selected)
    store = EpisodeStore(campaign)
    missing = [job for job in jobs if not store.done(job)]
    status = "complete" if not missing else "incomplete"
    atomic_write_json(panel_dir / "panel_manifest.json", _panel_manifest(spec, source, env, jobs, status))
    log(f"[PANEL {status.upper()}] ran={result['ran']} failed={len(result['failed'])} "
        f"(required {len(result['failed_required'])}) missing={len(missing)}")
    return 0 if not result["failed_required"] else 3


def command_status(cli) -> int:
    spec, source, env, jobs, campaign, panel_dir = prepare(cli)
    store = EpisodeStore(campaign)
    table: dict[tuple, list[int]] = {}
    walls = []
    for job in jobs:
        entry = table.setdefault((job.arm, job.city_id), [0, 0])
        entry[0] += 1
        if store.done(job):
            entry[1] += 1
            walls.append(store.read(job)["evaluation"]["wall_s"])
    for (arm, city), (count, done) in sorted(table.items()):
        print(f"{arm:32s} {city:18s} {done:4d}/{count:<4d}")
    failures = panel_dir / "failures.jsonl"
    if failures.is_file():
        lines = failures.read_text().strip().splitlines()
        print(f"{len(lines)} failure record(s) in {failures}")
    done = sum(v[1] for v in table.values())
    print(f"total {done}/{len(jobs)}; median episode wall "
          f"{sorted(walls)[len(walls)//2]:.0f}s" if walls else f"total {done}/{len(jobs)}")
    return 0


def _scenario_columns(job: Job, record: dict) -> dict:
    if job.scenario is None:
        return {}
    scenario = record.get("scenario")
    if scenario is None:
        raise RuntimeError(f"episode {job.key} declares a scenario but its record has none")
    applied, audit = scenario["applied"], scenario["audit"] or {}
    columns = {"scenario_hook": job.scenario["hook"]}
    columns.update({f"scenario_{k}": v for k, v in job.scenario["params"].items()})
    columns.update({
        "scenario_targeted_shelter_ids": applied.get("targeted_shelter_ids"),
        "scenario_expected_overrun_minute": applied.get("expected_overrun_minute"),
        "scenario_all_targets_overrun": audit.get("all_targets_overrun"),
        "scenario_targets_overrun_as_designed": audit.get("targets_overrun_as_designed"),
        "scenario_first_target_overrun_minute": audit.get("first_target_overrun_minute"),
        "scenario_released_population_at_targets": audit.get("released_population_at_targets"),
        "scenario_other_overruns": audit.get("other_overruns"),
    })
    return columns


def _recourse_columns(job: Job, record: dict) -> dict:
    if job.recourse is None:
        return {}
    recourse = record.get("recourse")
    if recourse is None:
        raise RuntimeError(f"episode {job.key} declares a recourse controller but its record has none")
    executed = list(recourse.get("executed") or [])
    columns = {"recourse_controller": job.recourse["controller"]}
    columns.update({f"recourse_{k}": v for k, v in job.recourse["params"].items()})
    columns.update({
        "recourse_orders": len(recourse.get("orders") or []),
        "recourse_closed_shelters": len(executed),
        "recourse_closed_shelter_ids": [int(r["shelter_id"]) for r in executed],
        "recourse_order_minutes": [int(r["minute"]) for r in executed],
        "recourse_order_dangers": [int(r["danger"]) for r in executed],
        "recourse_first_order_minute": min((int(r["minute"]) for r in executed), default=None),
        "recourse_occupants_evacuated": int(sum(int(r["occupants_evacuated"]) for r in executed)),
        "recourse_reservations_released": int(sum(int(r["reservations_released"]) for r in executed)),
        "recourse_false_alarm_regions": int(recourse.get("false_alarm_regions", 0)),
        "recourse_checks": int(recourse.get("checks", 0)),
        "recourse_forecasts": int(recourse.get("forecasts", 0)),
        "recourse_evaluations": len(recourse.get("evaluations") or []),
        "recourse_wall_s": float(recourse.get("wall_s", 0.0)),
    })
    return columns


def command_collect(cli) -> int:
    import csv

    spec, source, env, jobs, campaign, panel_dir = prepare(cli)
    store = EpisodeStore(campaign)
    rows, missing = [], []
    for job in jobs:
        if not store.done(job):
            missing.append(job)
            continue
        record = store.read(job)
        row = dict(record["row"])
        row.update({
            "panel_id": job.panel_id,
            "arm": job.arm,
            "deployment_strategy": job.strategy,
            "policy_replication": job.policy_replication,
            "city_id": job.city_id,
            "city_scale_rank": job.city_scale_rank,
            "city_scenario_replication": job.city_scenario_replication,
            "evaluation_variant_id": job.evaluation_variant_id,
            "evaluation_population": int(job.overrides.get("pedVol", 0)),
            "job_key": job.key,
            "episode_artifacts_dir": str(job.store_paths(campaign)[1]),
            "checkpoint_path": job.checkpoint_path if job.checkpoint_sha256 else "",
            "panel_wall_s": record["evaluation"]["wall_s"],
            "panel_worker_peak_rss_mb": record["evaluation"]["worker_peak_rss_mb"],
            "captured_planner_matrices": record["captured"].get("planner_matrices", ""),
            "captured_link_density": record["captured"].get("link_density", ""),
            **_scenario_columns(job, record),
            **_recourse_columns(job, record),
            **{f"label_{k}": v for k, v in job.labels.items()},
        })
        rows.append(row)
    maps: dict[str, set] = {}
    for row in rows:
        maps.setdefault(row["city_id"], set()).add(str(row.get("map_provenance", "")))
    inconsistent = {city: len(values) for city, values in maps.items() if len(values) > 1}
    if inconsistent:
        print(f"Episodes of one city were simulated on different map graphs: {inconsistent}. "
              "The OSM cache changed during the campaign; restore it before collecting.")
        return 4
    required_missing = [job for job in missing if not job.optional]
    if required_missing and not cli.allow_missing:
        for job in required_missing[:20]:
            print(f"missing: {job.arm} p{job.policy_replication} {job.city_id} r{job.city_scenario_replication}")
        print(f"{len(required_missing)} required episodes missing; run the panel first "
              "(or pass --allow-missing for a partial table)")
        return 2
    panel_dir.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    output = panel_dir / "evaluation_episode_summary.csv"
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (canonical_json(value) if isinstance(value, (list, dict, tuple)) else value)
                for key, value in row.items()
            })
    os.replace(temporary, output)
    status = "complete" if not missing else "partial"
    atomic_write_json(panel_dir / "panel_manifest.json", _panel_manifest(spec, source, env, jobs, status))
    print(f"[COLLECT] {len(rows)} rows ({status}) -> {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "status", "collect"):
        command = sub.add_parser(name)
        command.add_argument("--spec", required=True)
        command.add_argument("--campaign", help="override spec campaign_dir (smoke tests)")
        command.add_argument("--training", nargs="+", help="override spec training sources")
        command.add_argument("--replications", help="override replications, e.g. 1:2")
        command.add_argument("--cities", help="override cities, comma-separated")
        command.add_argument("--policies", help="restrict learned arms to these policy indices (smoke)")
        command.add_argument(
            "--allow-dynamics-change",
            action="store_true",
            help="v66: evaluate frozen checkpoints under changed simulator code "
            "(new campaign directory required; the changed files are recorded)",
        )
        if name == "plan":
            command.add_argument("--show-job", action="store_true")
        if name == "run":
            command.add_argument("--plain-workers", type=int, default=4)
            command.add_argument("--heavy-workers", type=int, default=1)
            command.add_argument("--max-attempts", type=int, default=2)
            command.add_argument("--only-arms", help="comma-separated subset of arms")
            command.add_argument("--pin-threads", action="store_true",
                                 help="set BLAS/OpenMP threads to 1 in workers")
            command.add_argument("--allow-spec-change", action="store_true")
        if name == "collect":
            command.add_argument("--allow-missing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    return {
        "plan": command_plan,
        "run": command_run,
        "status": command_status,
        "collect": command_collect,
    }[cli.command](cli)


if __name__ == "__main__":
    raise SystemExit(main())
