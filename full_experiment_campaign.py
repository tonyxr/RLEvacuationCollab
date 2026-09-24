#!/usr/bin/env python3
"""Run the complete staged-training and full-factorial evaluation campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_CURRICULUM = PROJECT_ROOT / "config" / "staged_training_curriculum_5000_convergence.json"
DEFAULT_DESIGN = PROJECT_ROOT / "config" / "evacuation_factorial_experiment.json"

# The campaign is a batch/headless workflow, including every subprocess it
# launches.  Establish safe defaults here so launchd, nohup, CI, and plain
# terminal execution cannot accidentally initialize the macOS GUI backend or
# rebuild Matplotlib's cache in an unwritable home directory.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "rlevac_matplotlib_cache"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_status(path: Path) -> str | None:
    if not path.exists():
        return None
    return str(json.loads(path.read_text(encoding="utf-8")).get("status"))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _behavior_gate_passes(path: Path) -> bool:
    return bool(_read_json(path).get("qualifies_as_learning_well_cross_city", False))


def _run_step(
    *,
    campaign: dict,
    campaign_path: Path,
    name: str,
    command: list[str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    campaign["steps"][name] = {
        "status": "running",
        "started_utc": _now(),
        "command": command,
        "log": str(log_path),
    }
    _write_json(campaign_path, campaign)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        campaign["steps"][name].update(
            {
                "status": "failed",
                "failed_utc": _now(),
                "returncode": int(completed.returncode),
            }
        )
        _write_json(campaign_path, campaign)
        raise RuntimeError(f"Campaign step {name!r} failed; see {log_path}")
    campaign["steps"][name].update(
        {
            "status": "complete",
            "completed_utc": _now(),
            "returncode": 0,
        }
    )
    _write_json(campaign_path, campaign)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-id", default="single_policy_campaign_5000_h3_20260915"
    )
    parser.add_argument(
        "--training-launch-id", default="single_policy_training_5000_h3_20260915"
    )
    parser.add_argument(
        "--factorial-launch-id", default="single_policy_full_factorial_20260915"
    )
    parser.add_argument("--launch-seed", type=int, default=20260915)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--policy-replicates",
        type=int,
        default=1,
        help=(
            "Number of independently trained policies. This campaign is designed "
            "for exactly one fixed policy; multi-seed sensitivity runs use "
            "multicity_backtest.py directly."
        ),
    )
    parser.add_argument("--train-episodes-per-city", type=int, default=120)
    parser.add_argument(
        "--behavior-gate-replications-per-city",
        type=int,
        default=5,
        help=(
            "Held-out 5,000-person RL-versus-heuristic pairs used to prove that "
            "a converged policy acquired useful, casualty-nonworsening behavior."
        ),
    )
    parser.add_argument("--curriculum", default=str(DEFAULT_CURRICULUM))
    parser.add_argument("--design", default=str(DEFAULT_DESIGN))
    parser.add_argument("--factorial-replications", type=int, default=None)
    parser.add_argument("--allow-nonconverged-source", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--visualize-first-replication",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)
    if (
        args.policy_replicates != 1
        or args.train_episodes_per_city <= 0
        or args.behavior_gate_replications_per_city < 2
    ):
        parser.error(
            "the single-policy campaign requires --policy-replicates 1, positive "
            "training counts, and at least two held-out replications per city"
        )
    if args.factorial_replications is not None and args.factorial_replications <= 0:
        parser.error("--factorial-replications must be positive")
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        from CityProfiles import load_city_suite
        from TrainingCurriculum import load_training_curriculum
        from factorial_backtest import execution_plan, load_design

        suite = load_city_suite()
        curriculum = load_training_curriculum(args.curriculum)
        design = load_design(args.design)
        if curriculum.episodes_per_city != args.train_episodes_per_city:
            raise ValueError(
                "Campaign training budget differs from the curriculum total"
            )
        plan = execution_plan(len(suite.cities), args.policy_replicates, design)
        if args.factorial_replications is not None:
            plan["scenario_replications_per_factor_cell"] = int(
                args.factorial_replications
            )
            plan["total_episodes"] = (
                plan["factor_cells"]
                * int(args.factorial_replications)
                * plan["episodes_per_scenario_replication"]
            )
        print(
            json.dumps(
                {
                    "cities": [city.city_id for city in suite.cities],
                    "training": {
                        "episodes_per_city_per_policy": curriculum.episodes_per_city,
                        "policy_replicates": int(args.policy_replicates),
                        "total_episodes": (
                            curriculum.episodes_per_city
                            * len(suite.cities)
                            * int(args.policy_replicates)
                        ),
                        "curriculum": str(Path(args.curriculum).resolve()),
                    },
                    "behavior_gate": {
                        "population": 5000,
                        "hazard_instances": 3,
                        "panic_rate": 0.5,
                        "replications_per_city": int(
                            args.behavior_gate_replications_per_city
                        ),
                        "strategies": ["rl", "heuristic"],
                        "total_episodes": (
                            len(suite.cities)
                            * int(args.behavior_gate_replications_per_city)
                            * (int(args.policy_replicates) + 1)
                        ),
                        "required": "fixed_policy_learning_well_cross_city",
                        "inference_scope": (
                            "conditional on one frozen trained policy; held-out "
                            "scenario seeds are resampled within each fixed city"
                        ),
                    },
                    "evaluation": plan,
                    "design": str(Path(args.design).resolve()),
                    "policy_cache_enabled": True,
                    "convergence_gate": not bool(args.allow_nonconverged_source),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    campaign_dir = RUNS_ROOT / args.campaign_id
    campaign_path = campaign_dir / "campaign_manifest.json"
    training_dir = RUNS_ROOT / args.training_launch_id
    factorial_dir = RUNS_ROOT / args.factorial_launch_id

    contract = {
        "launch_seed": int(args.launch_seed),
        "python": str(Path(args.python).expanduser().resolve()),
        "policy_replicates": int(args.policy_replicates),
        "train_episodes_per_city": int(args.train_episodes_per_city),
        "behavior_gate_replications_per_city": int(
            args.behavior_gate_replications_per_city
        ),
        "curriculum": str(Path(args.curriculum).expanduser().resolve()),
        "design": str(Path(args.design).expanduser().resolve()),
        "factorial_replications": args.factorial_replications,
        "allow_nonconverged_source": bool(args.allow_nonconverged_source),
        "visualize_first_replication": bool(args.visualize_first_replication),
        "training_launch_id": args.training_launch_id,
        "factorial_launch_id": args.factorial_launch_id,
    }
    if campaign_path.exists():
        if not args.resume:
            raise FileExistsError("Campaign exists; pass --resume or use a new campaign id")
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        if campaign.get("contract") != contract:
            raise ValueError("Campaign resume contract differs from the existing campaign")
    else:
        if args.resume:
            raise FileNotFoundError("Cannot resume a campaign without its manifest")
        campaign = {
            "schema_version": 1,
            "status": "running",
            "started_utc": _now(),
            "contract": contract,
            "steps": {},
        }
        _write_json(campaign_path, campaign)

    try:
        preflight_launch_id = f"{args.campaign_id}_map_preflight"
        preflight_artifact = RUNS_ROOT / preflight_launch_id / "map_preflight.json"
        if campaign.get("steps", {}).get("map_preflight", {}).get("status") not in {
            "complete",
            "reused_complete",
        }:
            _run_step(
                campaign=campaign,
                campaign_path=campaign_path,
                name="map_preflight",
                command=[
                    args.python,
                    str(PROJECT_ROOT / "multicity_backtest.py"),
                    "--launch-id",
                    preflight_launch_id,
                    "--preflight-maps-only",
                ],
                log_path=campaign_dir / "logs" / "map_preflight.log",
            )
            campaign["steps"]["map_preflight"]["artifact"] = str(
                preflight_artifact
            )
            _write_json(campaign_path, campaign)
        else:
            campaign["steps"]["map_preflight"]["artifact"] = str(
                preflight_artifact
            )

        training_manifest = training_dir / "experiment_manifest.json"
        training_status = _read_status(training_manifest)
        if training_status != "complete":
            command = [
                args.python,
                str(PROJECT_ROOT / "multicity_backtest.py"),
                "--launch-id",
                args.training_launch_id,
                "--launch-seed",
                str(args.launch_seed),
                "--policy-replicates",
                str(args.policy_replicates),
                "--train-episodes-per-city",
                str(args.train_episodes_per_city),
                "--training-curriculum",
                str(Path(args.curriculum).expanduser().resolve()),
                "--eval-replications-per-city",
                str(args.behavior_gate_replications_per_city),
                "--strategies",
                "rl,heuristic",
                "--train-only",
            ]
            if training_manifest.exists():
                command.append("--resume")
            _run_step(
                campaign=campaign,
                campaign_path=campaign_path,
                name="staged_training",
                command=command,
                log_path=campaign_dir / "logs" / "staged_training.log",
            )
        else:
            campaign["steps"]["staged_training"] = {
                "status": "reused_complete",
                "manifest": str(training_manifest),
            }
            _write_json(campaign_path, campaign)

        learning_assessment = training_dir / "learning_assessment.json"
        if not _behavior_gate_passes(learning_assessment):
            evaluation_summary = training_dir / "evaluation_episode_summary.csv"
            if evaluation_summary.exists():
                raise RuntimeError(
                    "The completed 5,000-person held-out behavior gate did not "
                    "qualify as learning well; use a new training launch after "
                    "reviewing learning_assessment.json"
                )
            behavior_command = [
                args.python,
                str(PROJECT_ROOT / "multicity_backtest.py"),
                "--launch-id",
                args.training_launch_id,
                "--launch-seed",
                str(args.launch_seed),
                "--policy-replicates",
                str(args.policy_replicates),
                "--train-episodes-per-city",
                str(args.train_episodes_per_city),
                "--training-curriculum",
                str(Path(args.curriculum).expanduser().resolve()),
                "--eval-replications-per-city",
                str(args.behavior_gate_replications_per_city),
                "--strategies",
                "rl,heuristic",
                "--bootstrap-draws",
                "5000",
                "--visualize-eval-pairs-per-city",
                "0",
                "--eval-only",
            ]
            _run_step(
                campaign=campaign,
                campaign_path=campaign_path,
                name="behavior_gate_5000",
                command=behavior_command,
                log_path=campaign_dir / "logs" / "behavior_gate_5000.log",
            )
            if not _behavior_gate_passes(learning_assessment):
                campaign["steps"]["behavior_gate_5000"]["status"] = (
                    "failed_quality_gate"
                )
                campaign["steps"]["behavior_gate_5000"]["assessment"] = str(
                    learning_assessment
                )
                _write_json(campaign_path, campaign)
                raise RuntimeError(
                    "The trained policy converged but did not pass the held-out "
                    "5,000-person RL-versus-heuristic behavior gate"
                )
        else:
            campaign["steps"]["behavior_gate_5000"] = {
                "status": "reused_complete",
                "assessment": str(learning_assessment),
            }
            _write_json(campaign_path, campaign)

        factorial_manifest = factorial_dir / "experiment_manifest.json"
        factorial_status = _read_status(factorial_manifest)
        if factorial_status != "complete":
            command = [
                args.python,
                str(PROJECT_ROOT / "factorial_backtest.py"),
                "--source-launch-dir",
                str(training_dir),
                "--launch-id",
                args.factorial_launch_id,
                "--launch-seed",
                str(args.launch_seed),
                "--design",
                str(Path(args.design).expanduser().resolve()),
            ]
            if args.factorial_replications is not None:
                command.extend(("--replications", str(args.factorial_replications)))
            if args.visualize_first_replication:
                command.append("--visualize-first-replication")
            if args.allow_nonconverged_source:
                command.append("--allow-nonconverged-source")
            if factorial_manifest.exists():
                command.append("--resume")
            _run_step(
                campaign=campaign,
                campaign_path=campaign_path,
                name="full_factorial_evaluation",
                command=command,
                log_path=campaign_dir / "logs" / "full_factorial_evaluation.log",
            )
        else:
            campaign["steps"]["full_factorial_evaluation"] = {
                "status": "reused_complete",
                "manifest": str(factorial_manifest),
            }

        campaign.update(
            {
                "status": "complete",
                "completed_utc": _now(),
                "training_manifest": str(training_manifest),
                "factorial_manifest": str(factorial_manifest),
            }
        )
        _write_json(campaign_path, campaign)
        return 0
    except Exception as exc:
        campaign.update(
            {
                "status": "failed",
                "failed_utc": _now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(campaign_path, campaign)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
