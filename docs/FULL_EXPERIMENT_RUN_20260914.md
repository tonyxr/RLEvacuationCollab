## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-14T15:31:06Z
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1
- Upstream Dependencies: staged_training_results_20260913, full_experiment_suite_v3

# Full experiment execution record — 2026-09-14

> **Superseded execution record.** The processes described below were stopped
> cleanly and their manifests are marked `superseded`. The current individual-
> pedestrian, social-force, panic, Malibu, and full-factorial workflow is
> documented in `CODE_PATH_DIAGNOSTIC_20260914.md` and launched by
> `full_experiment_campaign.py`.

## Superseded experiment

- **ID:** `full_e0_e6_confirmatory_staged_v2_20260914`
- **Type:** training followed by matched simulation evaluation
- **Status:** superseded
- **Working directory:** `/Users/huali/Desktop/RLEvacuationCollab`
- **Output root:** `runs/full_e0_e6_confirmatory_staged_v2_20260914/`
- **Launch seed:** 20260914
- **Training:** 8 independent pooled policies × 120 episodes per city × 5 cities = 4,800 episodes
- **Evaluation:** 90 held-out scenarios per city across five fixed cities
- **Strategies:** RL, active-population heuristic, hazard-weighted demand, accessibility deficit, random feasible, static demand-greedy, RL precommitment, and initial-only
- **Primary metric:** `objective_episode_return`
- **Inference:** equal-city macro estimates with 20,000 hierarchical bootstrap draws
- **Checkpoint gate:** held-out evaluation is not opened unless all eight training seeds pass the recorded convergence audit

The learned `rl` and `rl_precommit` strategies are each evaluated for all eight policy seeds. Each nonlearned comparator is evaluated once per matched scenario. The planned core evaluation therefore contains 9,900 episode runs after training.

The dependent finalization campaign is active at
`runs/full_campaign_20260914/campaign_manifest.json`. It does not modify or
restart the core process. After a completed/converged source manifest is
observed, it will run the prespecified five-city reproducibility sample, the
5,625-episode population/candidate stress matrix, the 1,500-episode map matrix,
the publication reporter, and the results-chapter generator. Any child-process
failure stops the campaign and is recorded rather than silently retried.

## Staged curriculum

The run uses `config/staged_training_curriculum_confirmatory.json`, SHA-256
`f5bf6b22d6f388d28a3604b6727038c428a2e983ee2d69c0c1378c52b3322de8`.
It contains 120 episodes per city. The first 28 episodes per city span increasing
population, candidate/shelter availability, hazard count, spread, casualty, and
mobility perturbations. The final 92 episodes per city use the stationary
50,000-person nominal target environment, so the prespecified 20% convergence
tail is entirely stationary.

Weighted cohorts are capped at 20 persons per moving agent, the largest level
that passed the registered training-surrogate fidelity screen. This makes the
run computationally feasible but does not convert cohort-level outcomes into
individual-microsimulation confirmation.

## Immutable inputs retained

- Resumable staged checkpoint: `runs/staged_operational_continuation_20260913/policies/policy_001/regional_policy.pt`, SHA-256 `bcfb9b17192d840bd44591ed2f7fe276d768a308c5afa22d3761ccf2eb28c2a6`
- Evaluation-only conservative checkpoint: `runs/staged_conservative_robustness_v1_20260913/policies/policy_001/regional_policy.pt`, SHA-256 `1ee4d8df370fc674facd76674127e1dd9f3e14df67c4e68fc2fd5988d8f41341`
- E0--E6 suite: `config/full_experiment_suite.json`, SHA-256 `08bbeb9f134943d143eb5d67ca92135008f745de4269f47a2e7d1b93707e39b7`
- OR journal suite: `config/or_journal_experiment_suite.json`, SHA-256 `f5400ba2633f6fc6b019630f86d5c551903a964a58d287dac47926f31195b052`

Neither earlier checkpoint was overwritten or used to initialize the eight
independent confirmatory policies.

## Preflight and validation

- 140 automated tests passed in the compatible Apple-Silicon environment.
- Python compilation and `git diff --check` passed.
- State College and Reading map preflight passed in
  `runs/full_experiment_preflight_20260914/map_preflight.json`.
- Spokane, Seattle, and Chicago map preflight passed in
  `runs/full_experiment_preflight_remaining_20260914/map_preflight.json`.
- Horizon, two-minute action cadence, five-addition budget, map footprint, and
  congestion contracts match the registered 60-minute evaluation design.

## Outputs already generated from completed evidence

The existing staged continuation now has E1 training plots and aligned
continuation diagnostics:

- `runs/staged_operational_continuation_20260913/full_experiment_figures/01_training_reward_and_convergence.png`
- `runs/staged_operational_continuation_20260913/full_experiment_figures/02_ppo_optimization_diagnostics.png`
- `runs/staged_operational_continuation_20260913/full_experiment_figures/02_ppo_update_averages.csv`

The conservative robustness run now has descriptive policy figures:

- `runs/staged_conservative_robustness_v1_20260913/full_experiment_figures/03_primary_policy_performance.png`
- `runs/staged_conservative_robustness_v1_20260913/full_experiment_figures/04_city_specific_paired_effects.png`
- `runs/staged_conservative_robustness_v1_20260913/full_experiment_figures/06_safety_casualty_timeliness_frontier.png`
- `runs/staged_conservative_robustness_v1_20260913/full_experiment_figures/figure_statistical_summary.csv`

Their readiness manifests correctly label them `partial_available_data`; they
must not be represented as the completed eight-seed E0--E6 study.

## Execution anomalies and controls

An initial four-worker policy-shard trial produced severe host memory pressure
(load approximately 25 with about 59 MB immediately free). Two workers still
produced load approximately 48 with about 45 MB free. Workers were stopped
before committing scientific results, and the canonical single-process launch
was selected. The shard code and fail-closed assembler remain available for a
higher-memory machine, but no shard result is included in the active run.

Two obsolete start attempts are preserved rather than deleted:

- `runs/full_e0_e6_confirmatory_staged_20260914/` — two valid training episodes, stopped before correcting the evaluation replication contract
- `runs/full_e0_e6_confirmatory_staged_20260914_shard_001/` through `_004/` — resource-calibration starts, not part of the active experiment

## Scope boundary

The active command executes the full eight-seed pooled training and the
90-scenario-per-city primary/benchmark policy matrix. The repository's broader
E0--E6 contract still has reporting-only components: simulator executors and
parameter mappings for E0 calibration, the E2 capacity-by-hazard-by-demand
factorial, E3 flat-action quality, E4 ablation training, and E5 leave-one-city-out
training are not yet all implemented. No placeholder rows will be generated for
those tables. The 5,625-episode population/candidate scale stress and the
1,500-episode map matrix remain separately registered downstream runs and are
gated on converged source checkpoints.

## Current progress

At the latest manual audit, 166 of 4,800 training episodes were durably written
and the process was continuing through policy seed 1 in the stationary S7 stage.
The rolling campaign manifest is the authoritative progress record. Completion
status must also be confirmed in
`runs/full_e0_e6_confirmatory_staged_v2_20260914/experiment_manifest.json`; a
`running` manifest is not a completed experiment result.

Three fail-closed postprocessing utilities were added without altering the
running simulator process:

- `reproduce_multicity_sample.py` re-runs one prespecified matched RL/heuristic
  scenario per city and compares scientific outputs while excluding timing.
- `generate_experiment_results_chapter.py` refuses a running, incomplete,
  nonconverged, or population-imbalanced source before generating the chapter.
- `full_experiment_campaign.py` sequences downstream runs and records every
  command, exit code, log path, copied table hash, and failure state.
