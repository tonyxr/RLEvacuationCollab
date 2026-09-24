# Proposal: model v26 as rollout-labelled fitted policy iteration (`fitted_value`)

## Why this is a proposal folder and not an in-place change

- **What was expected.** This implementation was written against the v25
  working tree.
- **What changed.** During the same session, another model-v26
  implementation appeared in this working tree. It uses
  `nmccPiActorObjective = "value_lcb"`, and `RLBridge.py` was still being
  edited at 19:15 UTC on 2026-09-22.
- **Overlap.** It changes the same files: `RLBridge.py`, `GNN.py`,
  `DecisionInterface.py`, `NMCCPolicyImprovement.py`, `NMCCPIConfig.py`,
  `TrainingCurriculum.py` and `backtest.py`.
- **Decision.** Overwriting them would destroy that work, and the two cannot
  be mixed blindly. The complete implementation is therefore here, and
  nothing in the main tree was modified.

## Contents

- `v26_fpi_files.zip`: every changed or new file, at its
  repository-relative path, as it should be after the change. These are
  full copies against the **v25** tree, not against the current `value_lcb`
  tree. They are zipped so that no copy of a test module is picked up by
  test discovery.
- `v26_fpi_against_v25.patch`: the same change as one unified diff, applicable
  with `git apply` to the v25 tree.
- `NMCC_FPI_V26_DESIGN_20260922.md`: design, evidence, run instructions and
  limits.
- `nmcc_v26_fpi_gate_validation_20260922.json`: gate operating
  characteristics on real CRN labels.
- `LOG.md` and `TODO.md` inside the zip: the v25 files with the v26 entries
  added.

## How the two v26 designs differ

This comparison is based on the `value_lcb` design text in
`docs/MDP_AND_OPTIMIZATION_DESIGN.md` and `README.md`. The `value_lcb`
code was not audited.

| | `value_lcb` (in tree) | `fitted_value` (this proposal) |
|---|---|---|
| Label horizon | Full horizon at the first two decisions; configured short horizon afterwards | Full horizon at every decision |
| Label storage | Bounded replay dataset inside the checkpoint | Separate contract-bound label store (atomic shards); parallel collect-only workers (`fpi_label_workers.py`) can fill it |
| Splits | Episode fit/validation | Hash of (city, seed) into fit / validation (early stopping) / holdout (gate) |
| Holdout labels | Held-out exact branches | Holdout episodes branched exhaustively at *every* decision and run without exploration, so any candidate rule is evaluated exactly |
| Gate | Lower bound of held-out paired gain over the prior, positive for a required number of updates | Performance-difference score per episode; equal-family weighting; min(cluster bootstrap, cluster-robust t) bound; Bonferroni over (model, λ, β) candidates; per-family harm test; champion versus challenger. Validated on real CRN labels: a no-skill model is certified in 0.5% of splits |
| Decision rule | Prior + (mean − κ·sd) of the intervention ensemble | Prior + λ(μ − βσ), where μ, σ are base-differenced so the unidentified per-member level cannot inflate σ; λ = 0 is the base policy exactly; λ, β chosen by the gate |
| Scale across scenarios | Reward units | Per-family normalization (city × P × H × panic) with shrinkage |
| Capacity parity | Not stated | Token catch-up for every strategy and in every branch; parity reported in interface verification |
| Staged pretraining | Natural/causal pretraining retained (2/3 rollouts) | Not used (the whole network is refit on the store each update) |
| Curricula | Schema v2 with top-level `learner_overrides` | Schema v1; strength-2 orthogonal arrays over P × H × panic (OA(25) factorial, OA(9) pilot); every variant declares the full contract |
| Evaluation | — | Applies the curriculum learner contract to every evaluated strategy; optional curriculum scenario cycling; `route_saving` benchmark strategy |

## Merge options

1. **Adopt this proposal.** Apply `v26_fpi_against_v25.patch` to a v25
   checkout, or unzip `v26_fpi_files.zip` over one. Keep `value_lcb` as an ablation by
   porting its objective into the merged tree.
2. **Port selected components onto `value_lcb`.**
   - Torch-free and self-contained: the label store, gate, family
     normalization, catch-up, `route_saving`, orthogonal-array curricula,
     `fpi_label_workers.py` and `FittedPolicyIteration.py`.
   - These need adapting to the other `RLBridge.py`: the refit, replay and
     decision-rule glue.
3. **Keep `value_lcb`.** Use this folder only as a reference.

## Verification status

- **Torch-free tests.** 107 tests across 12 modules pass: the new fitted
  policy iteration, reward-contract and curriculum tests, plus the existing
  NMCC, branch and curriculum suites.
- **Static checks.** `flake8 --select=F` is clean. `pyright` adds no errors
  over v25.
- **Review.** An independent static review found eight defects in the torch
  integration. All are fixed.
- **Not yet run.** `tests/test_fitted_policy_iteration_torch.py` and the
  full suite need the `rlevacuation` environment, which the implementing
  tool cannot reach.
