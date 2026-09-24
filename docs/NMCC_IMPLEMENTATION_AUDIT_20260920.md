# NMCC implementation audit and 3,000-person signal backtest

> Superseded on 2026-09-20 by
> `docs/HYBRID_NMCC_IMPLEMENTATION_AND_BACKTEST_20260920.md`. This file records
> the pre-v21 deficiencies that motivated the production implementation; its
> statements about missing configuration, learned models, checkpoint support,
> diagnostics, and policy backtesting are no longer descriptions of the live
> tree.

## Decision

The current tree contains a credible Stage-0 common-random-number brancher and
an opt-in Variant-A PPO advantage path. The paired estimator exposes a clear
local action signal on the synthetic real-dynamics testbed. The current tree
does **not**, however, support the stronger conclusion that an NMCC-trained RL
policy improves episode return: production `Core` never enables NMCC, the
multi-episode checkpoint path is broken, and no matched NMCC-on/NMCC-off
training and held-out evaluation has been executed.

## Material passport

- Date: 2026-09-20 (America/Los_Angeles)
- Git revision: `2fa2b513f181783ec3daa7c905b30635db461a2c`
- Branch: `main`
- Worktree: dirty; the audited NMCC files are uncommitted
- Signal-test runtime: Python 3.9.12, NumPy 1.21.5, NetworkX 2.8.4
- Torch functional-test runtime: the legacy `evacuationModel` environment with
  `KMP_DUPLICATE_LIB_OK=TRUE`; this workaround is acceptable only for a
  functional check and is not accepted as numerical paper evidence
- Fresh artifact:
  `runs/nmcc_audit_3000_seed_20260920.json`
- SHA-256:
  `cfbfbd484a8090a30c03e9d358b7263590f432f86c4e163b04a1150febdbac45`

## Implementation trace

Implemented:

1. `CounterfactualBranch.capture` and `restore` copy branch-varying simulator
   state, mutable map-flow values, and random-generator state.
2. Paired deployment-versus-`WAIT` branches reuse the same disturbance state.
3. The brancher computes interval rewards and an optional bootstrapped
   counterfactual advantage.
4. `RLBridge` can collect a pre-action `WAIT` branch, attach the selected
   action's causal advantage to a transition, and substitute/blend it into the
   PPO actor objective.

Not implemented or not operational in the production path:

1. `counterfactual_credit` defaults to false and `Core.initSimulator` does not
   pass any NMCC argument to `RLBridge`; no JSON configuration can enable it.
2. This is NMCC Variant A only. The learned natural-momentum model, residual
   model, dueling `V_wait + D` critic, and robust optimization layer described
   in the framework document are not implemented.
3. `nmcc_paired_experiment.py` documents `--real-core`, but the parser has no
   such option and always builds the synthetic grid. Thus the fresh run is not
   a State College test.
4. `RLBridge._serialize_transition` calls `.detach()` on the optional
   `counterfactual_advantage`. It crashes when that field is `None`, which is
   the normal case with NMCC disabled and an incomplete multi-episode rollout.
5. Both end-to-end NMCC tests pass the removed `first_decision_time` keyword;
   the factual-invariance test additionally requests training with the
   forbidden `heuristic` strategy.
6. `RLBridge._append_training_diagnostics` does not include the emitted
   `nmcc_*` fields, so its training CSV cannot support the planned comparison
   of counterfactual and GAE advantage dispersion.

## Verification results

### Tests as committed

- Counterfactual branch tests: 10/10 passed.
- Focused NMCC integration tests: 6/8 passed; both end-to-end tests errored on
  the stale constructor keyword.
- Full suite: 200/204 passed. The four errors were the two stale end-to-end
  NMCC tests and two partial-rollout checkpoint failures caused by serializing
  a `None` counterfactual advantage.
- Diagnostic-only compatibility shim: after dropping the obsolete constructor
  keyword, the live one-episode RL/PPO counterfactual collection test passed.
  The factual-invariance test then stopped at its separate invalid
  `train_mode=True, deployment_strategy="heuristic"` setup. No repository file
  was changed by this shim.

### Fresh paired-effect run

Command:

```text
/Users/huali/opt/anaconda3/bin/python nmcc_paired_experiment.py \
  --population 3000 --horizon 6 --tapes 6 --cells-per-epoch 6 \
  --epochs 6 --decision-interval 5 --first-decision 6 \
  --seed 20260920 \
  --output runs/nmcc_audit_3000_seed_20260920.json
```

This run uses the real hazard, pedestrian, congestion, shelter, routing, and
reward dynamics on the synthetic 16-by-16-node grid, aggregated into a 6-by-6
regional action space. It sampled six candidate cells at each of six decision
epochs and evaluated six disturbance tapes.

| Diagnostic | Result | Gate |
|---|---:|---:|
| Paired effect SD | 0.000478 | — |
| Unpaired effect SD | 0.002000 | — |
| Variance reduction | 94.30% | at least 50%: pass |
| Between-cell signal SD | 0.024313 | — |
| Paired signal/noise | 46.79 | at least 1: pass |
| Unpaired signal/noise | 22.82 | — |
| Paired rank recovery | 0.959 | at least 0.6: pass |
| Unpaired rank recovery | 0.712 | paired exceeds by 0.1: pass |
| Mean-effect shift z | 0.00 | at most 2: pass |

All five registered estimator gates passed. The paired estimator therefore
provides a stable, action-discriminating signal on this testbed and sharply
reduces natural-trajectory noise. This establishes estimator validity, not
policy learning or reward improvement.

## What can and cannot be concluded

Supported now:

- Snapshot/restore and common-noise pairing work on the synthetic real-dynamics
  harness.
- Pairing makes candidate-cell effects substantially easier to distinguish.
- The core PPO integration can execute one training episode when the stale
  test argument is removed diagnostically.

Not supported now:

- that production State College training uses NMCC;
- that training return has a positive trend;
- that a final NMCC policy beats its initialization, ordinary recurrent PPO,
  or any heuristic on held-out matched scenarios;
- that casualty, time-to-safety, or exposure outcomes improve.

The next valid experiment is a matched NMCC-on versus NMCC-off training study,
with the wiring and checkpoint failures repaired first. It must report actor
advantage dispersion, learning curves, action divergence, and held-out paired
policy outcomes. A return curve alone is insufficient.
