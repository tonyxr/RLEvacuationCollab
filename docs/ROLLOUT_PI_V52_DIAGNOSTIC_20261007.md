# Rollout policy-iteration diagnostic — 2026-10-07

## Scope held fixed

This review did not change the complete-ledger reward, random-policy reference
potential, observations, cellular hazard model, pedestrian simulator, action
schedule, or evaluation objective. It audited only the simulator-graded rollout
target, actor fit, branch-worker lifecycle, and their diagnostics.

## Exact failures confirmed

1. **Per-state normalization erased consequence.** A noisy 0.001 action
   difference and a consequential 0.10 difference were both expanded to the
   same unit-scale target. The actor could not distinguish important decisions
   from near-ties.
2. **The executed panel was not always anchored by objective-greedy.** The
   learner could spend a full branch batch without measuring the residual
   correction against the policy it is intended to improve.
3. **Two-future reliability was inadequate.** A six-future exhaustive audit of
   one held-out state found that two-future subsamples selected the six-future
   best cell only 40% of the time; four futures increased this to 73.3%.
4. **The first global precision fit was ill-conditioned.** Its variance floor
   was the smallest positive four-tape sample variance. In the matched smoke it
   became `4.94e-11`, so a few accidental near-zero estimates dominated the
   iteration.
5. **Precision alone optimized the wrong decision importance.** Even after a
   robust median variance floor, accurately measured near-ties consumed more
   model capacity than large operational wins. Loss decreased while top-value
   policy quality decreased.
6. **Worker I/O had no deadline and interrupted construction could leak
   children.** Historical interrupted runs left workers owned by PID 1 and a
   failed startup could lose references to already-launched processes.
7. **Zero-signal late decisions polluted the SNR diagnostic with NaN.** This
   was reporting-only but obscured label quality.

## Corrections

- Rollout utilities remain in raw complete-cost units, centered only to remove
  the within-state additive constant. One scale is estimated across the entire
  frozen-policy iteration.
- Every branch panel contains the executed action and the exact
  objective-greedy base action, plus strong and least-evaluated feasible cells.
- Policy-improvement training requires at least two common futures; the
  production curriculum uses four.
- The paired-variance floor is the median positive paired variance across the
  complete iteration, not its unstable minimum.
- Pair regression weight is measured regret divided by paired standard error.
  Thus clear, repeatable cost differences dominate both noisy comparisons and
  immaterial near-ties. This changes the learner loss, not the reward.
- Persistent workers use bounded protocol reads, partial startup is
  transactionally closed, interrupted batches close the entire pool, and a
  worker watchdog exits if its parent disappears.
- SNR aggregation ignores undefined zero-signal records instead of making the
  update-level diagnostic NaN.

No KL gate, acceptance gate, fallback controller, action override, or policy
rollback was added to the rollout actor path.

## Verification evidence

- Focused learner/simulator suite: **166/166 tests passed**.
- Matched four-episode State College, 250-person smoke, four futures per
  decision:

| Actor fit | Pair accuracy before → after | Expected rollout advantage before → after | KL | Entropy before → after |
|---|---:|---:|---:|---:|
| Minimum-variance precision | 0.926 → 0.756 | 0.00206 → 0.00142 | 0.134 | 0.836 → 0.913 |
| Median-floor precision | 0.814 → 0.780 | 0.00206 → 0.00152 | 0.146 | 0.836 → 0.928 |
| Regret / standard-error weighting | **0.844 → 0.858** | **0.00206 → 0.00606** | **0.266** | **0.836 → 0.820** |

The final update reduced its aligned regression loss from `1.1241` to `0.5778`
and increased expected simulator-graded advantage by `0.00400`. Top-1 agreement
fell from 0.719 to 0.375 because several low-cost near-ties changed order; the
cost-weighted value moved in the correct direction, which is the relevant
criterion.

The one-update deterministic held-out smoke is not a completed training result.
On two matched scenarios RL beat uniform random on risk-weighted cost by
`0.01896` and on mean time to safety by `3.06` minutes, but trailed
objective-greedy by `0.05298` normalized cost and `8.92` minutes.

The foresight ablation also shows that one update is insufficient to claim a
learned forward model. Across six held-out decision states, removing current
momentum or hazard forecast changed no action. Removing temporal history changed
one action, and the history-dependent RL choice was worse by `0.09155`. The
paths receive gradients and are present in the actor, but useful foresight must
be demonstrated after the long curriculum rather than inferred from wiring.

## Operational prerequisite and next experiment

Process inspection found only historical `UE` workers from old code paths; the
repaired smokes created no new orphans. Because those kernel-stuck processes do
not respond to TERM or KILL, macOS must be restarted before the expensive run.
After restart, use
`config/state_college_training_curriculum_2500_rollout_pi_v53.json`, then run a
matched deterministic panel with enough scenario and policy seeds to measure:

- complete-ledger cost versus random and objective-greedy;
- cost-weighted rollout advantage across policy iterations;
- momentum/history/forecast ablation regret at later decisions;
- action coverage and four-future ranking reliability; and
- decision-time compute.

These are diagnostics and evaluation criteria, not training gates.
