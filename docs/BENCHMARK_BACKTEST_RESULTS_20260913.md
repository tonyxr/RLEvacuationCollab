# Benchmark Policy Integration Backtest — 2026-09-13

> Historical-result warning: this engineering backtest predates model version
> 13's exact-candidate action and revised observation/reward contract. It is
> retained for provenance, not as validation of the current policy.

## Scope

This is the completed engineering backtest for the benchmark-policy extension.
It verifies implementation, paired execution, accounting, analysis, and figure
generation. It is deliberately not labeled as the confirmatory five-city
performance experiment: the latter requires the registered independent policy
seeds, training horizon, scenario replications, and convergence gate.

## Executed design

- Study area: Reading, Pennsylvania OSM place graph
- Grid: 4 by 4 adaptive cells
- Population: 32
- Hazards: 2, stochastic evolution
- Shelter candidates: 10 sampled from 1,399 eligible OSM-linked sites
- Shelter budget: 2 initially active plus at most 3 additional installations
- Horizon: 8 one-minute transitions
- Training: one complete eight-episode PPO rollout
- Evaluation: two matched held-out scenarios
- Policies: RL, static initial-only, maximum active population, uniform random,
  hazard-weighted demand, and accessibility deficit

The executable command and dependency versions are preserved in
`runs/benchmark_policy_backtest_final_20260913/experiment_manifest.json`.

## Verification results

- Repository test suite: 113 tests passed.
- Five-city profile validation: passed.
- Five-city OSM preflight: all five registered maps ready, with nonempty mapped
  shelter-feature pools.
- Matched-interface audit: passed for all dynamic strategies. The audit checks
  scenario/component seeds, initial observation and mask digests, complete
  hazard-trajectory digests, deployment budgets, map/partition identity, and
  the common lower-level maximum-capacity site optimizer.
- End-to-end run status: complete; all declared tables, checkpoint, diagnostics,
  and figures exist.
- OSMnx compatibility: endpoint, timeout, and edge-speed APIs were verified
  across the 1.x and 2.x layouts used by available project environments.

## Descriptive integration results

The return is the action-count-invariant full-episode objective. Positive
improvement denotes RL minus the benchmark after applying each metric's
preferred direction.

| Benchmark | Mean RL objective | Mean benchmark objective | Mean RL improvement |
|---|---:|---:|---:|
| Static initial-only | -0.988672 | -1.075391 | 0.086719 |
| Maximum active population | -0.988672 | -0.988672 | 0.000000 |
| Uniform random feasible cell | -0.988672 | -1.055469 | 0.066797 |
| Hazard-weighted demand | -0.988672 | -0.988672 | 0.000000 |
| Accessibility deficit | -0.988672 | -0.988672 | 0.000000 |

These values are a software-integration diagnostic only. With one trained
policy seed, an eight-episode training rollout, and two evaluation scenarios,
they cannot establish superiority or equivalence. The identical outcomes for
several policies indicate that they selected outcome-equivalent deployments in
these small scenarios; this is not evidence that the policies are generally
interchangeable.

## Primary artifacts

- `runs/benchmark_policy_backtest_final_20260913/benchmark_comparison.csv`
- `runs/benchmark_policy_backtest_final_20260913/benchmark_comparison.md`
- `runs/benchmark_policy_backtest_final_20260913/all_benchmark_comparisons.png`
- `runs/benchmark_policy_backtest_final_20260913/interface_parity.json`
- `runs/benchmark_policy_backtest_final_20260913/evaluation_episode_summary.csv`
- `runs/benchmark_map_preflight_20260913/map_preflight.json`

## Remaining confirmatory gate

The registered large study remains a separate computational experiment. It
must use the full policy-seed and five-city design, pass the convergence audit
before opening held-out evaluation, and report the primary RL-minus-maximum-
active contrast separately from the secondary benchmark family. The existing
runtime audit estimates that study at more than 205 core-hours, so this report
does not misrepresent the tractable integration run as completed confirmatory
evidence.
