# Convergence-gated regional-policy pilot (2026-09-03)

## Status and claim boundary

This is a reduced computational pilot, not the preregistered confirmatory
study. It uses two independent trained policies and 20 held-out stochastic
scenarios with 75 pedestrians, a 40-step simulation, 30 deployable shelter
candidates, three initial shelters, and eight common dynamic deployments.
The result supports implementation validity and experiment sizing. It does not
establish statistical superiority.

## Training result

Both policies were trained for 160 episodes using 20 complete eight-episode
on-policy PPO rollouts. The held-out set remained closed until both passed the
machine-readable convergence audit.

| Policy seed | Tail mean return | Tail entropy | Trend span (SD) | Window shift (SD) | Tail KL violations |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.16168 | 2.58663 | 0.19773 | 0.32136 | 0% |
| 2 | 0.11825 | 2.28177 | 0.03087 | 0.01087 | 0% |

The preregistered limits were 0.5 SD for trend span, 0.5 SD for adjacent-window
shift, and 10% for target-KL violations.

## Held-out matched comparison

The evaluation contains 80 episodes: both RL policies on each scenario and one
heuristic and random run per scenario. All 60 policy-versus-scenario interface
checks passed, including equal seeds, byte-identical initial observations and
masks, equal budgets, and byte-identical complete exogenous hazard trajectories.

Positive improvement values favor RL.

| Outcome | RL mean | Heuristic mean | Mean RL improvement | Two-way bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Episode return | 0.13117 | 0.12052 | 0.01065 | [-0.02725, 0.04361] |
| Safe completed | 62.425 | 62.000 | 0.425 | [-1.050, 1.826] |
| Casualties | 3.025 | 2.900 | -0.125 | [-0.550, 0.200] |
| Unfinished | 9.550 | 10.100 | 0.550 | [-1.000, 2.001] |
| Restricted mean time to safety | 24.8117 | 25.2120 | 0.4003 | [-0.0590, 0.8680] |
| Normalized risk-weighted person-time | 0.57816 | 0.58815 | 0.00998 | [-0.00283, 0.02283] |

The random policy's mean return was 0.03049. RL policy seeds 1 and 2 had mean
return improvements of 0.00199 and 0.01931 over the heuristic, respectively.
Their deterministic held-out action agreement rates with the heuristic were
65.0% and 53.8%, showing that the learned residual sometimes changed the
priority region rather than merely copying the benchmark.

The primary return interval crosses zero, so the automated classification is
`inconclusive`. With only two independent policy seeds, the exact two-sided
sign-randomization p-value is necessarily coarse (`p = 0.5`). The appropriate
next experiment is the planned five-policy, 320-episode, 50-scenario study;
the present held-out set must not be reused for hyperparameter selection.

## Primary artifacts

- `runs/learning_backtest_residual_final_2x104_seed_20260903/experiment_manifest.json`
- `runs/learning_backtest_residual_final_2x104_seed_20260903/training_convergence_diagnostics.json`
- `runs/learning_backtest_residual_final_2x104_seed_20260903/interface_parity.json`
- `runs/learning_backtest_residual_final_2x104_seed_20260903/paired_comparison.csv`
- `runs/learning_backtest_residual_final_2x104_seed_20260903/performance_assessment.json`
- `runs/learning_backtest_residual_final_2x104_seed_20260903/training_diagnostics.png`
- `runs/learning_backtest_residual_final_2x104_seed_20260903/paired_policy_comparison.png`

