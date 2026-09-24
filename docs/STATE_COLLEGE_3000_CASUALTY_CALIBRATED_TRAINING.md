# State College 3,000-person casualty-calibrated recurrent training

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-09-19
- Verification Status: **VERIFIED for execution, population accounting,
  recurrent PPO accounting, and artifact integrity; NOT VERIFIED for
  convergence, policy superiority, or real-world casualty calibration**
- Version Label: `state_college_3000_recurrent_v17_c40_seed_20260918`
- Experiment type: recurrent PPO training and paired simulation backtest
- Working directory: `/Users/huali/Desktop/RLEvacuationCollab`
- Run status: complete
- Started: `2026-09-19T06:47:19.705812+00:00`
- Completed: `2026-09-19T07:13:14.226189+00:00`
- Curriculum SHA-256:
  `8fbc79e97281a2401bef2b36cfc2182a31efe62962431fa55b060450b88353b2`
- Checkpoint SHA-256:
  `c2adb1b901c00a3abb318555518186b618bec615047075d564d0d54fec5e7299`
- PPO diagnostics SHA-256:
  `f1a636ba0f1fc1e9a655f2134f7268e73ab4635a788624e62be8c8e0444baee3`

## Casualty calibration

The simulator parameter is not a per-minute or population-wide death rate. A
mean of 40 percent means a pedestrian continuously exposed to maximum danger
(Level 5) for the full 60-minute reference duration has a 40 percent cumulative
casualty probability. Danger levels 1--3 remain nonlethal; Level 4 applies half
severity; shorter exposure is converted to a duration-consistent conditional
probability. The variance of 9 percentage-points squared gives a 3-point
standard deviation across hazard sources.

The preceding 2,500-person run used a 10 percent Level-5 rate and produced 2.375
casualties per episode. Scaling its realized exposure by population and
cumulative hazard predicted 13.82 casualties per 3,000-person episode at 40
percent, or 0.461 percent of the population. The calibration was frozen before
this run. Its intended engineering band was approximately 0.3--0.8 percent
mean realized casualties: sufficiently visible to the factorized casualty
critic without making death an unavoidable dominant outcome.

Observed training incidence closely matched the prior calculation:

| Quantity | Result |
|---|---:|
| Training episodes | 8 |
| Population per episode | 3,000 individuals |
| Total simulated person-episodes | 24,000 |
| Casualties by episode | 0, 9, 0, 26, 0, 13, 50, 9 |
| Episodes with casualties | 5/8 (62.5%) |
| Mean casualties | 13.375 |
| Mean casualty fraction | 0.4458% |
| Median casualties | 9 |
| Range | 0--50 (0--1.667%) |

The broad range is expected: fatality requires the policy-dependent intersection
of pedestrian position, the stochastic hazard footprint, and danger level 4 or
5. It is preferable to a forced minimum casualty count, which would disconnect
the safety outcome from shelter decisions.

## Reward influence

With population `P=3000`, every casualty changes return by `-3/P = -0.001`.
Avoiding ten deaths therefore improves return by 0.01. Across training:

| Reward component | Mean | Range |
|---|---:|---:|
| Safe completion | 0.67233 | 0.47133 to 0.79800 |
| Casualty | -0.01338 | -0.05000 to 0 |
| Evacuation time | -0.64375 | -0.76523 to -0.55693 |
| Hazard exposure | -0.08572 | -0.17064 to -0.03476 |
| Total return | -0.07051 | -0.43164 to 0.17335 |

The casualty branch is material but does not dominate the objective. In the
50-casualty episode it changed return from -0.06113 to -0.11113, accounting for
45 percent of the absolute realized return. In the final episode its -0.009
contribution was 47 percent of the absolute net return. The factorized critic
also normalizes the casualty target separately, so it does not lose its gradient
merely because its raw scale is smaller than accumulated evacuation time.

Seven simulator boundaries recorded nonzero casualty increments. Complete
post-action accounting assigned all 107 casualties to recurrent action
intervals; the maximum absolute terminal reconciliation error was
`1.11e-16`.

## Recurrent PPO diagnostics

| Diagnostic | Result |
|---|---:|
| Complete sequences | 8 |
| Action transitions | 38 |
| Cached graph frames | 308 |
| Mean observation history | 8.1053 frames |
| Mean credit duration | 12.4211 min |
| Maximum credit duration | 29 min |
| PPO epochs | 4 |
| Optimizer steps | 8 |
| Approximate KL | 0.0001269 |
| Clip fraction | 0 |
| Casualty-head value loss | 0.44052 |
| Residual RMS | 0.00802 |

All four critic losses and other optimizer diagnostics were finite. Headwise
explained variance remained zero after this single optimizer event, so learned
value calibration has not been established. The registered convergence audit
correctly failed: eight episodes are below the 100-episode minimum and the tail
stationarity thresholds were not met.

## Held-out paired backtest

Both evaluation policies used 3,000 pedestrians, the same 40 percent conditional
Level-5 casualty distribution, identical scenario seeds, identical initial
observations, and byte-identical exogenous hazard trajectories.

| Pair | Policy | Return | Safe | Casualties | Unfinished |
|---:|---|---:|---:|---:|---:|
| 1 | RL | 0.12300 | 2,312 | 0 | 688 |
| 1 | Heuristic | 0.13726 | 2,358 | 0 | 642 |
| 2 | RL | 0.33787 | 2,557 | 0 | 443 |
| 2 | Heuristic | 0.33787 | 2,557 | 0 | 443 |

Both held-out seeds happened to avoid lethal exposure, so this two-scenario
backtest cannot estimate a casualty-policy contrast. Mean RL return improvement
was -0.00713 with bootstrap interval `[-0.01426, 0]`; the performance assessment
is **inconclusive**, not superior. Pair 1 also shows why longer training is
required: after one PPO update, the learned policy selected one action different
from the heuristic and completed 46 fewer evacuations.

## Verification

- Interface parity: verified for both paired scenarios.
- Population identity held in every episode:
  `safe completed + casualties + unfinished = 3000`.
- Full repository suite: `183/183` tests passed in 19.964 seconds.
- Fallacy scan: 11/11 categories checked.
- Primary cautions: one city, one policy seed, two evaluation scenarios, no
  convergence, no empirical real-world fatality dataset, and no independently
  time-stamped preregistration.

The within-simulator common-random-number design supports paired policy
comparisons only. It does not establish that 40 percent is an empirical
State College disaster fatality rate or imply real-world causal effectiveness.

## Execution command

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/rlevac_matplotlib_cache \
/Users/huali/opt/anaconda3/envs/evacuationModel/bin/python \
  multicity_backtest.py \
  --cities state_college_pa \
  --launch-id state_college_3000_recurrent_v17_c40_seed_20260918 \
  --launch-seed 20260918 \
  --policy-replicates 1 \
  --train-episodes-per-city 8 \
  --training-curriculum config/state_college_training_curriculum_3000_recurrent_casualty_calibrated.json \
  --eval-replications-per-city 2 \
  --bootstrap-draws 1000 \
  --strategies rl,heuristic \
  --no-require-convergence \
  --visualize-eval-pairs-per-city 0 \
  --no-policy-cache \
  --override pedVol=3000 \
  --override 'hazardCasualtyRate=[40,9]'
```

## Anomalies

The terminal progress display paused for approximately 15 minutes between
timesteps 1 and 2 of the final heuristic evaluation, consistent with host sleep
or terminal suspension. The same process subsequently resumed, completed all
remaining timesteps, exited normally, and wrote a complete manifest. Scientific
outputs and paired state digests remained valid. As in the preceding engineering
run, `KMP_DUPLICATE_LIB_OK=TRUE` was required by duplicate OpenMP runtimes in the
current environment; long confirmatory training should use the clean declared
environment instead.

## Artifacts

- Run directory: `runs/state_college_3000_recurrent_v17_c40_seed_20260918`
- Curriculum:
  `config/state_college_training_curriculum_3000_recurrent_casualty_calibrated.json`
- Checkpoint: `policies/policy_001/regional_policy.pt`
- PPO diagnostics: `policies/policy_001/ppo_diagnostics.csv`
- Training summary: `training_episode_summary.csv`
- Evaluation summary: `evaluation_episode_summary.csv`
- Paired analysis: `paired_comparison_by_city.csv`
- Interface audit: `interface_parity.json`
- Convergence audit: `training_convergence_diagnostics.json`
