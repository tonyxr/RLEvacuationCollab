# Recurrent GNN-PPO engineering backtest — 2026-09-18

## Material Passport

- Material ID: `state_college_2500_recurrent_v17_smoke_seed_20260918`
- Material type: implementation verification and paired engineering backtest
- Model: version 17 recurrent relational GNN-PPO
- Scope: State College, Pennsylvania; eight 2,500-person training episodes;
  two held-out paired RL/heuristic scenarios
- Verification status: **VERIFIED for implementation and accounting; NOT
  VERIFIED for convergence, benchmark superiority, or external validity**
- Run status: complete
- Started: `2026-09-19T06:14:45.394622+00:00`
- Completed: `2026-09-19T06:25:31.709755+00:00`
- Policy checkpoint SHA-256:
  `64c1760d88293e34f32b626b117fb00034abb80ad83e36343b118f34b2776c68`
- PPO diagnostics SHA-256:
  `0f05084d993e733a9eef19cc02c9cb4969dd035685b7ec8e796c314590cf12a9`
- Full code-source, map, curriculum, dependency, random-stream, and command
  fingerprints are recorded in `experiment_manifest.json` under the run
  directory.

## Implemented learning design

The policy retains the interpretable three-layer regional graph observation
(physical environment, hazard, and evacuee status) and adds an episode-level
LSTM. A causal momentum vector is computed at every simulator minute from the
current and preceding administrator-visible observations. Raw graph frames are
cached and replayed during PPO, so gradients pass through the observation
history instead of treating each shelter decision as an independent sample.

Each executed action now owns the complete reward interval until the next
executed action or the true environment terminal. Exhausting the installation
budget, or temporarily having no feasible candidate, no longer terminates the
last action's reward interval. The terminal reconciliation check compares the
sum of all post-action transition rewards with the independently computed
post-first-action episode objective.

The critic has four heads aligned with the declared objective:

1. safe completion;
2. casualties;
3. evacuation time; and
4. hazard exposure.

Component-specific duration-aware GAE targets train those heads, while their
sum supplies the scalar value used by PPO. Whole episodes are the recurrent
sampling unit; an episode is never split across minibatches.

## Verification results

The complete repository suite passed:

```text
Ran 183 tests in 7.715s
OK
```

The suite includes direct tests for LSTM memory, factorized-value summation,
observation-history caching, terminal reward ownership, casualty credit after
budget exhaustion, recurrent checkpoint round trips, and PPO updates.

The eight training episodes used 2,500 individually simulated evacuees, three
hazards, and a 0.5 panic susceptibility rate. The recurrent rollout contained:

| Diagnostic | Observed value |
|---|---:|
| Complete recurrent episodes | 8 |
| Action transitions | 38 |
| Cached graph frames replayed | 308 |
| Mean frames per action history | 8.1053 |
| Mean action-credit duration | 12.4211 min |
| Maximum action-credit duration | 29 min |
| PPO epochs completed | 4 |
| Optimizer steps | 8 |
| Approximate KL | 0.0000980 |
| Clip fraction | 0.0000 |
| Maximum absolute reward-accounting gap | 1.11e-16 |

All four critic losses were finite after the update: 0.4497 (safe completion),
0.5383 (casualty), 0.4539 (evacuation time), and 0.4238 (hazard exposure).
Headwise explained variance remained zero after this single update; therefore
this run verifies gradient and accounting behavior, not learned critic
calibration.

The casualty branch was active rather than structurally zero. Training episodes
contained 0–8 casualties (mean 2.375), producing casualty penalties from 0 to
-0.0096. The other reward branches also varied: safe-completion reward ranged
from 0.4708 to 0.8036, evacuation-time penalty from -0.7589 to -0.5490, and
hazard-exposure penalty from -0.1721 to -0.0347.

Mean training return was -0.0543. The first-four versus last-four descriptive
means were -0.1689 and 0.0603, respectively. This apparent increase is not a
learning estimate because the curriculum contains only eight stochastic
episodes and one optimization event.

## Held-out paired backtest

The evaluation used the State College profile's 5,000-person baseline rather
than the 2,500-person training override, providing a small out-of-training-scale
stress test. Initial-observation and hazard-trajectory digests matched within
both paired scenarios.

| Scenario | Policy | Return | Safe | Casualties | Unfinished | Normalized risk-time |
|---:|---|---:|---:|---:|---:|---:|
| 1 | RL | -0.03709 | 3,211 | 0 | 1,789 | 0.67649 |
| 1 | Heuristic | -0.03709 | 3,211 | 0 | 1,789 | 0.67649 |
| 2 | RL | 0.21754 | 3,993 | 0 | 1,007 | 0.57946 |
| 2 | Heuristic | 0.21754 | 3,993 | 0 | 1,007 | 0.57946 |

The frozen RL checkpoint agreed with the active-population heuristic on every
evaluation decision. Its paired return improvement was therefore exactly zero
with a `[0, 0]` scenario-bootstrap interval. The correct conclusion is
**inconclusive**: the implementation ran correctly, but eight training episodes
did not move the deterministic policy beyond its safe heuristic initialization.
The convergence audit also failed by design (8 episodes versus the registered
minimum of 100).

## Reproduction command

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/rlevac_matplotlib_cache \
/Users/huali/opt/anaconda3/envs/evacuationModel/bin/python \
  multicity_backtest.py \
  --cities state_college_pa \
  --launch-id state_college_2500_recurrent_v17_smoke_seed_20260918 \
  --launch-seed 20260918 \
  --policy-replicates 1 \
  --train-episodes-per-city 8 \
  --training-curriculum config/state_college_training_curriculum_2500_recurrent_smoke.json \
  --eval-replications-per-city 2 \
  --bootstrap-draws 1000 \
  --strategies rl,heuristic \
  --no-require-convergence \
  --visualize-eval-pairs-per-city 0 \
  --no-policy-cache
```

The named conda environment contains duplicate OpenMP runtimes, so this
engineering run used `KMP_DUPLICATE_LIB_OK=TRUE` and one OpenMP thread. This is
an environment workaround, not a recommended production training setup. A
clean environment from `environment.yml` should be used before long training or
runtime benchmarking.

## Interpretation safeguards

The registered fallacy scan covered 11/11 categories. It found no Simpson
reversal, outcome-conditioned sampling, post-treatment adjustment, survivor
deletion, or reverse temporal ordering. Two cautions remain: the system-level
simulation does not support individual behavioral inference, and the design was
not independently time-stamped before development. Results are conditional on
one simulated city, one trained checkpoint, and two scenarios; they neither
establish real-world causal effectiveness nor demonstrate superiority.

## Artifacts

- Run directory: `runs/state_college_2500_recurrent_v17_smoke_seed_20260918`
- Checkpoint: `policies/policy_001/regional_policy.pt`
- Recurrent diagnostics: `policies/policy_001/ppo_diagnostics.csv`
- Training episodes: `training_episode_summary.csv`
- Held-out episodes: `evaluation_episode_summary.csv`
- Paired analysis: `paired_comparison_by_city.csv`
- Convergence audit: `training_convergence_diagnostics.json`
- Interface audit: `interface_parity.json`
