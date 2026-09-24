# Recurrent PPO reward and credit-assignment validation

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-09-19
- Verification Status: **MECHANISM VERIFIED; REWARD IMPROVEMENT NOT VERIFIED**
- Version Label: `credit_validation_results_v1`
- Model: version 17 recurrent relational GNN-PPO
- Run: `state_college_3000_credit_validation_v17_32x3_seed_20260919`
- Scope: State College, Pennsylvania; 3 independently initialized policies;
  32 training episodes per policy; 3,000 individual pedestrians per episode;
  8 paired held-out scenarios
- Started: `2026-09-19T21:52:42.507204+00:00`
- Completed: `2026-09-19T23:22:14.245067+00:00`

## Result

The recurrent architecture now transports delayed credit correctly and the
reward is completely accounted through the true terminal boundary. These
engineering claims passed every registered mechanism test. The stronger
scientific claim did not pass: reward did not improve consistently across
policy seeds, the casualty critic became unstable in the final rollout, and
the frozen policies did not outperform the heuristic on held-out return or
casualties.

| Frozen criterion | Result | Evidence |
|---|:---:|---|
| Reward telescoping error at most `1e-6` | PASS | Maximum randomized component error `5.96e-8`; real training gap `1.11e-16` |
| Delayed GAE credits every eligible action | PASS | 9,021/9,021 actions credited in the registered audit; one-step credit covered 2,000/9,021 |
| Nonzero actor and casualty-critic gradient at 10/30/60 frames | PASS | Registered checkpoint and all 3 final checkpoints passed; every reset-memory control was exactly zero |
| Positive training-return trend across seeds | **FAIL** | Mean final-minus-first block change `-0.03608`; seed changes `+0.04400`, `-0.15825`, `+0.00601` |
| Component critic calibration improves | **FAIL** | Early loss reduction reversed in the final sparse-casualty rollout; final casualty-head loss exceeded its first-block value for all 3 seeds |
| Positive held-out return improvement | **FAIL** | RL minus heuristic `-0.00596`; 95% CI `[-0.02746, 0.01076]` |
| Held-out casualty point estimate nonworsening | **FAIL** | RL `6.75` versus heuristic `6.00` casualties per episode |
| Formal convergence | **NOT TESTED TO THRESHOLD** | 32 episodes per policy; registered minimum is 100 |

Accordingly, `reward_improvement_supported=false` and
`learned_credit_improvement_supported=false`. The second result means learned
critic performance was not shown to improve; it does not negate the verified
differentiable credit path.

## Mechanism and censoring audit

The registered audit used 2,000 randomized reward trajectories, 2,000
randomized delayed-terminal GAE trajectories, and 8 graph fixtures at each of
10, 30, and 60 recurrent frames.

- Scalar telescoping error: `4.44e-16` maximum.
- Component telescoping error: `4.47e-8` maximum in the registered audit.
- GAE closed-form error: exactly `0`.
- Cross-component leakage: exactly `0`.
- Full recurrent credit coverage: `100%` of 9,021 eligible actions.
- One-step comparison coverage: `22.17%`.
- Minimum registered 60-frame actor/critic gradient across fixtures: nonzero;
  the global minima across all horizons were `6.63e-10` and `5.93e-10`.
- Reset-memory actor/critic gradients: exactly `0`.

All 12 production PPO updates used 8 complete episode sequences. The minimum
mean observation history was 8 frames and the maximum physical action-credit
duration was 49 minutes. No recurrent episode was split into an unrelated
transition minibatch.

The same gradient audit was repeated after training on all three frozen
checkpoints. Every checkpoint passed. Across the final policies, the minimum
full-memory actor and casualty-critic gradient norms were `2.12e-9` and
`2.96e-9`, while all reset-memory norms remained exactly zero. Optimization
therefore did not erase the long-lag computational path.

The full simulator corroborated the synthetic accounting test: maximum
absolute reward-accounting error was `1.11e-16` across 96 training episodes.
Casualties, exposure, active-person time, and safe completions after the last
shelter decision were included through the true environment terminal. There
is no evidence of the prior terminal censoring defect.

## Training reward

Mean episode return by policy seed and 8-episode on-policy rollout block:

| Policy seed | Block 1 | Block 2 | Block 3 | Block 4 | Block 4 − Block 1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.00571 | 0.03605 | -0.06957 | 0.04971 | +0.04400 |
| 2 | 0.06351 | 0.09885 | 0.04223 | -0.09474 | -0.15825 |
| 3 | -0.04023 | 0.12873 | -0.10931 | -0.03422 | +0.00601 |
| Seed-equal mean | 0.00966 | 0.08788 | -0.04555 | -0.02641 | **-0.03608** |

The mean within-seed linear slope was `-0.02417` return per rollout block.
The three-seed bootstrap interval for the endpoint change was
`[-0.15825, 0.04400]`; the exact sign-flip p-value was `1.0`. Two seeds ended
slightly above their initial block, but one materially negative seed dominated
the mean. This is not reproducible reward improvement.

Reward components moved materially, so the flat result is not caused by a
constant reward logger:

| Component | Mean | SD | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Total return | 0.00639 | 0.22260 | -0.62611 | 0.40918 |
| Safe completion | 0.69993 | 0.12169 | 0.35467 | 0.93167 |
| Casualty penalty | -0.00610 | 0.01031 | -0.06200 | 0.00000 |
| Evacuation-time penalty | -0.62379 | 0.08298 | -0.82154 | -0.46371 |
| Hazard-exposure penalty | -0.06365 | 0.03237 | -0.15924 | -0.01636 |

The calibrated lethality was active but not overwhelming: 586 casualties were
recorded over 288,000 simulated person-episodes (`0.203%`), with nonzero
casualties in 50/96 training episodes. Mean casualties per episode were 6.10,
and the range reached 62. The casualty signal is therefore not structurally
zero. It remains the smallest-variance objective branch, which makes its critic
more sample-sensitive than the dense safe/time/exposure branches.

## Critic behavior and diagnosis

Seed-equal PPO diagnostics at each optimizer event:

| Block | Total value loss | Casualty-head loss | Exposure-head loss | Aggregate explained variance | Casualty explained variance | Gradient norm |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.47205 | 0.57763 | 0.42600 | 0.00000 | 0.00000 | 38.87 |
| 2 | 0.30621 | 0.28541 | 0.19907 | 0.00375 | 0.00199 | 3.25 |
| 3 | 0.26586 | 0.30615 | 0.17255 | 0.00203 | 0.00886 | 11.72 |
| 4 | 0.73802 | 2.38729 | 0.13938 | 0.01019 | -0.13155 | 106.95 |

The stronger critics learned useful structure initially: total value loss fell
through block 3 for every seed, and the dense evacuation-time and exposure
heads continued improving in block 4. The casualty head did not remain stable.
Mean casualties per episode shifted from 3.04 to 13.83 to 7.13 to 0.42 across
the four rollout blocks. The final low-event block followed casualty-heavy
middle blocks, causing a target-distribution shift: casualty-head loss ended
higher than its first-block value for all three seeds, casualty explained
variance became negative on average, and critic gradient norms spiked.

Actor updates themselves were conservative: maximum approximate KL was
`0.000722`, clip fraction was always zero, and the learning rate remained
`0.0003`. Thus the observed failure is not an oversized PPO policy step. The
evidence instead implicates limited, unstratified casualty samples and a
nonstationary component target across small eight-episode rollouts.

## Held-out paired backtest

All three policies were frozen before the eight held-out scenario seeds were
opened. Each policy/heuristic comparison used the same initial observation,
candidate table, hazard trajectory, and independent pedestrian random streams;
24 interface-parity comparisons passed.

| Benefit-oriented outcome | RL mean | Heuristic mean | RL improvement | Two-way bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Episode return | 0.02432 | 0.03029 | -0.00596 | [-0.02746, 0.01076] |
| Safe completions | 2105.00 | 2123.88 | -18.88 | [-61.00, 10.75] |
| Casualties avoided | 6.75 casualties | 6.00 casualties | -0.75 | [-3.00, 0.17] |
| Unfinished avoided | 888.25 unfinished | 870.13 unfinished | -18.13 | [-62.08, 10.92] |
| RMTS reduction | 36.56 min | 36.76 min | +0.207 min | [-0.418, 1.126] |
| Normalized risk-time reduction | 0.67059 | 0.67167 | +0.00108 | [-0.00846, 0.01287] |

The primary return randomization p-value was `0.25`. Mean return differences
by policy seed were `-0.00343`, `-0.01123`, and `-0.00323`; none of the three
checkpoints improved the held-out mean. The policies agreed with the heuristic
on 76.7%–88.3% of evaluation decisions, so training did produce different
actions, but those deviations were not beneficial on average.

## Interpretation

The recurrent GNN, complete terminal accounting, duration-aware component GAE,
and whole-episode PPO solve the structural credit-path and censoring problems.
They are necessary, but this campaign shows they are not sufficient for stable
learning with the current sample schedule.

The next training change should target casualty-critic variance rather than
blindly increasing hazard severity. Raising severity would create more deaths
but could worsen gradient noise and change the planning problem. A defensible
next experiment is:

1. balance training scenario generation over prespecified hazard-dose/risk
   strata while keeping evaluation naturalistic;
2. increase the on-policy rollout to include enough casualty-bearing and
   zero-casualty episodes in every update;
3. normalize or robustify each critic head (for example, per-head target
   normalization and Huber loss) while leaving the declared scalar reward
   unchanged;
4. add a dense auxiliary hazard-dose/survival-risk prediction head that helps
   representation learning but does not alter the policy objective;
5. run at least the registered 100 episodes per policy and substantially more
   than eight held-out scenarios.

An explicit recurrent-versus-memory-reset training ablation would be required
to attribute any later policy improvement specifically to recurrence rather
than to the other simultaneous fixes. The present reset-memory gradient audit
isolates computational connectivity, not end-to-end policy efficacy.

## Verification and reproducibility

The complete repository suite passed after the campaign:

```text
Ran 183 tests in 6.382s
OK
```

The registered fallacy scan covered 11/11 categories. It found no Simpson
reversal, post-treatment adjustment, survivor deletion, or reverse temporal
ordering. The ecological and garden-of-forking-paths cautions remain: these are
system-level simulation results, and the repository protocol was frozen before
this campaign but not independently time-stamped before all model development.
The result applies to this simulated State College environment and does not
establish real-world causal effectiveness.

The campaign command was:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/rlevac_matplotlib_cache \
/Users/huali/opt/anaconda3/envs/evacuationModel/bin/python \
  multicity_backtest.py \
  --cities state_college_pa \
  --launch-id state_college_3000_credit_validation_v17_32x3_seed_20260919 \
  --launch-seed 20260919 \
  --policy-replicates 3 \
  --train-episodes-per-city 32 \
  --training-curriculum config/state_college_training_curriculum_3000_recurrent_credit_validation_32.json \
  --eval-replications-per-city 8 \
  --bootstrap-draws 10000 \
  --strategies rl,heuristic \
  --no-require-convergence \
  --visualize-eval-pairs-per-city 0 \
  --no-policy-cache \
  --override pedVol=3000 \
  --override 'hazardCasualtyRate=[40,9]'
```

The named environment contains duplicate OpenMP runtimes, so the documented
single-thread workaround was used. Timing measurements are not interpreted.
The campaign itself completed once without retry. A development smoke test
before registration exposed and corrected an expected disconnected-gradient
reporting edge case. A later deterministic report-integration error was also
corrected without rerunning any experiment.

## Artifact inventory

- Frozen protocol: `docs/CREDIT_ASSIGNMENT_VALIDATION_PROTOCOL_20260919.md`
- Registered mechanism audit: `runs/credit_assignment_mechanism_validation_seed_20260919/validation.json`
- Full analysis: `runs/state_college_3000_credit_validation_v17_32x3_seed_20260919/credit_assignment_validation_analysis.json`
- Training ledger: `runs/state_college_3000_credit_validation_v17_32x3_seed_20260919/training_episode_summary.csv`
- Evaluation ledger: `runs/state_college_3000_credit_validation_v17_32x3_seed_20260919/evaluation_episode_summary.csv`
- Paired inference: `runs/state_college_3000_credit_validation_v17_32x3_seed_20260919/paired_comparison_by_city.csv`
- Interface audit: `runs/state_college_3000_credit_validation_v17_32x3_seed_20260919/interface_parity.json`
- Convergence audit: `runs/state_college_3000_credit_validation_v17_32x3_seed_20260919/training_convergence_diagnostics.json`
- Final checkpoint audits: each `policies/policy_00*/post_training_credit_audit.json`
- Policy checkpoints: each `policies/policy_00*/regional_policy.pt`

