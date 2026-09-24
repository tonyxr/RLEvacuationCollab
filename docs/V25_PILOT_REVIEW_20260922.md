# Why the v25 State College pilot's reward is not improving or converging

## Material Passport

- Origin: Claude (Cowork), at the user's request: "Review the latest run and
  version of the model code, investigate carefully why the reward is not
  improving well and converging."
- Origin Mode: `review` + `experiment`. No model code was changed.
- Origin Date: 2026-09-22
- Evidence reviewed:
  - the run `runs/state_college_2500_nmcc_score_v25/`: episode summary,
    `ppo_diagnostics.csv`, the audit JSON, the convergence JSON, and
    per-decision progress logs for all 64 episodes;
  - the code at model version 25: `RLBridge.py`, `NMCCPolicyImprovement.py`,
    `backtest.py`, and the v25 curriculum.
- New experiments:
  - `nmcc_v25_signal_audit.py`: 16 testbed episodes, 3 independent CRN tapes;
    every feasible cell branched to the full horizon, with the return
    recorded at +10, +20 and +30 steps on the same trajectories. v25 masks and
    behavior were used.
  - Learnability tests with the numpy actor/critic replica.
- Verification Status: `STATE COLLEGE FINDINGS FROM RUN ARTIFACTS; MECHANISM
  TESTS ON THE CALIBRATED SYNTHETIC TESTBED ONLY`.
  - The testbed has 43 and 77 decision states, so effects are measured with
    wide intervals.
  - The replica is not the GNN.
- Version Label: `v25_pilot_review_v1`

## 0. Verdict

The pilot learned *mechanics*, but not a *better policy*. Four problems stack,
and each on its own would keep the reward curve flat:

1. **The reward curve cannot show improvement at this sample size.**
   - About 46% of return variance is explained by how many tokens the safety
     mask allowed.
   - The remaining scenario-to-scenario SD (0.146) is two to three times any
     achievable policy effect.
   - The "not converged" verdict is partly built into the gate: it requires
     100 episodes and applies PPO's 0.015 KL limit to NMCC-PI updates.
2. **The actor moved but did not generalize.** Its out-of-sample hit rate on
   the branch-best cell is 0.41 after training, versus 0.40 for the untrained
   route-saving prior. The in-sample gain (0.41 → 0.49) is memorization of
   each rollout's labels.
3. **Most of the information needed to generalize is thrown away.**
   - Every exact branch label is used for one update and discarded, so each
     update learns from about 19 decision states.
   - The labels do not depend on the actor, so they could all be kept.
   - The testbed learning curve shows held-out improvement grows steadily with
     labelled episodes (0 → 30% of the available gain from 4 to 24 episodes)
     and has not saturated.
4. **The labels at the decision that matters are myopic and incomplete.**
   - A 20-step branch from t = 1 ends before most of the return is realized:
     86% of State College return variance accrues after t = 21.
   - With only 6 of about 18 cells branched, the truly best first cell is in
     the branched set in only 38% of testbed states.

The only component that learned a transferable ranking — the intervention-value
ensemble — is never allowed to act.

## 1. What the run shows

| quantity | value |
|---|---|
| episodes / optimizer updates / actor updates | 64 / 16 / 11 (132 actor steps) |
| mean return, SD | 0.154, 0.196 |
| last-8 minus first-8 return, 95% bootstrap | +0.027 [−0.118, +0.176] |
| return slope over full-budget episodes | −0.00007 ± 0.00110 per episode |
| trained (ep. 29–64) minus frozen (ep. 1–24), full-budget episodes | −0.016 ± 0.041 |
| episodes installing all 5 tokens | 53 / 64 |
| per-update realized KL | 0.03–0.15 (cap 0.3, never reached) |
| residual RMS | 0 → 1.66 → 1.0 (oscillating) |
| behavior entropy | 0.58 → 0.08 |
| critic explained variance | ≈ 0 throughout (≤ 0.04) |

The policy did change:

- **More deterministic.** Scores went from an entropy of 0.58 to 0.08.
- **Different first choice.** The favored first cell moved from cell 29
  (episodes 1–24) to cells 28 and 27 (episodes 29–64).

The question is whether the change was toward better cells. The evidence says
no.

## 2. The reward curve is the wrong instrument

- **Every episode is a new scenario** (new seed, new hazard trajectory).
- **Token count dominates.** The number of tokens the mask allowed explains
  46% of return variance:

  | tokens installed | episodes | mean return |
  |---:|---:|---:|
  | 5 | 53 | +0.200 |
  | 3 | 6 | −0.018 |
  | 1 | 1 | −0.688 (145 casualties) |

- **Too few episodes to see an effect.** Among full-budget episodes the SD is
  still 0.146. Detecting a +0.05 improvement from unpaired training returns
  needs about 134 episodes per arm, and +0.02 needs about 840. The whole pilot
  has 64.
- **The convergence gate is structurally unpassable for this design.** In
  `backtest.py` it requires at least 100 episodes. It also counts an update as
  a violation when `approximate_kl > 0.015`, PPO's target. NMCC-PI updates are
  designed to move 0.03–0.3 nats, so the reported
  `tail_kl_violation_rate = 0.75` is an artifact.
- **Partial budgets are a fairness problem.** 11 of 64 episodes installed
  fewer than 5 tokens. A policy comparison is not valid until realized
  capacity is equal.

**Consequence.** No training-return curve from a run of this size can show
convergence or improvement, whatever the learner does. Progress has to be
measured by *paired* evaluation: a frozen checkpoint versus route saving on the
same held-out CRN seeds, where scenario severity cancels.

## 3. The actor moved, but not toward better cells

`nmcc_pi_top1_before` is the actor's hit rate on the branch-best cell, measured
on each new rollout *before* training on it. It is the out-of-sample test:

| phase | rollouts | out-of-sample top-1 (mean) |
|---|---:|---:|
| frozen actor (route-saving prior, zero residual) | 5 | 0.40 |
| trained actor | 11 | 0.41 |

`top1_after` is measured on the same states the actor was just fitted to. The
LOG's "+0.079 top-1" (0.41 → 0.49) is this in-sample number.

- **Chance level** with about 5.5 branched cells is about 0.18. The prior
  already carries real skill, and training added none out of sample.
- **The residual oscillates.** It rose to 1.66 logits, then fell back to 0.64
  and 1.0.
- **The fit barely moves.** Fit KL falls only 25–35% per update, and
  `fit_converged_updates = 0`.

Together these are the signature of a model re-fitting a small, noisy,
near-one-hot label set each update (T = 0.03 against a within-state value SD
of 0.02–0.046) and overwriting the previous one.

Meanwhile the intervention-value ensemble, trained on the same exact branches
by regression, improves steadily. Its held-out within-state Spearman,
measured before each rollout trains it, went 0.05 → 0.25 → 0.50 in the frozen
phases and averaged about 0.40 afterwards (range 0.12–0.57). The fill-in gate
requires 0.6 against one-tape, 20-step labels, so this signal never reaches a
decision.

## 4. Why the actor cannot generalize: data, not only noise

Testbed learnability experiment (replica: encoder plus within-state advantage
head, acting on prior + prediction). Held-out gain is measured on 4 held-out
episodes, in full-horizon return, against the route-saving choice; 12 random
splits:

| labelled training episodes | held-out gain (se) | of available |
|---:|---:|---:|
| 4 | −0.000 (0.013) | 0.059 |
| 8 | +0.008 (0.006) | 0.054 |
| 16 | +0.015 (0.008) | 0.064 |
| 24 | +0.020 (0.007) | 0.067 |

- **More data keeps helping.** Held-out improvement rises steadily with the
  number of labelled episodes and has not saturated at 24.
- **Label quality is not the limit at this scale.** With 12 training episodes,
  perfect labels (3-tape mean, full horizon, all cells) give the same held-out
  gain as one-tape full-horizon labels: +0.009 and +0.009. One-tape 20-step
  labels halve it to +0.005.
- **v25 discards almost all of its labels.** The pilot produced about 300
  labelled decision states (a branch set every rollout from episode 4 on).
  The actor was only ever fitted to the most recent ~19. Keeping them is
  valid: a label is a CRN branch value under the *fixed* route-saving base
  policy, so it does not depend on the actor. Only which states get visited
  does.
- **Retention alone is not enough.** Replaying the pilot's schedule offline
  (11 updates, 4 episodes each) is less clear:
  - The v25 listwise objective with replay, and with full-horizon labels,
    stays at +0.002 ± 0.004 and −0.003 ± 0.005 after 11 updates.
  - Regression with replay gains +0.007–0.008 early, then decays as the
    residual grows and saturates.
  - Refitting from scratch on all retained labels, as in the learning curve,
    is what realizes the gain. The learner has to treat the label set as a
    growing supervised dataset, not as a stream of small incremental
    corrections.

## 5. Why the labels at the first decision are weak

- **The branch ends before the return arrives.** At the start of a State
  College episode the 20-step branch covers t = 1 → 21. Cumulative return is
  −0.006 on average at t = 21, against 0.154 at the end. The var of the return
  earned by t = 21 is 0.005; the var of the return earned after it is 0.030.
  The first-decision label mostly measures early person-time penalties, not
  safe completions.
- **The testbed confirms the loss.** A one-tape 20-step contrast ranks
  first-decision cells against the full-horizon truth at ρ = 0.47, versus
  0.56 for a one-tape full-horizon contrast. At later decisions the gap is
  small: 0.81 vs 0.80 at t = 11, 0.67 vs 0.83 at t = 21.
- **The best cell usually isn't branched.** Branching 6 of about 18 cells (top
  3 by the prior plus 3 random) contains the true best first cell in only 38%
  of states. The label then recovers 21% of the gain available at t = 1,
  against 48% when all cells are branched to the full horizon.

## 6. What to change (fundamental, not parameter tuning)

The problem is not RL credit assignment any more. NMCC exact branching has
turned it into supervised learning of a within-state cell-value function under
a fixed base policy. The design should treat it that way: rollout-labelled
fitted policy iteration.

1. **Make the label set a persistent dataset.**
   - Store every exact branch (state history, branched cells, values, tape id)
     in the checkpoint.
   - Refit the within-state value model on the whole dataset each update, with
     episode-level held-out early stopping, instead of taking 12 incremental
     steps on the newest rollout.
   - Labels do not depend on the actor, so they can also be generated *in
     parallel ahead of training*, on many cores, from scenarios visited by the
     prior. DAgger rounds are only needed to cover the states the improved
     policy visits.
2. **Act on the model that generalizes.**
   - Use the intervention-value ensemble to select cells: argmax of prior plus
     the lower confidence bound of the predicted advantage.
   - Alternatively, distill it into the actor head. Either way, drop the
     per-rollout one-hot listwise fit.
   - Replace the Spearman ≥ 0.6 gate with the criterion that matters: paired,
     held-out return gain over the prior on reserved branch labels.
3. **Spend the branch budget where the value is.**
   - At the first two decisions, branch every feasible cell to the full
     horizon.
   - Later decisions can stay truncated and sparse: the best available gain at
     t = 21 is 0.016 on the testbed, and truncation costs little there.
   - Keeping the per-episode cost the same means moving branches from late
     decisions to early ones.
4. **Measure progress by paired evaluation.**
   - Every K updates, evaluate the frozen checkpoint against route saving on a
     fixed set of held-out CRN seeds, with equal realized capacity.
   - In PI mode, retire the training-return stationarity gate and the 0.015 KL
     threshold.
5. **Fix capacity viability before any comparison.** A hazard-aware
   capacity-viability and deadline rule must guarantee both arms install the
   same number of tokens. It is already on TODO.

Plan the next run with the learning curve in mind. The testbed reaches 30% of
the available within-state gain at 24 labelled episodes with all cells
branched, and is still rising. The State College curve should be measured
first, from a parallel label-generation job, before committing to another
6-hour sequential pilot.

## 7. Limits

- The mechanism tests use the synthetic testbed, not State College. Only the
  return-timing (§5) and top-1 (§3) findings come from the State College run
  itself.
- The replica is a small MLP on hand-built per-cell features, not the GNN/LSTM.
  Its learning curve bounds what is learnable from these features, not what
  the GNN can learn.
- The testbed has 16 + 12 episodes (77 states). The learning-curve
  confidence intervals are about ±0.014.

## 8. Artifacts

- `nmcc_v25_signal_audit.py`: collect, analyze and sequential subcommands.
- `runs/nmcc_v25_signal_audit_states_20260921.pkl`: 43 states, 3 tapes, all
  cells, returns at +10/+20/+30/full.
- `runs/nmcc_v25_signal_audit_20260921.json`: horizon, target-validity and
  single-update conversion tables.
- `runs/nmcc_v25_sequential_audit_20260921.json`: 11-update schedule replay,
  variants A–F.
- `runs/nmcc_v25_learnability_20260922.json`: the perfect-label test and the
  learning curve.
- `runs/state_college_2500_nmcc_score_v25/_claude_decisions_extract.json`:
  per-decision cells and cumulative return at t = 11/21/31/41/51 for all 64
  episodes.
