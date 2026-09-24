# Why the policy is not converging: review of v22/v23 and a headroom measurement

## Material Passport

- Origin: Claude (Cowork) code-and-artifact review, at the user's request
- Origin Mode: `review` + `experiment`. No learner code was changed. Added
  `nmcc_headroom_experiment.py` and `headroom_lib.py`; added two
  default-preserving options (`spacing_m`, `candidate_count`) to
  `nmcc_testbed.py`; committed three experiment reports under `runs/`.
- Origin Date: 2026-09-21
- Evidence reviewed: `runs/state_college_2500_nmcc_staged_v22_retry_seed20260920`
  (latest complete 64-episode model), `runs/temporal_credit_learning_backtest_v23_smoke*`,
  the v22/v23 curricula, and `RLBridge.py` / `GNN.py` at model version 23.
- Verification Status: `DIAGNOSIS SUPPORTED BY RUN ARTIFACTS; HEADROOM MEASURED ON A
  CALIBRATED SYNTHETIC MAP ONLY`. Sections 1–3 read the committed State College
  artifacts and current code. Section 4 was executed on the torch-free testbed,
  whose map is a uniform-density grid, not State College; its numbers must be
  re-measured on State College before they are quoted. Torch is still not
  installable from this environment, so no learner run was performed.
- Version Label: `convergence_review_v1`

## 0. Verdict

NMCC is doing what it was built to do: the paired targets are about 93% lower
variance than GAE in the v22 run. The policy is not converging for reasons that
sit upstream and downstream of that signal, and three of them are decisive on
their own:

1. **The policy never moves.** Across all 64 v22 episodes the behavior policy is
   close to uniform over feasible cells, the learned residual reaches 0.12
   logits, and the deployed argmax policy is the heuristic. v23 moves it even
   less (residual 0.0023 after two accepted updates). A flat learning curve is
   the expected outcome of this configuration, not evidence that learning failed
   to find something.
2. **In v23 no NMCC signal reaches the actor, and the actor no longer shapes
   useful features either.** Every NMCC actor weight is 0.0, and the parameter
   partition gives the whole encoder to the actor, so the critic and the world
   model are shallow heads on features that barely train.
3. **The problem's value sits almost entirely in the first deployment, where the
   registered heuristic is weakest and NMCC's 10-minute target sees least.** On
   a synthetic map calibrated to State College's outcomes, a simple
   rollout policy beats the heuristic by +0.21 return (implementable) to +0.26
   (perfect information); nearly all of it comes from the t = 1 decision, where
   the heuristic was not the best choice in any of the six scenarios tested and
   active population is slightly anti-correlated with true cell value.

The third point changes the strategy. There is a large prize, a free greedy
rule already captures more than half of it, and the rollout policy supplies a
dense, full-horizon, CRN-paired ranking of every cell — which is a far cleaner
training signal than any policy-gradient estimator in the current design.

## 1. What the most recent full model actually did (v22 retry, 64 episodes)

### 1.1 The behavior policy is near-uniform for the whole run

| quantity | value | reading |
|---|---|---|
| normalized behavior entropy | 0.95–0.99 in almost every episode | near-uniform over feasible cells |
| heuristic agreement rate | mean 0.18 | ~2.7x uniform chance (~1/15); far from the 0.77–0.88 of 09-19 |
| residual RMS at the six actor updates | 0.012, 0.026, 0.048, 0.103, 0.099, 0.119 | tiny against the prior |
| actor updates in the run | 6 (episodes 24–64) | ~35 decisions each, ~210 actor samples total |

The cause is in `GNN.py`:
`base_logits = HEURISTIC_PRIOR_SCALE * relative_active + learned_residual`, with
`HEURISTIC_PRIOR_SCALE = 1.0` and `relative_active = active / max(active)` in
[0, 1]. The prior's entire logit spread across cells is therefore at most 1.0,
and `RLBridge._select_rl` divides it by an action temperature of 1.5 decaying to
1.0. Over ~15 feasible cells that cannot concentrate a sampling distribution.

Evaluation uses `argmax` (`deterministic = not self.train_mode`). With a residual
far smaller than the prior's spread, the argmax is the heuristic's cell except at
near-ties. Two consequences follow:

- the training-return curve reports an exploration policy that barely changes,
  so no convergence can appear in it;
- the deployed policy is, to first order, the heuristic, so RL-minus-heuristic
  near zero is built in rather than learned. That is consistent with every
  held-out comparison so far (09-19: −0.006; v21 backtest: −0.027).

### 1.2 The trust region bounds how far the policy can travel

For small logit changes `delta`, `KL(pi_old || pi_new) ~ 0.5 Var_pi(delta)`, so
the 0.015 target caps each accepted update at roughly 0.17 logits. Moving one
cell from 1/15 to 50% probability requires about ln(14) ~ 2.6 logits: at least
16 accepted updates all pointing the same way, i.e. at least 128 episodes at 8
per rollout, before any noise. The registered 64-episode schedule cannot reach a
concentrated policy even with a perfect gradient. v22 did hit the ceiling — its
last three updates exceeded the KL target (0.021, 0.017, 0.025) — so the trust
region was binding, not idle.

### 1.3 The critic never learned

Explained variance at every v22 update: −0.006 to +0.031. It is ~0 in v23 too.
Any bootstrap term therefore contributes noise, which matters for NMCC's
`gamma^L (V_a - V_0)` tail (section 3, item 3).

### 1.4 The counterfactual signal is mostly "deploy now", not "deploy here"

At every v22 update the counterfactual advantage mean was about equal to its SD
(for example 0.027 vs 0.033). `A_CF = Q(s,a) - Q(s, WAIT)` mixes two things:
how much this cell beats other cells (what the actor needs), and how valuable
deploying anything is at this state (irrelevant when deployment is mandatory).
Section 4 measures the split directly: **62% of the target's variance is
between states.** After batch or lagged centering, that component becomes
advantage that tracks the decision's position in the episode rather than which
cell was picked.

### 1.5 What drives return

| correlation with episode return | r |
|---|---|
| people still unfinished at 60 min | −0.954 |
| safe completions | +0.966 |
| number of tokens actually deployed (1–5) | +0.625 |
| casualties (median 0, mean 7 of 2,500) | −0.544 |

Return is essentially "how many people reached a shelter within the hour". The
+0.63 with deployments is largely scenario-driven variance the policy can barely
influence: a token is lost whenever an epoch has no feasible cell (mean 4.36 of 5
deployed).

## 2. What v23 changed, and why it cannot converge either

v23's engineering is careful — transactional KL, disjoint optimizers, a
critic-independent actor target — and the audit trail is excellent. But three of
its choices remove the paths that were working:

- **No NMCC credit reaches the actor.** The v23 curriculum sets
  `nmccCounterfactualWeight`, `nmccJointCounterfactualWeight`,
  `nmccGuidanceMaximum` and `nmccTeacherCoefficient` all to 0.0. The smoke run
  confirms NMCC-on and NMCC-off actor trajectories are identical. The actor gets
  the full Monte Carlo return-to-go — the highest-variance target available —
  even though the same run measured a counterfactual SD about half the MC SD
  (0.10–0.12 vs 0.17–0.24).
- **The baseline stratum collapses to decision position.**
  `_regime_position_keys` builds `city | population | hazards | decision=k`.
  In a single-city, single-configuration run only `decision=k` varies, so
  scenario difficulty — the dominant variance source, return SD 0.186 across
  episodes — stays in the advantage untouched.
- **The partition inverts representation learning.** `critic_prefixes` gives the
  critic partition only `critic_trunk`, `critic_heads`, `natural_outcome_head`
  and `causal_outcome_heads`. The GNN encoder, message layers, LSTM and
  `candidate_hidden` projections are actor-owned and trained only by a 1e-4,
  one-epoch PPO step on a near-uniform policy. The critic and the causal ensemble
  are fitting shallow MLPs on features that barely change from initialization,
  which is the most likely reason explained variance stays at zero. In v22 the
  dense NMCC supervision was the encoder's best teacher; v23 cuts it off.

The step budget also falls to roughly 1/12 of v22's (1e-4 x 1 epoch against
3e-4 x 4). Under section 1.2's arithmetic, v23's registered run will not move the
policy measurably.

## 3. Smaller defects worth fixing

1. **Lagged-baseline cold start.** `_normalize_with_lagged_baseline` passes the
   raw return through when a stratum has no history, so the first actor rollout
   is effectively unbaselined. `_update_lagged_baseline` then starts the EMA
   variance at 0, so for early rollouts the scale falls to
   `advantage_scale_floor = 0.05` against a true SD near 0.18, inflating those
   z-scores roughly 3.6x relative to later strata. Initialize from the first
   batch (Welford), or apply EMA bias correction as Adam does.
2. **The causal ensemble cannot learn within-state contrasts.** Its loss uses
   `causal_samples_now[row, :, chosen_actions, :]` — one cell per state — and all
   three members receive the identical target. It sees no two cells from the same
   state, and its disagreement is mostly initialization spread on cells it has
   never seen, not epistemic uncertainty. Bootstrapped member datasets
   (Osband et al. 2016) would make the uncertainty meaningful.
3. **Short counterfactual horizon on the decision that matters.** With L = 10 and
   an uninformative critic, `A_CF` is effectively a 10-minute effect. For the
   first decision that captures about a third of the value spread and ranks
   cells at rho ~ 0.5 against full-horizon value (section 4).
4. **Lost tokens.** An infeasible epoch forfeits its token until the next
   10-minute epoch. That adds uncontrollable return variance and breaks
   capacity parity between policies. Retrying each minute until the next epoch
   would remove most of it.
5. **The convergence gate measures the wrong object.** It is computed on
   training returns of the stochastic behavior policy under changing scenarios
   (`training_convergence_diagnostics.json`), and `learning_assessment.json`
   records `evaluation_available: false`.

## 4. How much can any policy gain? (new measurement)

`nmcc_headroom_experiment.py` runs the real stochastic dynamics — hazard spread,
keyed casualty and panic shocks, social force, congestion, shared site rule — on
a synthetic city calibrated to State College's outcome profile, not its map:

| | testbed (heuristic) | State College v22 |
|---|---|---|
| safe at 60 min | 75% | 73% |
| unfinished | 24% | 27% |
| casualties | 0.2% | 0.3% |
| tokens deployed | 4.5 / 5 | 4.4 / 5 |

(Grid 20x20 at 150 m, 8x8 cells, 800 pedestrians, token 160 to keep State
College's token-to-population ratio, 3 hazards, 20 candidates, decisions at
t = 1, 11, 21, ... as in `RLBridge`.)

The rollout policy (Bertsekas) evaluates every feasible cell by installing it in
a branch and continuing the episode with the heuristic to the horizon, then picks
the best. The perfect-information version shares the episode's future noise and
is an upper bound. The implementable version scores each cell by the mean over
two independent future tapes; the tapes are common across cells within a
decision, so the comparison stays paired, but they carry no knowledge of the
real future.

### 4.1 Headroom

Paired differences against the heuristic on matched scenario seeds:

| policy | scenarios | minus heuristic (paired) | 95% CI | wins |
|---|---:|---:|---|---|
| accessibility deficit | 12 | +0.014 | [−0.008, +0.037] | 3/12 (9 ties) |
| uniform random | 12 | +0.044 | [−0.018, +0.106] | 8/12 |
| v22/v23 behavior policy at zero residual | 12 | +0.036 | [−0.012, +0.085] | 6/12 |
| **greedy route-time saving** | 12 | **+0.125** | [+0.061, +0.188] | 10/12 |
| **rollout, implementable (2 tapes)** | 4 | **+0.214** | [+0.109, +0.319] | 4/4 |
| rollout, perfect information | 6 | +0.257 | [+0.197, +0.317] | 6/6 |

Every difference is paired on matched scenario seeds; the heuristic's own mean
return over the 12 scenarios is 0.167. On the four scenarios of the implementable
run, people still unfinished at 60 minutes fall from 193 under the heuristic to
122 under the rollout (−37%); over all 12, greedy route saving cuts them from 173
to 129 (−25%).

On the six seeds where both were run, greedy route saving captures 58% of the
perfect-rollout gain. The remaining route-saving-to-rollout gap is +0.109
[+0.046, +0.172] (5/6) with perfect information and +0.088 [−0.003, +0.179]
(3/4) implementable. **That remainder, not the gap to the registered heuristic,
is what learning genuinely has to find.**

### 4.2 Where the value is

| decision | feasible cells | heuristic is best | heuristic regret | full-value spread across cells | rho(active pop, value) | rho(10-min effect, value) |
|---|---:|---:|---:|---:|---:|---:|
| t = 1 | 18 | 0 of 6 / 0 of 4 | 0.199–0.214 | 0.35–0.45 | −0.13 / −0.18 | 0.53 / 0.51 |
| t = 11 | ~11 | 67% / 25% | 0.016–0.027 | 0.23–0.31 | +0.37 / +0.35 | 0.66 / 0.57 |
| t = 21 | ~5 | 83% / 25% | 0.017–0.030 | 0.06–0.08 | +0.64 / +0.72 | 0.55 / 0.91 |
| t >= 31 | 2 | 100% | 0 | 0 | — | — |

(pairs are perfect-information over 6 scenarios / implementable over 4; the 4
are a subset of the 6, so the t = 1 result is six distinct scenarios, not ten)

Nearly all of the problem's value is the first deployment. There, 18 cells are
open, the choice spans about twice the across-scenario return SD, the heuristic
did not pick the best cell in any scenario tested, and active population carries no information or
slightly the wrong information. It is also the decision with the least temporal
context: at t = 1 the LSTM has seen one frame, so the recurrent machinery cannot
help it. What should drive it is geometry — where people are relative to the
existing shelters and the fire — which is exactly what `candidate_route_time_saving`
encodes, and why that single feature does so well.

### 4.3 What this says about the NMCC target

- 62% of the variance of the Variant-A target `R_L(s,c) - R_L(s, WAIT)` is
  between states (between-state 0.00150, within-state 0.00094). The WAIT branch
  cancels the noise, as designed, but leaves "deploying now is valuable", which
  the actor cannot act on.
- The 10-minute effect's within-state spread (0.031) is about a third of the
  full-horizon spread (0.086), and it ranks the first decision's cells at
  rho ~ 0.5. The target is well aimed for late decisions, which barely matter,
  and weakly aimed for the first, which carries the value.

### 4.4 Caveat

This is a uniform-density grid. State College's population is concentrated, so
the active-population heuristic may do better there, and the size of the gap
will differ. The mechanisms — first-decision dominance, geometry over headcount,
the between-state share of the NMCC target — are what should transfer, and all
three can be checked on State College with the same experiment once it is run
against a real `Core` (not yet wired; see section 6).

## 5. Recommendations, in order

Each item states what to change, why, and how to tell whether it worked.

### R1. Measure the right object before changing anything else

Build learning curves from frozen checkpoints evaluated deterministically on a
fixed set of held-out CRN tapes, paired with the heuristic, greedy route saving
and the MC rollout on the same tapes. Report the stochastic behavior policy's
return separately. Without this, no change below can be judged.
*Check:* the RL-minus-heuristic curve should start at ~0 (argmax ~ heuristic),
which confirms the diagnosis in 1.1.

### R2. Train the actor by distilling the rollout policy (approximate policy iteration)

At each labeled decision, the rollout supplies a full-horizon, CRN-paired value
for **every** feasible cell. Train the actor with a listwise loss — cross-entropy
against `softmax(z(Q_rollout) / tau)` over feasible cells — on states the student
itself visits (DAgger; Ross et al. 2011). Then roll out over the distilled
policy and repeat (expert iteration; Anthony et al. 2017). Each label is a
ranking over ~18 cells rather than one noisy scalar for one sampled cell, so a
few hundred labeled decisions carry more information than the entire current
training budget.

Cost at State College: an episode takes ~34 s (~0.3 s per step); a rollout label
at t = 1 costs about 18 cells x 1 tape x ~60 steps, roughly 5–6 minutes on one
core, and later decisions far less. Label the first two decisions exhaustively,
subsample cells after that, parallelize across episodes. With CRN across cells a
single tape already gives a paired comparison. The implementable 2-tape rollout
came within about 0.01 of perfect information on two of four scenarios and 0.08
and 0.15 below it on the other two, so more tapes help where the future is most
uncertain.

This also gives the paper a defensible contribution: the rollout decides in
minutes, the distilled GNN in 0.6 ms (`mean_policy_selection_latency_ms`), and
the claim becomes "near-rollout quality at amortized cost".
*Check:* held-out return of the distilled policy against route saving and the
rollout on common tapes; top-1 agreement with the rollout at t = 1.

### R3. If you keep policy gradients, change the estimator, not only its weight

(a) **Baseline for a forced choice.** Replace the WAIT baseline with the
policy-weighted predicted effect:
`A(s,a) = [G(a,U) - G(WAIT,U)] - sum_c pi(c|s) D_hat(s,c)`.
It is action-independent, so unbiased, and it removes the 62% between-state
component.

(b) **Better: a doubly robust all-action gradient.**
`g = sum_c grad pi(c|s) D_hat(s,c) + grad log pi(a|s) [Delta_exact(a) - D_hat(s,a)]`.
It is unbiased for any `D_hat`, because the sampled term corrects the model at the
chosen cell, and it uses the model's prediction for every feasible cell at every
decision instead of one sample. This is the action-dependent control variate of
Q-Prop (Gu et al. 2017) and Liu et al. (2018). Tucker et al. (2018) found such
baselines add little when trajectory variance dominates; CRN pairing has already
removed that, which leaves exactly the regime where they help.

(c) **Longer horizon for early decisions.** Branch the t = 1 and t = 11
decisions to the horizon with heuristic (or current-policy) continuation instead
of 10 minutes, or lengthen `L` once the critic explains variance.
*Check:* within-update SD of the actor advantage, and the fraction of it that is
between-state, logged per update.

### R4. Make the causal model identifiable within a state

At a fraction of decisions — all of them at t = 1 — branch k additional
uniformly sampled cells, not only the chosen one. Track NMCC's own Stage-2
gates, which no run has measured yet: within-state Spearman between `D_hat` and
exact effects on held-out states, and top-1 regret. Give each ensemble member a
bootstrap resample of the data.
*Check:* within-state rank correlation on held-out branch sets rising above the
10-minute target's own ~0.5.

### R5. Give the encoder to the world model, not the actor

Let the natural, causal and TD losses train the GNN and LSTM, and put the actor
head on detached features (or a separate small encoder). That keeps v23's
attribution guarantee — actor gradients cannot move the world model — while
restoring representation learning from dense supervision, which is how
auxiliary-task agents are built (UNREAL, Jaderberg et al. 2017; world models,
Ha and Schmidhuber 2018; Dreamer, Hafner et al.).
*Check:* critic explained variance moving off zero within a few updates.

### R6. Replace or re-express the prior

Max active population is the wrong inductive bias for the decision that matters.
Start from greedy route saving as the prior (it is already in
`CANDIDATE_FEATURE_NAMES`), from the causal model's scores, or from the distilled
policy (R2). Whatever the prior, give it enough logit range to shape sampling if
it is meant to act as a base policy; a 1.0-logit spread does neither job.
Add greedy route saving and the MC rollout to `BENCHMARK_MODEL_PROTOCOL.md`;
reviewers will ask why a stronger, equally transparent greedy rule was not the
comparator.

### R7. Plan the budget from the arithmetic

With policy gradients under a 0.015 KL target, concentrating even one decision
takes 16+ consistent accepted updates. Increase decisions per update (more
episodes per rollout, parallel environments) and budget for hundreds of
episodes. Under R2 the budget is set by teacher labels instead.

### R8. Small fixes

Welford or bias-corrected initialization for the lagged baseline; no unbaselined
first actor rollout; token retry each minute after an infeasible epoch; bootstrap
resampling for the causal ensemble.

## 6. What was not done

- No learner run. torch remains uninstallable from this environment.
- The headroom experiment runs on the synthetic testbed only. Running it on
  State College requires constructing a real `Core` with a heuristic deployment
  strategy and passing it to `headroom_lib.run`; the episode runner uses only
  `CounterfactualBranch`, `RegionalObservationBuilder` and
  `RegionalShelterExecutor`, which are the production objects, but that path
  has not been built or tested and no `--real-core` option exists.
- Rollout results rest on 6 seeds (perfect information) and 4 (implementable);
  greedy and random baselines on 12.

## 7. Artifacts

- `runs/headroom_perfect_information_20260921.json` — 12 seeds, 6 with rollout
- `runs/headroom_mc2_implementable_20260921.json` — 4 seeds, 2 future tapes
- `runs/headroom_greedy_20260921.json` — greedy route saving and reroutable
- `nmcc_headroom_experiment.py`, `headroom_lib.py`
- `nmcc_testbed.py`: new `spacing_m` and `candidate_count` options, defaults
  unchanged; `tests/test_counterfactual_branch.py` still passes 10/10.
