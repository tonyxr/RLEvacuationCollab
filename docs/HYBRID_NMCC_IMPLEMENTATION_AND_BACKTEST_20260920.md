# Hybrid NMCC implementation and engineering backtest

## Decision

Model version 21 implements the complete production Hybrid Natural-Momentum
Counterfactual Control (NMCC) learning path. It combines exact, short-horizon
common-noise branches with learned natural evolution and action-residual
models, recurrent PPO, factorized value estimation, robust feasible-action
scoring, scheduled exploration, and complete post-action accounting.

The implementation passes its functional and signal-quality gates. The matched
engineering backtest shows a materially less variable causal actor target and
a slightly steeper training slope, but its eight-scenario held-out mean is
below ordinary recurrent PPO. This is deliberately labeled engineering
evidence: 16 small synthetic-map training episodes do not establish policy
improvement or State College convergence.

## Material passport

- Date: 2026-09-20 (America/Los_Angeles)
- Git revision at implementation: `2fa2b513f181783ec3daa7c905b30635db461a2c`
- Branch: `main`
- Worktree: dirty; the implementation and artifacts are uncommitted
- Origin skill: `academic-research-suite/experiment-agent`
- Runtime: project `evacuationModel` Conda environment
- Runtime limitation: functional Torch execution currently requires
  `KMP_DUPLICATE_LIB_OK=TRUE`; results from that workaround are not accepted as
  publication-grade numerical evidence
- Learning artifact: `runs/nmcc_learning_backtest_20260920.json`
- Artifact SHA-256:
  `ae7caeca90bb5fa5b60a312e9d86dc742a71e3879d0dfe8763b369aa9321c80e`
- Evidence status: `ENGINEERING_BACKTEST`

## Implemented NMCC sequence

At each shelter-deployment epoch, the learner executes this sequence:

1. The GNN encodes flexible regional cells using physical-network,
   infrastructure, hazard, pedestrian, route-delay, wellness, shelter-capacity,
   and candidate-feasibility features. Spatial and current route relations are
   represented as graph edges.
2. The LSTM consumes the full sequence of pre-decision graph embeddings, so
   the policy can infer whether evacuation, exposure, and danger are improving
   or deteriorating instead of treating each observation as independent.
3. A simulator snapshot is captured before the selected shelter is installed.
   From that identical snapshot and random-generator state, NMCC runs the
   chosen action and `WAIT` for the same fixed horizon `L`.
4. Both branches return the six-coordinate physical target: safe completions,
   casualties, active person-time, exposure person-time, final active
   population, and final risk mass.
5. The exact component-level target is
   `A_CF = (R_action - R_wait) + gamma^L (V_action - V_wait) - action_cost`.
   The factual transition closes at exactly `L`; the ordinary SMDP reward for
   that action continues through the next decision or true terminal. This
   prevents both horizon leakage and terminal reward censoring.
6. An action-independent natural head learns the `WAIT` outcome. Three
   candidate-local residual heads learn the action-minus-`WAIT` outcome. Their
   mean is the expected causal effect and their dispersion measures epistemic
   uncertainty.
7. The actor is trained with recurrent PPO, using exact counterfactual
   advantages where a valid paired branch exists and GAE elsewhere. A
   factorized critic separately predicts safe-completion, casualty, evacuation-
   time, and exposure components before summing them to the scalar value.
8. Auxiliary Smooth-L1 losses train the natural outcome head, selected-action
   causal residual ensemble, and dueling consistency relation. Exact branch
   outcomes are the supervision; the policy reward is not reused as a proxy
   world-model label.
9. A robust planner ranks every forecast-safe cell by ensemble-mean causal
   reward minus an uncertainty penalty. Its detached scores provide a delayed,
   fading teacher and bounded guidance term. Detachment is essential: PPO or
   imitation gradients cannot alter the causal world model to make the
   actor's current choice appear better.
10. Entropy and sampling temperature start high and decay by optimizer update.
    This supplies broad early exploration while making late behavior more
    stable. Teacher guidance begins only after a world-model warmup, ramps in,
    and then fades so the final controller is optimized for the global return.

## Physical constraints in the learned model

The natural model is structurally constrained rather than merely penalized:

- predicted safe completions, casualties, and final active population are
  nonnegative and sum exactly to the current active population;
- final risk mass lies between one-half of and all of the predicted final
  active population;
- active and exposure person-time are bounded by the population-horizon scale;
- the natural prediction has no action input, while action residuals are local
  to the candidate cell and share the recurrent global context;
- infeasible or forecast-unsafe cells are masked before policy sampling and
  robust optimization.

These constraints eliminate impossible targets that otherwise destabilize a
small-data auxiliary world model and make the learned quantities directly
auditable against simulator outcomes.

## Configuration contract

All NMCC and exploration fields are validated in `Core`, passed explicitly to
`RLBridge`, included in effective-configuration output, and embedded in the
checkpoint signature. A training resume fails closed if that contract changes.
The learner settings must also be identical across all stages and variants of
a curriculum, preventing an apparent continuation from silently changing the
model or optimizer.

The registered full State College contract is
`config/state_college_training_curriculum_3000_nmcc_hybrid.json`. It fixes:

- 3,000 individually represented pedestrians;
- three stochastic hazards and casualty distribution `[40, 9]`;
- 50% panic susceptibility;
- five additional shelters, each adding exactly 500 places;
- a 10-timestep paired counterfactual horizon and full causal-advantage weight;
- a three-member residual ensemble;
- natural, causal, and dueling auxiliary loss weights `0.5`, `1.0`, and `0.25`;
- teacher coefficient `0.2`, 64-update decay, eight-update guidance warmup,
  16-update ramp, and maximum guidance weight `0.5`;
- uncertainty penalty `0.5`;
- entropy decay from `0.005` to `0.001` and action temperature decay from
  `1.35` to `1.0` over 128 updates.

## Capacity-controlled optimization

`shelterCapacityToken` separates where a shelter can physically be installed
from how much capacity an installation contributes. A site's raw capacity is
still used for eligibility and deterministic within-cell site resolution;
sites rated below one token are removed before candidate sampling and remain
masked/rejected in every preview and execution path. Once chosen, every
dynamic policy receives the same capacity token. Therefore five
installations under the State College contract add exactly 2,500 places for RL
and every heuristic, even if they choose different buildings.

The matched learning backtest fails before comparison unless every policy has
identical added-shelter count and added capacity. All eight held-out scenarios
passed with exactly two additions and 600 added places for recurrent PPO,
Hybrid NMCC, and the heuristic.

## Verification

The complete repository suite passed: **210/210 tests**. The focused coverage
includes exact snapshot replay, branch-order invariance, stochastic common-
noise pairing, fixed-horizon accounting, recurrent trajectory caching, PPO
optimizer and checkpoint recovery, bounded and action-local NMCC heads,
natural population conservation, detached teacher scores, configuration and
curriculum validation, forecast-safe masks, and equal-capacity execution.

The matched learning backtest used 16 training episodes, four episodes per PPO
update, eight deterministic held-out scenarios, identical initial weights,
matched scenario seeds, and the real project dynamics on a small synthetic
road map.

| Diagnostic | Hybrid NMCC result |
|---|---:|
| Completed optimizer updates | 4 |
| Updates with exact counterfactual targets | 100% |
| Counterfactual advantage SD | 0.090298 |
| Raw GAE advantage SD | 0.213234 |
| Counterfactual/GAE SD ratio | 0.4235 |
| Implied target-variance reduction | 82.1% |
| Training-return slope per episode | +0.020739 |
| Recurrent-PPO training slope | +0.020467 |
| Held-out NMCC minus PPO mean return | -0.014123 |
| Held-out difference SD | 0.021723 |
| Held-out NMCC win fraction | 0.000 |
| Held-out NMCC minus heuristic mean return | -0.026568 |
| Equal installed shelter count/capacity | pass |

Several paired scenarios tie exactly, and the small sample is not an
inferential study. There is no held-out improvement claim. The valid conclusion
is narrower: exact counterfactual targets reach the PPO optimizer, are
substantially less noisy than the raw GAE targets, training is numerically
stable, and the resulting policy is evaluable under strict action-and-capacity
parity.

### State College 3,000-person system smoke

The eight-episode smoke curriculum was executed on the cached 6-km State
College walking network: 6,973 consolidated road nodes, 25,804 edges, 64
adaptive regional cells, 2,375 detected shelter-capable sites, 20 sampled
candidates, 3,000 individually represented pedestrians, and a 60-minute
horizon. The run completed and finalized its checkpoint and manifest at
`runs/state_college_3000_nmcc_v21_physical_smoke_20260920`.

- All eight episodes preserved exact population accounting.
- Casualties were `[34, 4, 0, 42, 2, 0, 29, 0]`: mean 13.875 and range 0–42
  out of 3,000. This is nontrivial reward variation without population-scale
  mortality.
- Mean safe completions were 2,147.125 per episode.
- The completed recurrent batch contained eight whole episodes, 34 decisions,
  268 cached observation frames, and four PPO epochs.
- Exact counterfactual coverage was 100% at the optimizer update.
- Counterfactual advantage SD was 0.031394 versus raw GAE SD 0.132154, a
  0.2376 ratio and approximately 94.4% target-variance reduction.
- Natural, causal, dueling, and teacher losses were finite: 0.058581,
  0.000388, 0.119346, and 2.168975. Ensemble causal uncertainty was 0.001057.
- Gradient norm was 3.7138, approximate KL was 0.000012, and reward-accounting
  gap was zero on the update episode.
- Deployments were `[5, 5, 4, 3, 5, 5, 4, 3]`. Four scenarios lost one or two
  actions to the shared safety/feasibility mask. This training-only
  smoke is therefore not a capacity-parity policy comparison; the confirmatory
  evaluation must enforce the common deployment deadline and parity gate.

The actor was held fixed until the eighth episode and updated once at the end,
so the episode-return sequence is scenario variation, not a learning curve.
The smoke establishes real-map execution, credit-path coverage, finite
optimization, and casualty sensitivity only. The runner correctly reports
`training_not_converged` and does not produce a held-out performance claim.

Smoke artifact hashes:

- manifest:
  `aecfbcd513e3a594286d56c851fde19efc498d2206bbec42296ad8e81abc85d8`
- training summary:
  `068518c7d2be7aae312d8b32a75922e5df4d25dba873e9286924e15c234d76df`
- PPO/NMCC diagnostics:
  `304a2a3833ab6907fa4f5f86eb93dd20c0d4c82f679dc10a4bc62a3fe21f7ec3`

## Remaining evidence boundary

The implementation does not yet prove that Hybrid NMCC is superior on State
College. That requires the registered 64-episode 3,000-person training run,
multiple independent policy seeds, a supported OpenMP runtime, and matched
held-out comparison against recurrent PPO and heuristics. Results must report
return, casualties, safe completions, evacuation person-time, exposure person-
time, uncertainty calibration, action/capacity parity, and seed-level
confidence intervals. A rising training curve alone is insufficient.
