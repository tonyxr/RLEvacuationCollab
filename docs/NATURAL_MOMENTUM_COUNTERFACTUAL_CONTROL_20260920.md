# Natural-Momentum Counterfactual Control for Crowd-Interactive Systems

## Material Passport

- Origin Skill: `academic-research-suite/experiment-agent`
- Origin Mode: `plan`
- Origin Date: `2026-09-20`
- Verification Status: `UNVERIFIED` (proposed framework; no implementation yet)
- Version Label: `nmcc_framework_v1`

## Experiment Overview

- **Title:** Natural-Momentum Counterfactual Control (NMCC)
- **Objective:** Learn the causal contribution of shelter interventions while
  separating it from hazard evolution, crowd inertia, congestion, and outcome
  randomness.
- **Hypothesis:** A policy trained on paired action-versus-`WAIT` effects under
  common structural noise will have substantially higher gradient signal-to-
  noise and better held-out return than recurrent PPO trained on realized total
  return alone.
- **Type:** staged model-based reinforcement learning and simulation
  optimization.

## Central claim

The user's proposed order—learn natural system evolution first, then learn the
controller—is correct but incomplete. A world model can predict total system
evolution accurately while still learning the wrong action effect, because
natural evolution dominates the loss and actions are selected precisely when
conditions are bad. The controller can consequently learn that interventions
"cause" casualties merely because interventions are correlated with dangerous
states.

The fundamental solution is to combine staged dynamics learning with explicit
interventional data. At a decision state, compare an action and `WAIT` from the
same simulator snapshot under the same hazard, casualty, panic, and behavior
noise. The difference is the action's causal contribution; the shared natural
trajectory cancels.

No neural architecture can identify that contribution from one unpaired
trajectory without additional assumptions. Identification requires at least
one of randomized interventions, repeated matched scenarios, a structural
causal model, or a simulator that can generate counterfactual branches. This
project has a simulator and is therefore unusually well positioned to solve the
problem directly.

## Why standard recurrent PPO remains noisy

Let `G(a, U)` denote the future global return after action `a`, where `U`
contains all uncontrolled system disturbances. Standard PPO observes one value,
`G(a_t, U_t)`. It must infer whether a low return came from the selected shelter,
hazard growth, crowd congestion, panic, or casualty randomness.

The current LSTM helps remember temporal context, but memory does not create the
missing counterfactual. Longer GAE propagates an outcome farther backward, but
it still propagates the whole realized outcome. It improves temporal credit
without separating cause from coincidence.

This distinction is consistent with work on exogenous-state RL, which shows
that uncontrolled state and reward variation can slow learning and that an
endogenous/exogenous decomposition can reduce variance
([Dietterich, Trimponias, and Chen](https://arxiv.org/abs/1806.01584)). It also
matches counterfactual credit-assignment work whose explicit goal is to
separate skill from luck
([Mesnard et al.](https://proceedings.mlr.press/v139/mesnard21a.html)).

## Structural model

NMCC treats the evacuation simulator as a structural partially observed MDP
with four state blocks:

- `P`: static physical structure—road graph, candidate sites, installed
  shelters, capacities, and deployment resources;
- `H`: action-independent hazard state—sources, wind, spread, severity, and
  forecast uncertainty;
- `C`: crowd state—regional population, route flows, congestion, wellness,
  shelter assignments, and unfinished demand;
- `M`: controller memory—previous observations, past interventions, remaining
  capacity tokens, and time.

For one decision interval, use the structural factorization

\[
H_{t+1}=F_H(H_t,U^H_t),
\]

\[
C^0_{t+1}=F_0(C_t,P_t,H_t,H_{t+1},U^C_t),
\]

\[
(P^a_{t+1},C^a_{t+1})=(P^0_{t+1},C^0_{t+1})+
\Delta F(P_t,C_t,H_{t:t+1},a_t,U^C_t).
\]

`F_H` is the hazard model, `F_0` is natural momentum under no new controller
intervention, and `Delta F` is the intervention residual. The additive notation
describes the learning target; the residual network may be nonlinear and
conditioned on the complete baseline forecast, so crowd-hazard interactions are
not assumed additive in the physical system.

### Natural momentum

Natural momentum means the evolution that would occur over the next interval
if no new shelter decision were made, conditional on everything already in the
system. It includes:

- motion along existing routes;
- congestion propagation;
- arrivals at previously installed shelters;
- hazard spread and exposure;
- panic and wellness evolution;
- capacity filling at existing shelters.

It does not erase the effects of earlier decisions. It asks only: "given the
current state, what happens next without another intervention?"

### Intervention residual

For horizon `h` and outcome vector `Y`, define

\[
\Delta Y_{t,h}(a,U)=Y_{t+h}(a,U)-Y_{t+h}(WAIT,U).
\]

`Y` should remain vector-valued:

- new safe completions;
- casualties or expected casualty risk;
- active pedestrian-time;
- hazard-exposure pedestrian-time;
- route-time burden;
- capacity deficit;
- congestion burden;
- unfinished population.

Predicting the vector before applying objective weights makes the model
auditable and reusable. It also prevents a large scalar reward branch from
hiding a poor casualty or exposure prediction.

## Structural noise tape

Paired branches are valid only if they share the same exogenous disturbances.
Copying the current global random-number generator is insufficient: an action
may change which agents execute code, changing the number and ordering of
random draws.

Before an episode, generate or deterministically address a structural noise
tape:

\[
U=(U^H_{cell,time},U^{cas}_{person,time},U^{panic}_{person,event},
U^{move}_{person,time},U^{policy}_{decision}).
\]

Every stochastic mechanism retrieves its value by immutable keys such as
episode seed, mechanism, entity identifier, and simulation time. It never
consumes a shared sequential stream. The factual and counterfactual branches
therefore experience the same potential hazard and person-level shocks whenever
the corresponding structural event exists.

The current simulator already isolates hazard streams and uses keyed
person-time shocks for important pedestrian outcomes, so this is an extension
of an existing design rather than a replacement. Snapshot/restore must include
all mutable simulator state, route assignments, accumulated hazard exposure,
shelter flows, controller memory, and noise-tape position or key namespace.

Common random numbers are a classical simulation-optimization variance-
reduction method for comparing alternatives; paired noise increases correlation
between alternatives so their difference is more precise
([Kleinman, Spall, and Naiman](https://pubsonline.informs.org/doi/10.1287/mnsc.45.11.1570)).

## NMCC architecture

### 1. Factored graph belief encoder

Retain the resolution-flexible regional GNN and recurrent belief state, but
give the branches explicit roles:

- hazard encoder: consumes `H` and cannot consume the current action;
- natural crowd encoder: consumes crowd, routes, current infrastructure, and
  predicted hazard;
- candidate intervention encoder: attends from each candidate to the regions
  and routes it can affect;
- recurrent belief: summarizes partial observation and forecast error over
  time.

The graph remains aggregated by flexible regions. Individual pedestrian paths
contribute route-flow and provisional assignment features; 3,000 individual
pedestrian nodes are unnecessary.

### 2. Natural-momentum world model

The natural model `M0` predicts multi-step distributions under `WAIT`:

\[
p_0(H_{t+1:t+L},C^0_{t+1:t+L}\mid b_t).
\]

Use multiple forecast horizons, for example the next decision interval, three
decision intervals, and terminal horizon. Predict distributions rather than
means. Hazard spread, casualty risk, and congestion are stochastic and their
uncertainty is decision-relevant.

The model must satisfy hard consistency checks:

- population mass is conserved across active, safe, casualty, and unfinished
  categories;
- shelter flow never exceeds installed capacity;
- hazard predictions remain bounded;
- `WAIT` cannot change infrastructure;
- graph predictions are equivariant to region and candidate permutations.

World-model approaches show that policies can be trained from imagined future
trajectories, but NMCC uses the model primarily to expose the no-action baseline
and action residual rather than to replace the simulator wholesale
([Hafner et al.](https://arxiv.org/abs/2301.04104)).

### 3. Intervention-residual ensemble

The intervention model predicts a distribution of multi-horizon differences
for every feasible candidate:

\[
p_\Delta(\Delta Y_{t,1:L}(a)\mid b_t,a,M_0(b_t)).
\]

Train an ensemble so disagreement estimates epistemic uncertainty. Enforce the
identity

\[
\Delta Y(WAIT)=0
\]

exactly by construction, not by a soft penalty. Candidate residuals should be
localized through candidate-to-region and candidate-to-route attention. This
lets the network model the sparse part of the system that the action can
actually change.

The natural model and residual model must not be trained as one unconstrained
next-state predictor. Otherwise the much larger natural trajectory can absorb
the intervention signal. Pretrain `M0`, freeze it while the first residual model
is learned, then allow only slow natural-model updates from a dedicated passive
buffer.

### 4. Dueling causal critic

Factor the action value as

\[
Q(b_t,a)=V_0(b_t)+D(b_t,a),
\]

where `V0` predicts the global return under `WAIT` and `D` predicts the causal
uplift of action `a` over `WAIT`. Set `D(b_t, WAIT)=0`.

This is more than ordinary dueling-network parameterization: each head has a
separately observable simulator target from paired branches. The critic should
retain component heads for safe completion, casualty, time, exposure, and
intervention cost, with calibrated per-head normalization.

The action ordering is unchanged by subtracting the same no-action baseline:

\[
\arg\max_a Q(b_t,a)=\arg\max_a [Q(b_t,a)-Q(b_t,WAIT)].
\]

Thus exact counterfactual differences improve learnability without changing the
optimal action for the original global objective.

### 5. Counterfactual-advantage PPO

For the chosen action, fork the simulator into factual and `WAIT` branches
under the same structural noise and the same continuation policy. For branch
horizon `L`, train on

\[
A^{CF}_t=
\sum_{k=0}^{L-1}\gamma^k(r^a_{t+k}-r^0_{t+k})
+\gamma^L[V(s^a_{t+L})-V(s^0_{t+L})]-c(a).
\]

Use `A_CF` in the clipped PPO surrogate. The `WAIT` return is a control variate
independent of the sampled action, so an exact paired estimator preserves the
policy-gradient direction while removing the shared natural trajectory.

Counterfactual credit-assignment algorithms similarly seek low-variance action
contributions rather than assigning every later event to every earlier action
([COCOA](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d8bd445c2abe1343cce0e14b361b2fb3-Abstract-Conference.html)).

If exact branching is too expensive for every update, use the learned residual
as a control variate and periodically anchor it with exact paired branches. Do
not silently replace exact causal effects with unvalidated model predictions.
Model error can otherwise produce a confident but biased policy.

### 6. Robust receding-horizon optimization teacher

At each decision, the world model supplies a distribution of causal effects for
each candidate. A constrained optimizer scores candidate sequences by

\[
E[\Delta G]-\kappa\,Uncertainty-\lambda_{cost}Cost,
\]

subject to safety masks, equal-capacity tokens, the installation budget, and
casualty/exposure constraints. Because only a small number of shelter additions
is allowed, a masked beam search or scenario-tree model-predictive controller is
more transparent than a second large RL agent.

The optimizer has three purposes:

1. create high-quality teacher actions before PPO has learned;
2. provide a benchmark between the heuristic and learned policy;
3. query exact counterfactual branches for actions that are uncertain or nearly
   tied.

Distill its masked action distribution into the recurrent GNN actor early in
training. Fade distillation later so the actor can improve beyond the finite-
horizon planner. During deployment, use the fast actor; invoke the planner or a
safe heuristic only when residual uncertainty exceeds a prespecified threshold.

## Staged learning protocol

### Stage 0 — Make counterfactuals valid

Before neural training:

- complete the keyed structural noise tape;
- implement exact simulator snapshot/restore;
- verify branch-order invariance;
- implement equal shelter-capacity tokens;
- add `WAIT` and complete interval accounting;
- prove factual replay from a restored snapshot is bitwise identical.

This stage is a scientific prerequisite. Training on invalid twins would create
high-confidence false causal labels.

### Stage 1 — Passive natural-dynamics pretraining

Generate episodes with no new intervention after the starting configuration.
Randomize initial shelter configurations, population, hazard source, wind,
capacity regime, and grid resolution so `M0` covers states that later policies
will visit. A dataset containing only the original initial shelters would not
cover post-deployment states.

Train `M0` on:

- multi-step hazard prediction;
- regional population and route-flow prediction;
- congestion and wellness prediction;
- continuous exposure-dose and expected casualty-risk prediction;
- safe-completion and capacity-flow prediction.

Do not train on reward initially. Learn physical outcomes first. Rare realized
casualties remain an evaluation target, while exposure dose and expected
casualty risk provide dense predictive supervision without artificially raising
the casualty rate.

Promotion criteria should include multi-horizon calibration, population
conservation, forecast coverage, and performance across held-out hazard/wind
regimes—not only one-step mean squared error.

### Stage 2 — Randomized paired intervention learning

At sampled decision snapshots:

1. execute `WAIT` under noise tape `U`;
2. restore the snapshot;
3. execute a safe candidate under the same `U`;
4. measure the vector difference at several horizons;
5. repeat for multiple candidates and a small number of independent tapes.

Query all candidates only in small toy graphs. At full scale, combine:

- uniformly sampled feasible candidates to maintain support;
- high-uncertainty candidates for active learning;
- near-tied candidates important for policy ranking;
- a limited number of heuristic-high-score candidates.

Never collect only the heuristic's preferred actions; that would preserve the
original selection bias and leave alternatives unidentified.

Freeze `M0` initially and train the residual ensemble, causal critic, and
candidate effect heads. Evaluate sign accuracy, ranking accuracy, uncertainty
coverage, and top-action regret on untouched paired branches.

### Stage 3 — Optimization teacher and actor warm start

Use robust receding-horizon optimization over the learned causal effect model.
Train the actor by masked policy distillation and train the recurrent encoder on
auxiliary natural-forecast and residual-effect tasks. Broad masked exploration
remains active so the actor does not collapse to the teacher.

No policy-performance claim should be made at this stage; the actor is learning
operational mechanics and the model's current approximation.

### Stage 4 — Counterfactual PPO

Run on-policy episodes. For selected decisions, collect exact chosen-action
versus `WAIT` branches and compute `A_CF`. For the remaining decisions, use the
residual ensemble only when uncertainty is below its calibrated threshold;
otherwise query an exact branch or exclude the approximate causal target.

Training loss contains:

- clipped PPO loss using causal advantage;
- global value loss;
- natural-baseline value loss;
- component residual losses;
- fading planner-distillation loss;
- fading dense operational proxy loss;
- entropy-target loss and intervention cost.

The exact causal advantage may remain throughout training because it represents
the same global objective with a control variate. What must fade are proxy
rewards and imitation losses that are not themselves the global objective.

### Stage 5 — Objective-only confirmation

Freeze all curricula. Evaluate the actor without counterfactual branches,
exploration, or proxy reward. Compare RL, the active-population heuristic, the
robust model-predictive controller, and static placement under:

- common scenario seeds and hazard paths;
- identical initial shelters;
- identical capacity-token schedules;
- identical masks and physical executor;
- untouched held-out scenarios.

Only the original global outcomes—safe completion, casualties, evacuation time,
exposure, and declared intervention cost—determine the final comparison.

## Why this is stronger than reward shaping

Dense shaping asks the designer to invent a proxy correlated with long-term
success. NMCC instead estimates the counterfactual change in the actual outcome
vector. It distinguishes:

- **forecast:** what is likely to happen anyway;
- **effect:** what this action changes;
- **uncertainty:** what the model does not know;
- **preference:** how the administrator weights safety, time, exposure, and
  resource cost.

Preferences can change without relearning physical dynamics. A casualty weight
change alters optimization over predicted effects; it does not require the
world model to rediscover how people and hazards move.

## Identifiability and failure conditions

NMCC is not automatically causal. Its claims require:

1. **Structural-noise consistency:** paired branches use the same exogenous
   potential disturbances.
2. **Action support:** every action that may be selected has intervention data
   in comparable states, or calibrated uncertainty forces conservative use.
3. **State sufficiency:** the recurrent belief contains the variables that
   jointly affect action selection and outcomes.
4. **No action leakage into the hazard model:** shelters do not affect hazard
   physics; the model and data pipeline must preserve that restriction.
5. **Correct continuation:** paired branches use the same continuation-policy
   rule after their first action.
6. **Model pessimism:** the optimizer penalizes epistemic uncertainty and cannot
   exploit unvalidated extrapolation.
7. **No outcome-conditioned sampling:** branch queries may depend on the
   pre-action state and model uncertainty, not on which branch later happened
   to produce casualties.

If these assumptions fail, a simpler exact paired simulator optimizer may be
more trustworthy than a learned residual policy.

## Recommended implementation variants

### Variant A — Exact paired PPO

- For each sampled action, run one matched `WAIT` branch.
- Use exact `A_CF`; no learned residual model is required.
- Highest causal fidelity, highest simulation cost.
- Best first proof that variance separation helps.

### Variant B — Residual world model plus optimizer

- Learn `M0` and `MDelta` from an offline paired dataset.
- Choose actions with robust receding-horizon optimization.
- No PPO required initially.
- Most interpretable operational benchmark.

### Variant C — Hybrid NMCC (recommended)

- Pretrain `M0` and `MDelta`;
- warm-start from the robust optimizer;
- fine-tune with recurrent paired PPO;
- actively request exact branches where uncertainty is high.

Variant C combines simulator truth, planning, and amortized policy speed. It
should be attempted only after Variant A demonstrates a material reduction in
advantage variance and Variant B demonstrates reliable effect ranking.

## Ablation ladder

| ID | System | Purpose |
|---|---|---|
| N0 | Current recurrent PPO | Baseline. |
| N1 | Standard PPO plus natural-forecast features | Tests whether prediction alone helps. |
| N2 | Exact chosen-action versus `WAIT` paired PPO | Tests causal variance cancellation without model complexity. |
| N3 | Natural model plus residual causal critic | Tests learned effect generalization. |
| N4 | Robust model-predictive controller | Tests optimization without actor learning. |
| N5 | Planner-distilled recurrent policy | Tests amortization. |
| N6 | Full hybrid NMCC | Tests the complete framework. |
| N7 | N6 without common keyed noise | Negative control; should show higher effect variance. |
| N8 | N6 with a monolithic world model | Tests whether explicit residual factorization matters. |

## Primary metrics and promotion gates

### Dynamics layer

- multi-horizon hazard and crowd negative log likelihood;
- calibration/coverage of predictive intervals;
- population and capacity conservation error;
- held-out no-action return error;
- forecast robustness across grid dimensions.

### Causal-effect layer

- mean absolute error of each paired outcome difference;
- sign accuracy for beneficial versus harmful interventions;
- candidate rank correlation;
- top-one and top-three decision regret under exact branches;
- 90% interval coverage of residual outcomes;
- variance ratio
  `Var[G(a,U)-G(WAIT,U)] / Var[G(a,U)]`;
- causal-advantage gradient signal-to-noise across policy seeds.

### Policy layer

- held-out global return relative to recurrent PPO and heuristic;
- casualty mean and tail risk;
- exposure and evacuation-time differences;
- inappropriate intervention rate when exact marginal benefit is negative;
- `WAIT` calibration;
- capacity parity and capacity-time reporting;
- policy stability under held-out hazard/wind shifts.

Initial promotion gates:

- N2 must reduce advantage variance by at least 50% without changing the mean
  paired effect beyond Monte Carlo error;
- residual sign accuracy must exceed 80% and exact top-three action coverage
  must exceed 90% on held-out branches before model-only actions are trusted;
- predicted residual intervals must achieve at least 85% empirical coverage for
  nominal 90% intervals before uncertainty gating is enabled;
- N6 must outperform N0 and the heuristic on paired held-out return without a
  worse casualty tail;
- every claim must survive at least five policy seeds and untouched scenario
  seeds in the confirmatory stage.

These are engineering promotion thresholds, not claimed theoretical constants.

## Minimal first experiment

The first experiment should test the causal premise without building the full
world model.

1. Select 200 diverse decision snapshots from State College training scenarios.
2. At each snapshot, sample one feasible candidate and run candidate versus
   `WAIT` under four common noise tapes.
3. Repeat the same comparisons with independent noise tapes as a control.
4. Compare variance of return differences, casualty-risk differences, exposure
   differences, and route-time differences.
5. Train a small candidate residual head on 70% of snapshots; validate sign and
   ranking on 15%; reserve 15% untouched.

Proceed to the full staged world model only if common-noise paired effects are
materially more stable and predictably ranked. This prevents a large
architecture investment before the core identification mechanism is verified.

## Expected outputs

| Output | Format | Success criterion |
|---|---|---|
| Structural-noise parity audit | JSON | factual snapshot replay is exact and branch-order invariant |
| Paired-effect dataset | Parquet/NPZ plus manifest | action and `WAIT` share state/noise digests |
| Natural-model calibration report | JSON/CSV/figures | all conservation and coverage gates pass |
| Residual-effect report | JSON/CSV/figures | sign, ranking, uncertainty, and variance gates pass |
| NMCC checkpoint | PyTorch plus contract | exact resume and model/interface hashes pass |
| Paired policy backtest | CSV/JSON/figures | return improves with no worse casualty tail |

## Research contribution

The defensible contribution is not merely "GNN plus LSTM plus PPO." It is a
general control architecture for stochastic crowd-interacting systems that
learns three different objects:

1. what the uncontrolled system will do;
2. what an intervention changes relative to that natural momentum;
3. which intervention is preferred under safety and resource constraints.

That decomposition is interpretable to administrators, compatible with graph-
structured observations, and testable through exact simulator counterfactuals.
It directly targets the failure observed in the current experiments: the policy
gradient sees too much system variance and too little decision-maker variance.
