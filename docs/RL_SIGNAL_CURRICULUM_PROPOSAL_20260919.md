# RL Signal Curriculum and Intervention-Control Proposal

## Material Passport

- Origin Skill: `academic-research-suite/experiment-agent`
- Origin Mode: `plan`
- Origin Date: `2026-09-19`
- Version: `signal_curriculum_plan_v1`
- Verification Status: `UNVERIFIED` (design proposal; no model changes in this document)

The deeper causal refinement is specified in
`NATURAL_MOMENTUM_COUNTERFACTUAL_CONTROL_20260920.md`. The NMCC paired-branch
pilot should be run before committing to the full reward curriculum: if exact
action-versus-`WAIT` effects supply a stable causal advantage, that signal is
preferable to hand-designed local proxies.

## Decision

The next model should not be tuned by merely increasing the casualty weight or
the fixed entropy bonus. The current recurrent credit path is mechanically
working, but the policy is learning from a noisy and weakly action-specific
signal. Three structural changes should come first:

1. add an explicit `WAIT` action so the controller can decline an unjustified
   shelter installation;
2. replace fixed exploration with a masked, update-indexed entropy schedule;
3. teach immediate operational mechanics with bounded, action-difference
   rewards that fade away, while retaining the unchanged global evacuation
   objective throughout training.

The GNN should then be extended only enough to estimate the marginal effect of
each feasible candidate. This sequence gives the policy a learnable causal
signal without making the model unnecessarily large.

## What the present implementation is telling us

The latest State College experiment does **not** show that gradients or delayed
credit are absent. Complete recurrent histories reach PPO, component GAE is
nonzero, the actor and critic receive gradients, and terminal reward accounting
closes. However, mean held-out return did not improve over the heuristic and
the training trend was not consistently positive. The practical conclusion is
that the existing signal is valid but statistically inefficient.

Four current design choices explain most of that inefficiency:

- Exploration is fixed at `entropy_coef = 0.005`; it does not begin broadly and
  contract as the policy becomes competent.
- The exact candidate mask contains only install actions. Whenever a decision
  is due and any site is feasible, the controller must install a shelter. It
  cannot learn operational restraint and an action-cost penalty would therefore
  punish behavior it has no ability to avoid.
- The actor scores each candidate from its host-region embedding and eight
  candidate features. It does not directly aggregate the geographically
  distributed regions that the candidate could serve.
- The reward is a correct population objective, but most of its variation
  arises after stochastic pedestrian and hazard evolution. It contains no
  immediate, action-specific measure of whether an installation improved the
  current route assignment, capacity deficit, or forecast-safe coverage.

The bounded learned residual is large enough to exceed the demand heuristic
prior, so the prior itself is not the primary bottleneck. The missing causal
contrast and forced-action design are more important.

## Proposed decision process

At every fixed decision epoch, the action set is

\[
\mathcal A_t = \{\text{WAIT}\}\cup\{\text{install candidate }j\}.
\]

`WAIT` is always feasible, consumes no deployment budget, and advances the
simulator to the next decision epoch. An installation remains irreversible and
consumes one unit of the deployment budget. Time, evacuation-time cost, and
hazard-exposure cost continue to accumulate after `WAIT`, so waiting cannot be
used to farm reward.

### Two-level action mask

The mask should distinguish physical infeasibility from predicted
inefficiency.

Hard-mask only actions that are certainly invalid:

- already installed or unavailable candidate;
- nonpositive capacity or exhausted installation budget;
- no usable path from any active demand region;
- forecast danger above the shelter safety threshold over the deployment lead
  time and minimum service window.

Do not hard-mask a candidate only because another candidate has a higher
predicted benefit. Forecasts and route estimates are imperfect; aggressive
dominance masks can remove the optimal action and destroy exploration. Instead,
represent dominance and low marginal value as features and let the action cost
make them unattractive. A Pareto-dominance mask may be added later only when
candidate `i` is no worse than `j` in every deterministic quantity—safe
capacity, reachable population, route-time reduction, exposure reduction, and
cost—and is strictly better in at least one.

### Equal-capacity evaluation invariant

The current simulator controls the number of additional shelters, but each
installed shelter inherits its selected site's `nodeCap`. RL and the heuristic
can therefore use the same number of actions while receiving different total
capacity. That is a confound when the estimand is shelter placement quality.

For the primary RL-versus-heuristic comparison, replace candidate-determined
implemented capacity with a scenario-level capacity-token schedule

\[
Q=(q_1,\ldots,q_K),\qquad \sum_{k=1}^{K}q_k=B,
\]

fixed before either policy observes the scenario. At its `k`th installation,
every policy places exactly token `q_k`; the action chooses location, not
capacity. Prefer equal tokens `q_k=B/K`. Retain a site's physical capacity only
as an eligibility limit: candidate `j` is feasible for token `q_k` only if its
declared limit is at least `q_k`. The observation must expose the token capacity
that would actually be installed, not the unused raw building capacity.

The common initial shelters must also be byte-identical, including capacity.
Then every successfully completed matched episode satisfies

\[
C^{RL}_{final}=C^{heuristic}_{final}=C_0+B.
\]

An unrestricted `WAIT` action conflicts with exact implemented-capacity parity:
one policy could leave capacity unused. For the primary capacity-controlled
comparison, `WAIT` may be used while there is scheduling slack, but it becomes
infeasible when the number of remaining deployment opportunities equals the
number of remaining capacity tokens. This preserves timing discretion while
requiring both policies to place the same capacity by the deployment deadline.
If the scientific question instead includes whether capacity should be left
unused, run that as a separate resource-efficiency experiment with a common
capacity **budget** and report realized capacity as an outcome; do not call
that design equal-implemented-capacity evaluation.

The parity gate must run before performance statistics and fail closed on any
of the following:

- different initial-shelter identity or capacity digest;
- different capacity-token schedule or dynamic capacity budget;
- an installed shelter whose capacity differs from its action-ordinal token;
- final dynamic capacity different from `B` for either policy;
- final total capacity different between RL and the heuristic;
- installation above a candidate's physical capacity limit.

Log `initial_capacity`, `capacity_budget`, `capacity_schedule_digest`,
`dynamic_capacity_added`, `final_total_capacity`, and capacity-time area for
every episode. Equal final capacity controls total supply; capacity-time area
reveals differences caused by deployment timing. For a location-only estimand,
fix deployment times as well. Never match the heuristic's capacity schedule to
the RL trajectory after observing RL actions, because that would make the
benchmark dependent on the learned policy.

## Immediate action-difference signal

Before executing an installation, run the existing deterministic route and
capacity calculator twice on the same observation:

- once with the present shelter set (`WAIT` baseline);
- once with candidate `j` provisionally available.

No pedestrian or hazard randomness is advanced in this comparison. The result
is an immediate marginal-benefit vector attributable to the proposed action,
not a reward for occupying an already favorable state.

| Feature | Notation | Computation |
|---|---:|---|
| Reachable population gain | \(b^{pop}_{t,j}\) | Fraction of active people who gain a forecast-safe, capacity-feasible shelter after adding `j`, minus the `WAIT` result. |
| Route-time reduction | \(b^{time}_{t,j}\) | Reduction in capacity-weighted shortest-path pedestrian-minutes under the provisional reassignment, divided by initial population times episode horizon. |
| Exposure reduction | \(b^{exp}_{t,j}\) | Reduction in hazard-integrated route dose for provisionally reassigned people, divided by the corresponding scenario scale. |
| Overload reduction | \(b^{load}_{t,j}\) | Reduction in unmet regional demand after capacity-constrained assignment, divided by initial population. |
| Forecast safety | \(m^{safe}_{t,j}\) | Minimum normalized time/distance margin between predicted hazard arrival and shelter service over the required service window. |
| Affected share | \(q_{t,j}\) | Active-population fraction whose assigned shelter or route changes when `j` is provisionally added. |
| Redundancy | \(d_{t,j}\) | Fraction of `j`'s reachable demand that already has an equally safe shelter within the same or lower travel time. |
| Forecast uncertainty | \(u_{t,j}\) | Dispersion of candidate danger or hazard-arrival time under the configured wind/spread perturbations. |

Each quantity is one scalar candidate feature. It must be normalized with a
fixed scenario-level denominator, not batch extrema, so the same value has the
same operational meaning across region resolutions and population sizes.

Define the immediate benefit of an installation as

\[
B_{t,j}=w_p b^{pop}_{t,j}+w_t b^{time}_{t,j}
       +w_e b^{exp}_{t,j}+w_l b^{load}_{t,j}.
\]

The early local reward is the bounded change caused by the chosen action:

\[
r^{local}_t=
\begin{cases}
\operatorname{clip}(B_{t,j},-b_{max},b_{max}), & a_t=j,\\
0, & a_t=\text{WAIT}.
\end{cases}
\]

This is not an absolute-state reward. Repeating an action in a good region
earns nothing unless the new action produces additional route, capacity, or
exposure benefit.

### Intervention cost

Every installation pays a fixed normalized operating cost `c_install`. An
installation with nonpositive predicted benefit also pays a waste penalty:

\[
r^{cost}_t=-c_{install}\mathbf 1[a_t\ne\text{WAIT}]
            -c_{waste}\mathbf 1[a_t\ne\text{WAIT}, B_{t,a_t}\le 0].
\]

The fixed cost is the mathematical hurdle for intervention: a component should
change state only when its marginal benefit exceeds the cost. `WAIT` receives
no positive bonus. It merely avoids the intervention cost while continuing to
incur the true global time and exposure consequences.

## Local-to-global reward curriculum

The global objective should be present from the first update and must never be
annealed away. Its existing components remain interpretable:

- new safe completions;
- new casualties with a coefficient larger than the maximum artificial
  time-cost reduction a casualty could create;
- active pedestrian-time;
- hazard-exposure pedestrian-time.

These are interval changes or accumulated costs, not rewards for simply being
in a good state. Continue accumulating the components after every action for
auditing, but release their sum to PPO only at the true terminal boundary:

\[
r^{global}_t=\mathbf 1[done_t]R^{population}_{episode}.
\]

This makes the global signal genuinely sparse and delayed while preserving the
same population objective. Terminal accounting must continue to equal the sum
of the audited interval components exactly. The intervention cost extends the
system-efficiency objective and must be applied identically to RL and every
benchmark when returns are compared.

For optimizer update `n`, use

\[
r_t=r^{global}_t+\alpha(n)r^{local}_t+r^{cost}_t.
\]

Recommended initial curriculum:

| Training progress | Local coefficient \(\alpha\) | Dense-reward episode cap | Target normalized entropy | Auxiliary-loss coefficient |
|---|---:|---:|---:|---:|
| 0–20%: mechanics | 0.50 | 30% of global reward scale | 0.85 | 1.00 |
| 20–60%: transition | linearly 0.50 to 0.10 | linearly 30% to 10% | linearly 0.85 to 0.35 | linearly 1.00 to 0.25 |
| 60–100%: global | linearly 0.10 to 0.00 | linearly 10% to 0% | linearly 0.35 to 0.15 | linearly 0.25 to 0.10 |

The cap is applied to the **sum of absolute local shaping rewards in an
episode**, not independently at each step. This prevents many micro-rewards
from overwhelming the sparse global outcome. The action cost remains active in
all phases because it is part of the operational objective rather than a
temporary teaching signal.

Casualty, exposure, unfinished-population, and intervention-budget caps should
also be reported as terminal constraint slacks. Do not begin with binary bonus
rewards for crossing those thresholds; discontinuous bonuses would add more
variance. If the continuous objective improves but a safety cap is repeatedly
violated, introduce a Lagrangian penalty with a slowly updated dual coefficient
as a separate, testable experiment.

The numerical coefficients are pilot values, not final claims. They should be
calibrated so a clearly beneficial shelter has positive immediate net value,
an ineffective installation has negative value, and no feasible sequence can
earn more from local shaping than from materially improving population
outcomes.

## Exploration schedule

Use one consistent masked categorical policy. Do not add an unrecorded
epsilon-random override, because that makes the PPO behavior probability differ
from the probability used in the update.

Replace the fixed entropy coefficient with a bounded controller that targets
normalized entropy over feasible actions:

- target `0.85` early so the agent samples broadly among safe candidates and
  `WAIT`;
- decay the target to `0.35` while the local reward fades;
- finish near `0.15` so actions become selective without becoming completely
  deterministic during training;
- constrain the learned entropy coefficient to `[1e-4, 3e-2]`;
- continue normalizing by `log(number of feasible actions)` so entropy has the
  same meaning when the mask or grid resolution changes.

The schedule must use optimizer-update count, be saved in checkpoints, and be
reconstructed on resume. The exact scheduled distribution must generate both
the sampled action and stored old log-probability. Evaluation remains masked
argmax with no exploration.

## Minimal GNN improvement

Keep the existing three region encoders, spatial edges, route-flow edges, and
LSTM. Replace only the candidate scoring interface.

1. Treat candidates as explicit typed nodes, including `WAIT` as a learned
   global action token.
2. Add `candidate-serves-region` edges for regions with a usable path to the
   candidate. Edge features are travel time, congestion ratio, route hazard
   dose, forecast safety margin, and assignable capacity share.
3. Apply one masked candidate-to-region attention layer. This lets each
   candidate pool the dispersed demand it could actually affect rather than
   inheriting only its host cell's embedding.
4. Concatenate the eight existing candidate features with the action-difference
   features above, then score all candidate nodes through one shared actor.
5. Retain the LSTM over full observation histories. The graph represents the
   current spatial decision; the LSTM represents how that decision context is
   changing.

Do not create one pedestrian node per evacuee for the 3,000-person experiment.
Regional population and route-flow aggregates preserve resolution flexibility
and keep computation stable. Individual routes contribute to aggregate edge
features and the provisional reassignment calculation.

### Auxiliary learning heads

During early training, make the encoder predict the deterministic
action-difference vector for every candidate. These supervised targets are
available without waiting for stochastic casualties. The heads predict route
time reduction, exposure reduction, reachable population, and overload
reduction. Their loss fades but does not affect evaluation reward. This teaches
useful representations while preserving the final policy objective.

The critic should retain separate outcome heads, but normalize each head's
target with running statistics or PopArt. Casualty events are rare and their
target scale shifts across scenario batches; per-head normalization avoids one
rare block destabilizing the shared trunk. The actor still uses the sum of the
signed component advantages so it optimizes one coherent return.

## Reducing variation without hiding risk

Training rollouts should be stratified by ex ante scenario risk, not stopped or
selected according to realized casualties. For each PPO update, sample a fixed
number of episodes from low, medium-low, medium-high, and high predicted-risk
strata. A first pilot should use 24–32 complete episodes per update, six to
eight from each stratum. This yields more stable casualty and exposure targets
without conditioning on the observed outcome.

Use common scenario seeds only for paired evaluation and ablation comparison.
Do not couple alternate actions inside the on-policy rollout unless a formally
defined counterfactual estimator is introduced. The deterministic provisional
assignment above already supplies the low-variance local contrast.

Natural, unstratified State College scenarios remain the final evaluation
distribution. Report both mean return and tail safety (casualty CVaR or upper
quantile), because an improved mean must not conceal a worse hazardous tail.

## Implementation order

1. **Action semantics:** add `WAIT`, make it always feasible, preserve budget on
   wait, and record a complete SMDP transition through the next decision epoch.
2. **Reward invariants:** add intervention cost and deterministic marginal
   benefit; prove no positive reward is available from repeated idle states;
   preserve exact accounting for the global objective.
3. **Noise control:** stratified complete-episode rollouts and per-critic-head
   target normalization.
4. **Exploration curriculum:** update-indexed target entropy with checkpointed
   schedule state.
5. **Feature intake:** action-difference features and auxiliary prediction
   heads.
6. **GNN change:** one candidate-to-region attention layer and a `WAIT` token.
7. **Reward curriculum:** fade bounded local signal only after mechanics and
   intervention tests pass.

This order isolates causal failures. It also avoids interpreting an apparent
gain from a larger network when the true improvement came from action
semantics or sampling variance.

## Required tests before full training

- `WAIT` never consumes deployment budget and still advances time and global
  person-time accounting.
- Repeated `WAIT` cannot produce positive local or action-cost reward.
- Re-evaluating an unchanged installed shelter cannot produce local benefit.
- Every physical hard-mask reason is deterministic and administrator-visible.
- `WAIT` is feasible even when every shelter candidate is unsafe.
- The sampled action, old log-probability, replayed log-probability, and entropy
  all use the same scheduled masked distribution.
- Dense local reward is exactly zero after the curriculum reaches zero.
- The episode local-reward cap cannot be exceeded.
- Global reward component sums still close at terminal accounting precision.
- Candidate scores are permutation equivariant and work at multiple grid
  resolutions.
- Candidate-to-region edges never connect across graphs in a packed batch.
- Critic normalization can be saved, loaded, and resumed without changing a
  deterministic evaluation.

## Backtest ladder and acceptance criteria

Do not launch one large experiment containing every change. Use paired
ablations on identical training and evaluation scenarios:

| Experiment | Increment over current recurrent PPO |
|---|---|
| E0 | Current implementation; no changes. |
| E1 | `WAIT`, intervention cost, and hard physical masks. |
| E2 | E1 plus stratified 32-episode rollouts and critic target normalization. |
| E3 | E2 plus scheduled target entropy. |
| E4 | E3 plus deterministic action-difference features, auxiliary heads, and local reward curriculum. |
| E5 | E4 plus candidate-to-region attention. |

Pilot with at least three policy seeds and 64 training episodes per seed. Move
only promising configurations to a confirmatory run with at least five policy
seeds, 120 or more training episodes per seed, and 30 or more held-out scenario
seeds. Use paired bootstrap intervals over held-out scenarios.

Minimum promotion gates:

- held-out mean return exceeds both E0 and the operational heuristic;
- the paired 95% interval for the return improvement excludes zero in the
  confirmatory run;
- casualty mean and upper-tail risk are not worse than the heuristic;
- the policy uses `WAIT` more often when all candidate benefits are below the
  intervention cost, and installs more often when high-benefit safe candidates
  exist;
- training entropy follows the schedule, but held-out masked-argmax actions are
  stable;
- critic explained variance improves without gradient spikes or component loss
  domination;
- gains survive removal of the local reward in the final training phase and in
  evaluation.

If E4 improves learning but E5 does not, retain the simpler host-region GNN.
The candidate attention layer is justified only by out-of-sample gain, not by
architectural novelty.

## Expected result

This design gives early PPO updates a low-variance answer to “did this specific
installation improve the current evacuation plan?”, then gradually removes
that teaching signal so the mature policy must optimize actual safe completion,
casualties, evacuation time, and exposure. `WAIT` plus explicit intervention
cost turns shelter placement into a genuine intervention-control problem. The
resulting policy can explore broadly at first, become selective later, and
remain comparable to the same global objective used by every benchmark.
