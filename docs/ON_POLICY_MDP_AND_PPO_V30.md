# Shelter-placement MDP, direct PPO, and exact NMCC optimization (v30)

This document is the executable design contract replacing the conservative
fitted-correction experiment. The control path is deliberately linear:

\[
\text{MDP observation}\rightarrow
\text{neural policy action}\rightarrow
\text{physical outcome}\rightarrow
\text{RL return}\rightarrow
\text{exact NMCC credit}\rightarrow
\text{PPO update}.
\]

There is no heuristic controller inside this chain. NMCC is an optimization
and credit-assignment layer after the policy has selected its action; it is not
an action proposer, correction policy, safety gate, or fallback.

## 1. Control objective

The controller sequentially chooses regional cells in which to install
additional shelters during one urban-fire evacuation. A training episode is
one complete stochastic evacuation. The same policy that samples the shelter
actions also receives their realized consequences and is updated by PPO.

The implemented interval reward is

\[
r_k = \frac{\Delta S_k - 3\Delta C_k}{P}
      - \frac{E_k}{PH},
\]

where:

- \(P\) is the initialized population;
- \(H\) is the episode horizon in simulator timesteps;
- \(\Delta S_k\) is the number of new safe completions after action \(k\);
- \(\Delta C_k\) is the number of new casualties after action \(k\); and
- \(E_k=\sum_t\sum_p d_{p,t}\) is hazard-exposure person-time accumulated
  during the action interval, with normalized danger \(d_{p,t}\in[0,1]\).

There is no reward for selecting a particular site, matching a heuristic,
rerouting a person, filling a shelter, or agreeing with a counterfactual
model. Positive and negative reward arise only from simulated population
outcomes. The undiscounted episodic return is \(G_0=\sum_k r_k\).

The safe-completion and casualty terms remain explicit in addition to the
primary exposure-time term. They prevent the degenerate solution in which a
person leaves the active population through death and ensure that reaching a
valid destination remains an outcome of interest.

## 2. Formal decision process

### 2.1 Classification

The physical simulator is a finite-horizon stochastic MDP in its complete
internal state. The controller receives an aggregated observation rather than
every latent pedestrian and random-generator state, so the learning problem is
a finite-horizon partially observed semi-Markov decision process. An episode
contains at most eight shelter decisions in the registered v30 experiment.

### 2.2 Simulator state

The unobserved physical state \(x_t\) contains:

- every pedestrian's location, current edge, route, speed, capacity
  reservation, exposure state, panic state, and termination state;
- every walking-network edge and its current bidirectional occupancy and
  congestion;
- every hazard's source, position, age, intensity, spread state, wind inputs,
  and stochastic generator state;
- every installed shelter's location, capacity, reservations, admissions, and
  availability;
- every remaining candidate site, the deployment budget, clock, and cumulative
  outcome ledger.

Given \(x_t\), the chosen action, and the next exogenous random draws, the next
simulator state is independent of earlier physical states.

### 2.3 Decision epochs and transition duration

The first decision occurs at the first configured opportunity. After an
installation, the next opportunity occurs 15 simulated minutes later. An
action transition closes at the next action actually executed or at the
physical terminal boundary. Therefore transition \(k\) has duration
\(\Delta t_k\), and its reward contains all outcomes strictly after its
installation through the closing boundary.

The registered horizon is 120 simulated minutes (`stopTime = 121` under the
simulator's boundary convention), with a maximum of eight additional shelters.
At most one capacity token of 350 people is installed at a decision.

### 2.4 Observation

At every simulator boundary, the policy encoder receives an operational graph.
The recurrent state is advanced even between shelter decisions, so a decision
can depend on the observed trajectory rather than only the current snapshot.

Each of the 64 regional cells contains:

- active-population fraction;
- mobility-delay fraction;
- normalized mean remaining route time;
- long-route population share;
- stable, hazard-exposed, and panicked population shares;
- current and one-interval forecast danger;
- hazard-source proximity;
- remaining installed shelter capacity;
- shelter utilization including en-route reservations;
- deployable capacity and remaining-candidate share;
- road-network-node share;
- normalized east/north position and cell area.

The global vector contains:

- time remaining;
- active-population fraction;
- remaining-deployment fraction;
- network load;
- wind speed and direction;
- hazard-spread parameter;
- population/network density;
- hazard-instance load;
- configured panic susceptibility; and
- dynamic-deployment capacity coverage.

Every action slot also has candidate-specific features for the site that would
currently be installed in that cell:

- capacity;
- normalized position;
- distance from an existing open shelter;
- forecast danger;
- distance-based hazard safety margin;
- capacity-capped reroutable population; and
- the analytical risk-time-reduction estimate.

Forecast danger, hazard distance, reroutable population, and analytical
risk-time reduction are observations only. They do not mask actions, create
rewards, initialize policy preferences, or supervise the actor in v30.

The temporal input additionally reports recent rates of safe completion,
casualty incidence, active-population clearance, exposure reduction, route-time
reduction, long-route reduction, forecast-risk reduction, network-load
reduction, and time since the last deployment.

### 2.5 Action

The action space has one stable slot per regional cell:

\[
a_k\in\{0,\ldots,63\}.
\]

Selecting a cell installs the currently resolved candidate site in that cell.
The deterministic lower layer chooses the positive-capacity candidate with the
largest configured capacity, using OSM identifier as a stable tie break. This
lower-level site resolution is part of the environment, not part of the
learned policy.

The v30 mask contains only structural feasibility:

- a deployment token remains;
- the cell has an uninstalled candidate; and
- that candidate has positive deployable capacity.

The previous positive-benefit condition is disabled. Forecast danger up to the
full normalized range is allowed, and the minimum hazard-distance margin is
zero. The policy can therefore select a locally useless or hazardous but
physically valid site and receive its negative consequence.

There is currently no explicit `WAIT` action. If at least one physical action
is feasible at a due decision epoch, one shelter is installed. Adding `WAIT`
would change the scientific action space and should be a deliberate separate
decision rather than an implicit safeguard.

### 2.6 Transition and terminal condition

After every installation, the simulator recomputes the closest reachable
shelter for every active non-panicked pedestrian from that pedestrian's current
routing anchor. Capacity reservations are rebuilt over the full active
population; overflow pedestrians receive provisional closest-shelter routes so
they move while waiting for later capacity. Movement, congestion, exposure,
panic, casualties, shelter admission, ordinary arrival, and stochastic hazard
spread then evolve normally until the next decision or termination.

An episode terminates when the configured horizon is reached or no active
pedestrians remain. Unfinished pedestrians remain part of the terminal outcome;
they do not receive an artificial completion reward.

## 3. Policy and value model

### 3.1 Shared observation encoder

The network uses:

1. separate pedestrian, hazard, and infrastructure encoders per region;
2. two relational graph-message layers over spatial and route-flow edges;
3. attentive and maximum graph pooling;
4. the global dashboard vector; and
5. an episode LSTM advanced at every observed simulator boundary.

The actor scores every cell from its regional embedding, candidate features,
global context, and recurrent state. With `actorPrior = none`, logits are fully
learned and start at zero, giving a uniform distribution over feasible actions.
There is no heuristic logit term and no bounded correction residual in this
mode.

The critic predicts four signed return components—safe completion, casualty,
ordinary evacuation time, and hazard exposure—which sum exactly to the scalar
value. The ordinary evacuation-time reward weight is currently zero, but the
component is retained so reward accounting remains explicit.

No counterfactual outcome ensemble, intervention-value head, teacher, or
correction model is instantiated. Exact NMCC uses simulator outcomes directly.

### 3.2 Training action selection

For feasible set \(F_k\), the behavior policy is

\[
\pi_{\theta_{old}}(a\mid o_k)
=\operatorname{softmax}\left(z_\theta(o_k,a)/T_u\right),\qquad a\in F_k,
\]

where \(u\) is the number of completed actor rollout updates. The training
action is sampled from this categorical distribution. Over the 13 rollout
updates in the registered 104-episode campaign, temperature decreases linearly
from 1.5 to 0.35 and the entropy coefficient decreases linearly from 0.01 to
0.0005. There is no base-policy mixture, epsilon-greedy action override,
correction gate, or heuristic prior. Evaluation uses the feasible argmax
deterministically.

## 4. On-policy data and credit assignment

Eight complete episodes form one rollout. For every executed action the buffer
stores:

- the complete recurrent observation history available at selection time;
- the sampled action;
- its old-policy log probability;
- the four value predictions;
- the realized interval reward components;
- terminal status; and
- the physical duration of the interval;
- the exact same-noise `WAIT` return for that state; and
- the signed sampled-action-minus-`WAIT` NMCC effect.

No replay buffer supplies actor updates. No action is relabeled, replaced, or
rejected by a heuristic or counterfactual branch.

For component \(c\), complete realized return-to-go is

\[
G^c_k=r^c_k+\gamma^{\Delta t_k}G^c_{k+1},
\]

with \(\gamma=1\) in the registered experiment. Episode-terminal flags reset
the recursion. The scalar return is \(G_k=\sum_c G^c_k\).

The ordinary on-policy actor advantage is

\[
\hat A_k=G_k-V_{\theta_{old}}(o_k).
\]

This remains a diagnostic and is the actor target when exact NMCC is disabled.
The critic always trains on the complete realized component return \(G^c_k\),
not a one-step bootstrap or a branch-model estimate.

### 4.1 Exact NMCC credit after the RL action

Before the sampled action is installed, the simulator state is snapshotted.
One paired branch advances for one decision interval without installing a
shelter. The live trajectory then installs and executes the policy's sampled
action. Both trajectories use the same exogenous future-noise state. NMCC
therefore computes, component by component,

\[
A^{NMCC}_k=
R_k^{a_k}-R_k^{WAIT}
+\gamma^{\Delta t_k}
\left[V(o_{k+1}^{a_k})-V(o_{k+1}^{WAIT})\right].
\]

This is still credit for the policy's own action. `WAIT` is an
action-independent control variate that removes hazard and crowd evolution
shared by both futures. It does not tell the policy which cell to choose. A
sampled shelter that is worse than waiting produces \(A^{NMCC}_k<0\); a better
one produces \(A^{NMCC}_k>0\). Division by a positive rollout scale stabilizes
gradients without changing either sign.

The registered optimizer uses \(A^{NMCC}\) wherever the exact pair is
available. Coverage is required to be 100%; training fails loudly if a sampled
action lacks its paired branch rather than falling back to another controller
or credit target. It never substitutes a model prediction. The ordinary
realized return remains the critic target and the final evaluation objective.

## 5. PPO optimization

For each stored transition, PPO recomputes the current recurrent policy and
forms

\[
\rho_k(\theta)=
\frac{\pi_\theta(a_k\mid o_k)}
     {\pi_{\theta_{old}}(a_k\mid o_k)}.
\]

For the registered run, the actor uses
\(\hat A_k=A^{NMCC}_k/\max(\operatorname{sd}(A^{NMCC}),0.05)\) and minimizes

\[
L_{actor}=-\mathbb E_k\left[
\min\left(
\rho_k\hat A_k,
\operatorname{clip}(\rho_k,1-\epsilon,1+\epsilon)\hat A_k
\right)
\right]-\beta H(\pi_\theta),
\]

with clipping \(\epsilon=0.10\). The entropy coefficient and policy temperature
follow the exploration schedule above. Four actor epochs and four critic epochs
are used per rollout.

KL is reported as a diagnostic but does not veto an actor update, roll the
policy back, or reduce its learning rate. PPO's clipped surrogate is the only
policy-ratio constraint in the direct learner.

The critic minimizes the mean component-wise smooth-L1 error

\[
L_{value}=\frac{1}{4}\sum_c
\operatorname{SmoothL1}\left(V^c_\phi(o_k)-G^c_k\right).
\]

Actor gradients train the GNN, LSTM, and actor head. In the registered
`actor_owned` mode, critic gradients train only the critic trunk and value
heads; they cannot silently change the behavior policy between PPO ratio
calculations. Gradients are clipped to norm 0.5. Actor and critic use AdamW.
Their learning rates follow independent cosine decays over rollout-update
progress: both actor and critic decrease from \(10^{-3}\) to \(10^{-4}\).
The schedule starts
at its maximum on the first rollout update and reaches its minimum on the
thirteenth, so approximately the first 10--20% of episodes retain the largest
steps while later updates become progressively smaller.

PPO clipping, numerical gradient clipping, and action masking for physical
impossibility remain optimization/validity mechanisms. They are not policy
safeguards and do not prevent negative-return actions from entering training.

## 6. Mechanisms removed from the active learner

The v30 control and gradient path contains none of the following:

- the risk-reduction heuristic actor prior;
- epsilon-greedy mixing with that base policy;
- fitted intervention-value correction heads;
- the correction LCB;
- the held-out correction gate and three-update rule;
- all-candidate counterfactual search;
- a learned WAIT or counterfactual model;
- natural/causal staged pretraining;
- teacher guidance and imitation targets;
- positive operational-benefit action filtering;
- forecast-danger action filtering; and
- hazard-distance action filtering.

The bridge now rejects attempts to enable the retired fitted-improvement gate
or a heuristic actor prior. These are not dormant switches in the v30 learner.

The only counterfactual computation retained is the exact paired `WAIT` outcome
for the action the policy itself sampled. There is no held-out release gate and
no fallback action.

## 7. What constitutes learning evidence

A training run should be judged from quantities produced by the on-policy path:

- episodic objective return and its three active components;
- fraction safely completed, casualties, unfinished population, and total
  hazard-exposure person-time;
- policy entropy and feasible action diversity;
- raw advantage mean and standard deviation;
- NMCC action-minus-`WAIT` mean, standard deviation, negative fraction, and
  coverage;
- PPO loss, clipping fraction, and KL;
- critic loss and explained variance; and
- deterministic evaluation against fixed scenario seeds and heuristic/random
  benchmarks after training.

Correction-gate gain, fitted-branch rank, model uncertainty, and gate status are
not part of this learner and must not be used as success criteria.

## 8. Registered implementation

The executable curriculum is
`config/state_college_training_curriculum_2500_on_policy_ppo_v30.json`.
The reward is defined in `RewardProcessor.py`, observation/action construction
in `DecisionInterface.py`, the recurrent policy in `GNN.py`, and transition and
PPO optimization in `RLBridge.py`.
