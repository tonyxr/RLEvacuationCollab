# Hazard-weighted time-to-safety MDP and candidate-pair PPO (v31)

Model version 31 implements one control rule:

> At every deployment decision, increase the probability of the executable
> shelter that removes more expected future hazard-weighted pedestrian time
> than another shelter the current policy would install.

The control path is direct:

\[
\text{observation}\rightarrow
\text{categorical policy sample}\rightarrow
\text{installed shelter}\rightarrow
\text{paired physical outcomes}\rightarrow
\text{PPO update}.
\]

There is no heuristic prior, action correction, benefit gate, rollback gate,
teacher, fallback controller, or NMCC action proposal.

## MDP

The state contains the recurrent observation history of regional pedestrian
flow, current and forecast hazard, congestion and travel conditions, existing
capacity, candidate capacity and location, remaining time, and remaining
deployment budget. Momentum is observable history, not privileged future
state.

An action is one structurally executable regional cell. The shared lower layer
installs the maximum-capacity remaining building in that cell with a stable
identifier tie break. The action mask excludes only exhausted deployment
budget and cells without a positive-capacity candidate. Forecast danger and
estimated benefit remain policy inputs and never censor an otherwise physical
action.

Training samples from the masked categorical policy. Evaluation selects the
largest learned logit. Temperature and entropy decay are exploration controls;
they do not alter rewards or counterfactual labels.

## Objective

For active pedestrian (p) at timestep (t), normalized danger is
(d_{p,t}\in[0,1]). The running social cost is

\[
C_t=\sum_{p\in\mathcal A_t}(1+d_{p,t}).
\]

The fixed-normalization episode reward is

\[
R=-\frac{\sum_t C_t+\Phi(s_T)}{P_0H}
  -5\frac{D}{P_0}.
\]

Only ordinary arrival or actual shelter admission removes a pedestrian from
the active set. Rerouting and capacity reservation do not count as completion.
There is no separate completion reward.

For every pedestrian unfinished at the finite horizon,

\[
\Phi(s_T)=\sum_{p\in\mathcal A_T}
\widehat\tau_{p,T}(1+\widehat d_{p,T}).
\]

Remaining time uses assigned route distance and realized speed, has a one-step
minimum, and is capped at one episode horizon. A missing or unreachable route
receives the full continuation horizon. Estimated danger is the maximum of
current danger and the public operational forecast at the pedestrian's current
cell.

Running plus terminal cost is bounded by four normalized person-horizons.
Therefore casualty weight five makes death strictly worse than removing that
person's maximum possible future time debt.

## Exact candidate-pair NMCC credit

At decision state (s):

1. Sample and execute (a\sim\pi_\theta(\cdot\mid s)).
2. Independently sample (b\sim\pi_\theta(\cdot\mid s)) from the identical
   masked, temperature-scaled behavior distribution.
3. Install one capacity-matched shelter in each branch from the identical
   simulator snapshot.
4. Replay both branches under common hazard, casualty, panic, and route-choice
   randomness.
5. Score both with the same HWTTS reward and component critic.

The signed actor advantage is

\[
A_{\mathrm{pair}}(s,a,b)=
(R_a-R_b)+\gamma^L[V(s_L^a)-V(s_L^b)].
\]

Because (b) is independent of (a) and drawn from the behavior policy,

\[
\mathbb E_b[R_b\mid s]
=\mathbb E_{c\sim\pi_\theta}[R_c\mid s],
\]

so this is a one-sample estimate of the within-state policy-average baseline.
It directly produces positive credit for a better-than-policy-average shelter
and negative credit for a worse one, even when both installations would beat
WAIT. The reference is allowed to equal the factual action; that sample has
zero location credit. Excluding equality would make the baseline depend on the
chosen action and bias the gradient.

The branch horizon normally equals the deployment interval. The component
critic supplies identical-form continuation values at both endpoints and is
trained from complete realized episode returns. Missing paired credit is a
training error; ordinary PPO credit is never substituted.

## Capacity and difficult outcomes

Capacity is not approximated inside the reward. Both branches use the actual
reservation, admission, cohort-indivisibility, congestion, panic, and rerouting
logic. Consequently:

- unused or redundant capacity earns no invented benefit;
- overlapping shelters compete for the same people through realized outcomes;
- every installation rebuilds all rational pedestrians' closest feasible
  destinations and capacity reservations;
- people beyond committed capacity follow provisional closest-shelter routes,
  remain active, and continue accruing cost;
- harmful rerouting, congestion, future hazard, and casualties create negative
  credit;
- natural completions shared by both branches cancel under common noise; and
- with one executable candidate, factual and reference actions coincide and
  location-learning credit is correctly zero.

The principal diagnostics are raw hazard-weighted person-time removed,
casualties avoided, signed advantage mean and dispersion, positive and negative
fractions, and the equal-action fraction. Policy updates are standard clipped
PPO updates; KL remains diagnostic and does not reject or replace updates.
