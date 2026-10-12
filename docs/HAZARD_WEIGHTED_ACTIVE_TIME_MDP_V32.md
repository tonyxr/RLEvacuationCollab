# Summed hazard-weighted active-time MDP (v32)

Model version 32 gives the shelter policy one optimized objective:

> Reduce the future hazard-weighted exposure time summed over all pedestrians
> who are still active in the evacuation network.

## Reward

For active pedestrian (p) at timestep (t), let (d_{p,t}\in[0,1]) be the
normalized danger at that pedestrian's current cell. The population cost at
that timestep is

\[
C_t=\sum_{p\in\mathcal A_t}(1+d_{p,t}).
\]

The constant term values ordinary time-to-safety; the danger term makes the
same delay more costly in a hazardous location. The environment reward is

\[
r_k=-\frac{
\operatorname{active\_person\_time}_k+
\operatorname{danger\_exposure\_person\_time}_k+
\operatorname{terminal\_residual\_person\_time}_k
}{P_0H}.
\]

The denominator uses fixed initial population (P_0) and episode horizon
(H). It never divides by the current population. Only ordinary arrival or
actual shelter admission counts as safe completion and stops future active
time. People still active at the finite horizon receive a route-, speed-, and
forecast-danger-weighted continuation estimate.

There is no separate safe-completion bonus, casualty penalty, shelter-service
bonus, site-selection bonus, correction gate, or fallback reward. PPO and its
critic therefore optimize one component named `hazard_weighted_time`.

## Casualties

Casualty is a physical outcome, not a reward component. It remains present in:

- simulator and episode summaries;
- timestep progress logs;
- exact paired-branch physical outcome records;
- held-out paired evaluation and uncertainty intervals; and
- training and evaluation figures as a separately scaled casualty trend.

This separation makes it possible to inspect whether reducing exposure also
reduces casualties without giving the learner an additional sparse terminal
coefficient.

## Action and learning

At each deployment opportunity the recurrent graph policy samples one feasible
shelter candidate. An independent sample from the same categorical behavior
policy supplies a capacity-matched reference candidate. Both installations are
replayed from the same state under identical future hazard, casualty, panic,
and route-choice randomness.

The signed learning signal is

\[
A_{\mathrm{pair}}(s,a,b)=
(R_a-R_b)+\gamma^L[V(s_L^a)-V(s_L^b)].
\]

Because both (R) and (V) contain only summed hazard-weighted time, positive
credit means the sampled shelter removed more future hazard-weighted pedestrian
time than the independently sampled reference shelter. Casualties avoided are
computed directly from the two physical branch outcomes and logged only as a
diagnostic.

Training remains direct recurrent PPO. NMCC supplies exact paired credit but
does not propose, correct, veto, or replace the sampled action. Evaluation is
deterministic argmax over learned shelter scores.
