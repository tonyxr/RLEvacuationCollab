# Clean NMCC policy improvement (v35)

The active learner has two sequential optimization steps on each rollout.

## 1. PPO learns the sequential task

For every executed shelter action, PPO uses the complete realized return-to-go
from the episode and the learned value baseline:

```text
A_PPO(s_k, a_k) = G_k - V(s_k)
```

The usual clipped PPO loss updates the categorical shelter policy. NMCC does
not replace this advantage, so delayed consequences of early installations
remain part of learning.

## 2. NMCC improves the action ranking at visited states

Before executing the sampled action, the simulator captures the decision state
and evaluates:

```text
WAIT, ACTION(a_1), ACTION(a_2), ..., ACTION(a_m)
```

for every physically feasible candidate. All branches share the same captured
random state, hazard trajectory, rational pedestrian cohort, horizon, shelter
capacity, and routing rules. WAIT is simulated once and reused as the common
control. No later shelter is selected inside a branch, so there is no heuristic
continuation policy.

Let `Q_i` be candidate `i`'s positive matched-WAIT exposure-removal reward and
let `pi_old` be the exact behavior distribution. NMCC centers the full value
table:

```text
A_i = Q_i - sum_j pi_old(j | s) Q_j
```

and computes the KL-constrained MPO target:

```text
q(i | s) proportional to pi_old(i | s) exp(A_i / eta)
KL(q || pi_old) <= epsilon.
```

After PPO, the NMCC M-step minimizes `KL(q || pi_theta)` over the same visited
states. This updates all feasible candidate probabilities from one comparison,
instead of updating only one sampled candidate.

## Information boundary

The policy and benchmarks observe regional population, hazard, capacity,
geometry, accessibility, and momentum. They do not observe a precomputed
candidate ranking based on scanning every pedestrian route. The coarse
`risk_reduction` benchmark uses only capacity-capped active demand in the
candidate's own cell weighted by local observed/forecast danger.

The complete candidate consequence table exists only during training inside
the NMCC branch optimizer. Evaluation uses the learned policy with no branch
oracle.

## Exploration and learning-rate schedules

Training samples from a categorical policy with three half-cosine schedules:

- actor temperature;
- entropy coefficient;
- uniform feasible-action mixture probability.

Actor and critic learning rates also use half-cosine decay. The schedules are
continuous functions of rollout-update progress and never switch abruptly.

## Explicitly absent

- heuristic base or fallback policy;
- correction/safety gate;
- top-k candidate branching;
- learned values for unbranched actions;
- engineered candidate-ranking input;
- casualty or panic reward term.

Casualties and panic remain physical simulation outcomes and are still logged
for evaluation.
