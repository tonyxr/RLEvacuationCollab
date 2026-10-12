# Positive remaining-risk-time reduction MDP (v33)

At deployment epoch `k`, let `W_k^-` be the summed remaining
hazard-weighted travel time of all active pedestrians under the current
capacity reservations and routes. After the selected equal-capacity shelter is
installed and every rational pedestrian is reassigned, let the same quantity
be `W_k^+`. The sole reward component is

```
r_k = max(0, W_k^- - W_k^+) / (P H).
```

`P` is initial population and `H` is the episode horizon in minutes (120 in
the registered State College experiment). The environmental return is
nonnegative. PPO still obtains signed advantages by comparing each action's
positive return with its learned/batch baseline.

For a pedestrian with a capacity-reserved operational shelter, remaining time
comes from the actual remaining route distance and current realized speed.
Each route segment is multiplied by `1 + max(current danger, forecast danger)`
in its host cell. Ordinary route time is capped at `H`. A pedestrian without a
valid capacity reservation—including a panicked pedestrian—contributes `H`
weighted by danger at the current cell. A provisional overflow route is not
treated as shelter access.

Reward is evaluated at the installation instant. Simulator evolution between
deployments, safe-completion counts, casualties, capacity added, utilization,
and terminal residual time do not add reward. They remain outcome diagnostics.

The behavior policy samples the factual candidate. Exact NMCC independently
samples a second feasible candidate from that same behavior distribution,
restores the identical pre-installation state, installs the same capacity, and
computes the same immediate reduction. Actor credit is the signed difference

```
A_CF = scale * (r_chosen - r_reference).
```

There is no future-noise rollout, continuation-value bootstrap, action gate,
teacher, correction, or fallback in this comparison. The critic learns the
return-to-go over the sequence of positive installation rewards.
