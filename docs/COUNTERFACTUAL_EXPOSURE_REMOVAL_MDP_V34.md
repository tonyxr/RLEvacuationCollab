# Matched-WAIT exposure-removal MDP (v34)

> Reward contract retained by v35. The sampled candidate-pair actor credit in
> the final section is now an ablation; the active optimizer is documented in
> `CLEAN_NMCC_POLICY_IMPROVEMENT_V35.md`.

At deployment epoch `k`, the simulator captures one decision state and replays
two reward-only trajectories under the same hazard evolution:

1. `WAIT`: install nothing and perform the normal global route refresh.
2. `ACTION(a)`: install the selected equal-capacity shelter and perform the
   identical route refresh.

For branch `b`, through the next decision interval `L`, define

```text
plain_b = accumulated active-person minutes + endpoint plain time-to-safety
exposure_b = accumulated danger-weighted person minutes
             + endpoint hazard-exposure time-to-safety
J_b = plain_b + exposure_b
```

The signed causal gain and displayed environmental reward are

```text
g_k(a) = J_WAIT - J_ACTION(a)
r_k(a) = max(0, g_k(a)) / (P0 H_minutes).
```

The endpoint estimate uses the capacity-reserved shortest route and realized
speed. A rational pedestrian without a valid operational reservation
contributes the complete episode horizon: 120 minutes in the registered State
College experiment. The danger term is additional to plain evacuation time,
so an action that removes both delay and dangerous exposure receives more
credit than one that removes the same plain time in a safe area.

## Route contract

- A valid existing reservation is never displaced merely to rebuild the
  capacity ledger.
- A pedestrian changes destination only when the new capacity-valid shelter is
  strictly closer from the current routing anchor, or when the previous
  reservation was `NONE` or invalid.
- Reservation changes are atomic and cohorts remain indivisible.
- Capacity released by a strict improvement can be reassigned to an unserved
  pedestrian.
- An unchanged destination retains its current remaining shortest path. A
  missing or invalid route is repaired in both WAIT and ACTION and therefore
  creates no shelter reward.
- An unreserved pedestrian may keep a provisional route to remain mobile, but
  that route is not treated as shelter access.

## Panic and casualty

Panic and casualty remain enabled in the real episode and are reported in all
outcome summaries and graphs. They have no reward coefficient. To prevent a
death or panic transition from shortening the reward clock, the reward-only
twins use the rational, active cohort present at the decision and disable new
panic and casualty draws. Hazard spread, exposure-dependent speed, congestion,
movement, reservations, shelter admission, and safe completion remain active.

## Learning signal

The episode return sums the nonnegative `r_k`. PPO/NMCC action ranking retains
the signed raw gain before clipping. For independently sampled equal-capacity
candidates `a` and `b`, actor credit is

```text
A_CF = scale * (g_k(a) - g_k(b)).
```

Thus two harmful candidates do not both collapse to an indistinguishable zero
learning target, while the displayed environmental reward remains
nonnegative. There is no capacity bonus, casualty term, panic term, completion
bonus, site score, correction gate, or episode-end residual reward.
