# Scalable sampled NMCC (model v38)

## Why v37 was slow

At each of eight decisions, v37 simulated every feasible shelter and one WAIT
branch through the next 15 timesteps under two future tapes. With 20 feasible
sites this is 42 physical branches per decision and as many as 336 branches per
episode. On the 5,000-person State College network, one decision required
157.3 seconds. Continuation inference required only about one tenth of a
second; physical pedestrian/congestion simulation was the bottleneck.

## Sampled exact contrast

Let `a` be the action sampled and executed by the behavior policy. The branch
set `B(s)` always contains `a`; the remaining configured slots combine the
highest-probability actions and uniform samples from the feasible set. The
registered v38 run uses `|B(s)| = 2`, so the second action provides uniform
coverage. No heuristic chooses either action.

For tape `w`, every action in `B(s)` is restored from the same simulator state,
installs the same shelter-capacity token, and evolves for one complete decision
interval. Choose the executed action as an arbitrary comparator `a0`. The value
of each sampled action is represented up to a state constant:

```text
Q_hat(s, a0; w) = 0

Q_hat(s, a; w)
  = [J_L(s, a0; w) - J_L(s, a; w)] / (N H)
    + gamma^L [V_NMCC(s'_a) - V_NMCC(s'_a0)].
```

`J_L` is the realised plain plus danger-weighted time-to-safety cost over the
branch and its endpoint residual. Because policy improvement depends only on
action differences, setting the comparator to zero loses no information. It
also removes the old WAIT branch: equal capacity, hazard evolution, and other
shared natural momentum cancel directly between installed actions.

The physical calculation remains exact for sampled actions. It includes route
reassignment, reservations, congestion, pedestrian movement, hazard spread,
panic/casualty randomness, the 15-timestep interval, and the dedicated learned
continuation value. There is no shortened branch or surrogate simulator.

## MPO update without fabricated labels

Average `Q_hat` over the common-noise tapes. The E-step changes only the
conditional probability of the sampled actions:

```text
q_B(a | s) proportional to pi_old(a | s) exp(Q_hat(s,a) / eta),  a in B(s)

sum_{a in B} q(a|s) = sum_{a in B} pi_old(a|s)
q(a|s) = pi_old(a|s),                                      a not in B(s).
```

`eta` satisfies the configured KL budget. The M-step minimizes
`KL(q || pi_theta)` using the same recurrent actor and stops at the configured
absolute residual tolerance. An unbranched action is not labeled good, bad, or
average; its probability is unchanged in that state. Across rollout states,
uniform coverage supplies observations for the full candidate set while the
executed action keeps training concentrated on decisions the policy actually
makes.

The dedicated continuation head is unchanged. It is fitted by a backward
semi-Markov return sweep over all eight factual installations, so sampled NMCC
still values how a current placement changes later decisions.

## Complexity and measured timing

With two sampled actions and two tapes, v38 runs four physical branches per
decision instead of 42. Those four tasks occupy the four configured workers in
one wave.

- 5,000-person first State College decision: 157.3 s to 22.0 s, a 7.1x speedup
  and 86.0% reduction.
- Continuation batch at that decision: 0.013 s.
- Complete fresh-process 2,500-person action episode with eight installations:
  157.6 s including map initialization; the 180-timestep simulator loop took
  122 s.

The remaining irreducible cost is one 15-timestep physical simulator wave at
each decision. Reducing it further would require fewer tapes, fewer scored
decision states, reuse of the factual future, or optimization of the pedestrian
simulator itself; none of those changes is part of v38.

## Compatibility

The actor-target semantics are part of the checkpoint signature. V37
checkpoints therefore cannot resume v38 training. This is deliberate: silently
continuing a checkpoint under a different policy-improvement operator would
make the experiment unauditable.
