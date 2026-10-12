# Can the siting differential be separated from the capacity effect every installation buys?

## Material Passport

- Origin: Claude (Cowork), at the user's request: "I think one of the biggest
  current issue is any installation would mean an increase in remaining
  capacity, which would result in evacuation progress, so we still have to be
  able to differentiate the reduction speed of hazard-weighted exposure time
  from a optimal and sub-optimal installation."
- Origin Mode: `experiment` + `analysis`. No model code was changed.
- Origin Date: 2026-09-27
- Evidence:
  - `runs/nmcc_v25_signal_audit_states_20260921.pkl` — 16 calibrated-testbed
    episodes, 43 decision states, every feasible cell branched to the full
    horizon on 3 independent common-random-number tapes, with the cumulative
    return also recorded at +10, +20 and +30 timesteps along the same
    trajectories.
  - `runs/exposure_component_states_20260927.pkl` — new collection: the same
    16 episodes and the same 3 tape keys, at the first decision only, recording
    the **four signed reward components separately** at the same four
    truncations. Seeds 501–516 reproduce the audit's first-decision states, and
    the summed components reproduce the audit's scalar values to
    9.99e-16, so the two datasets compose exactly.
  - the live configuration: `NMCCPIConfig.py`, `NMCCPolicyImprovement.py`
    (`BranchValuer._continue`), and the six `config/*nmcc*.json` curricula.
- New experiments:
  - `exposure_signal_decomposition.py` — two-way ANOVA per decision state
    separating the state level, the cell effect, the tape effect and the
    pairing residual, at each truncation.
  - `exposure_label_estimators.py` — which one-tape label ranks cells best at
    equal simulation cost; control-variate blend of the cumulative windows.
  - `exposure_budget_allocation.py` — the truncation ceiling
    `corr(a_window, a_full)` estimated from tape-disjoint cross-products, and
    the resulting cells × tapes × horizon allocation frontier.
  - `exposure_component_signal.py` — which reward component carries the siting
    differential, and how it accumulates.
- Verification Status: `MECHANISM MEASURED ON THE CALIBRATED SYNTHETIC TESTBED
  ONLY; 16 EPISODES, 17 FEASIBLE CELLS ON AVERAGE AT THE FIRST DECISION`.
  - The variance components, truncation correlations and component shares are
    measured. The conversion from a correlation to an expected Spearman or
    top-1 rate uses a Gaussian surrogate over exchangeable cells, so those two
    columns are indicative rather than exact; the orderings they induce are
    robust to the surrogate because the underlying correlations differ by far
    more than the surrogate's error.
  - The full-horizon self-consistency check `corr(a_full, a_full) = 0.997`
    (first decision) and `0.999` (all decisions) validates the covariance
    estimator and its normalization.
  - No claim here is about State College; all numbers are the testbed's.

## Summary

The premise is right about the magnitude and already answered by the mechanism.
Installing anywhere rather than deferring one interval is worth `+0.2316` of
return; the choice among cells has a within-state standard deviation of
`0.0677`, so the capacity-driven common effect is `3.4x` the siting
differential. But the v26 label is not a comparison against an unconditional
baseline: it is a within-state contrast in which both arms install exactly one
identical token, from the same snapshot, under the same exogenous tape. The
common effect cancels identically, not statistically. The quantity that limits
the learner is what fails to cancel — the pairing residual, `0.0659` against a
cell effect of `0.0620` at the first decision, a signal-to-noise ratio of
`0.94`.

The hypothesis that the differential lives in hazard-weighted exposure time is
not supported. Hazard exposure carries `3%` of the total cell-effect variance
and has a *worse* signal-to-noise ratio (`0.85`) than the scalar return
(`0.94`). Active person-time is the cleanest single channel (`1.14`) and
safe completions the largest (`18%` of the variance), and `64%` of the total
signal variance sits in the covariance *between* components, so no single
component can stand in for the whole. Re-weighting all four by least squares
buys `+0.021` of Spearman — a `5%` relative gain with a *lower* top-1 rate.
There is no component-selection lever worth taking.

What the component collection does show is the mechanism behind the real
problem, and it is the "reduction speed" question in a sharper form. Between-cell
differences that exist early are *transient*: the safe-completion differential
peaks at +20 timesteps at `1.74x` its terminal magnitude and then erodes back
to `1.0` as slower cells catch up. So although `84%` of the total signal
*magnitude* has accumulated by +20, a noiseless +20 contrast correlates only
`0.598` with the full-horizon cell effect at the first decision. Truncation is
not a mild bias — it reorders the cells. The configuration default
`nmccPiBranchHorizon = 20` therefore caps the label's achievable rank agreement
at about `0.55` no matter how many tapes are added, and `BranchValuer._continue`
truncates with no tail term to absorb it.

The actionable consequence is an allocation error, not a modelling error. At the
first decision v26 branches every feasible cell on one tape to the full horizon:
about `1003` simulator timesteps, expected top-1 rate `0.359`, expected regret
`0.551` cell-effect standard deviations. The same budget spent as 9 cells on 2
tapes to the full horizon (`1062` steps) gives top-1 `0.429` and regret `0.416`
— a `25%` regret reduction at constant cost, from a one-line change to how the
branch budget is divided.

## 1. What cancels and what does not

Write the branch value at decision state `s`, candidate cell `c`, tape `k` as

    V(s, c, k) = mu(s) + a(s, c) + b(s, k) + e(s, c, k)

`mu(s)` is the state level: scenario severity plus "a shelter now exists at
all". `a(s, c)` is the siting signal. `b(s, k)` is the exogenous future common
to every cell at that state. `e` is the part of the future that does not cancel
between two cells that started from the same snapshot under the same tape.

Pooled over 43 decision states, at the full horizon:

| quantity | standard deviation |
| --- | --- |
| between-state level `mu` | 0.1711 |
| cell effect `a` | 0.0648 |
| tape effect `b` | 0.0641 |
| pairing residual `e` | 0.0476 |

The concern is about `mu`, and `mu` is exactly what a within-state contrast
removes — it is a single number per state, so it leaves the contrast
identically. Common random numbers then remove `b` exactly, because both arms
consume the same keyed noise tape from the same restored snapshot. That is worth
the cost of pairing: it lifts the signal-to-noise ratio from `0.81` unpaired to
`1.36` paired.

What remains is `e`, and `e` is the binding constraint. It arises because two
arms desynchronize: after a few timesteps the pedestrians have made different
routing choices, so the same keyed stream is consumed at different points. At
the first decision, where the v25 review located nearly all of the available
regret, the cell effect is `0.0620` and the pairing residual is `0.0659`.

So the answer to the premise is: the capacity effect is real and large, but it
is already removed exactly. The difficulty is not that every installation helps;
it is that the horizon is long, the system is stochastic, and two nearby sitings
diverge slowly.

## 2. Where the differential lives across the four components

New collection, 16 first-decision states, full horizon, three tapes, the four
signed reward components recorded separately:

| component | cell effect | pairing residual | SNR | share of total signal variance | common effect vs deferring | differential / common |
| --- | --- | --- | --- | --- | --- | --- |
| safe_completion | 0.0260 | 0.0292 | 0.89 | 0.18 | +0.0840 | 0.309 |
| casualty | 0.0093 | 0.0139 | 0.67 | 0.02 | +0.0168 | 0.555 |
| evacuation_time | 0.0222 | 0.0194 | 1.14 | 0.13 | +0.0969 | 0.229 |
| hazard_exposure | 0.0107 | 0.0126 | 0.85 | 0.03 | +0.0340 | 0.315 |
| **total** | **0.0620** | **0.0659** | **0.94** | **1.00** | **+0.2316** | **0.268** |

Three things follow.

The shares sum to `0.36`, so `64%` of the total signal variance is covariance
between components. A cell that completes more evacuations also drains active
person-time and hazard exposure faster; the components move together. That is
why no single component substitutes for the scalar: dropping the others drops
the covariance they contribute, not just their own variance.

Hazard exposure is the *smallest* carrier, at `3%` of the signal variance, and
its own signal-to-noise ratio is below the scalar's. The intuition is right in
direction — measured against the part of each component that every installation
buys, hazard exposure is more discriminating (`0.315`) than active person-time
(`0.229`) — but that ratio is not the operative constraint, because the common
part is removed exactly by pairing. Against the residual that actually survives,
hazard exposure loses.

Active person-time is the only component with a signal-to-noise ratio above one.
It is a smooth aggregate rather than a count of discrete events, so its pairing
residual is small relative to its differential. This is the component a
variance-reduction scheme would want to lean on, not hazard exposure.

### Ranking candidates on one component

One tape estimates, a held-out tape supplies the truth, and the truth is always
the **terminal** return — every estimator is judged as a surrogate for the
objective the controller is graded on. Mean over 16 states and all three
tape assignments:

| score | Spearman | top-1 | mean regret |
| --- | --- | --- | --- |
| safe_completion | 0.391 | 0.229 | 0.0663 |
| casualty | 0.192 | 0.167 | 0.1057 |
| evacuation_time | 0.448 | 0.229 | 0.0728 |
| hazard_exposure | 0.398 | 0.271 | 0.0558 |
| **total (the reward itself)** | **0.432** | **0.292** | **0.0565** |
| evacuation_time + hazard_exposure | 0.448 | 0.292 | 0.0643 |
| safe_completion + hazard_exposure | 0.410 | 0.208 | 0.0671 |
| least-squares re-weighting | 0.453 | 0.250 | 0.0714 |

The two time terms together edge the scalar on Spearman (`0.448` vs `0.432`) and
match it on top-1, and they do so at every truncation (`0.345` vs `0.337` at
+20, `0.416` vs `0.385` at +30). Hazard exposure alone is the best single
component at the shortest window (+10: `0.268` vs the scalar's `0.247`, top-1
`0.188` vs `0.125`), which is the part of the intuition that holds: the
hazard-weighted exposure differential is the fastest-emerging channel. But the
margins are `3–8%` relative, the mean regret does not improve, and the
least-squares re-weighting — which is free to find any single component if one
carried the signal — puts `+1.26` on active person-time and near-zero on
everything else, reaching only `0.453`. A `5%` relative gain, with a lower
top-1 rate, and it changes the target the label represents. This lever is not
worth taking.

## 3. Early differentials are transient, so truncation reorders cells

This is the "reduction speed" question in the form the data answers.

Share of the terminal magnitude accumulated by each offset, as signal / pairing
noise:

| component | +10 | +20 | +30 |
| --- | --- | --- | --- |
| safe_completion | 1.04 / 0.02 | 1.74 / 0.80 | 1.39 / 1.24 |
| casualty | 0.00 / 0.00 | 0.04 / 0.09 | 0.29 / 0.23 |
| evacuation_time | 0.08 / 0.00 | 0.38 / 0.10 | 0.58 / 0.33 |
| hazard_exposure | 0.04 / 0.04 | 0.19 / 0.14 | 0.37 / 0.32 |
| total | 0.46 / 0.01 | 0.85 / 0.39 | 0.78 / 0.69 |

The safe-completion row is the mechanism. Between-cell differences in
completions are *larger* at +20 than at the horizon — `1.74x` — and then erode
to `1.39x` and finally `1.0x`. A well-sited shelter absorbs its neighborhood
early; a poorly-sited one absorbs the same people later, by a longer route.
Early leads are repaid.

Magnitude share is therefore the wrong summary, and reading it as one was my
earlier error. What matters is the *correlation* between the truncated cell
effect and the terminal cell effect, and that is separately estimable: the
estimator uses one tape and the terminal proxy uses the other two, so their
residuals are independent and the cross-product is unbiased for
`cov(a_window, a_full)`.

First decision, 16 states, 17 feasible cells on average:

| window | steps per branch | cell effect | pairing residual | correlation with the terminal cell effect |
| --- | --- | --- | --- | --- |
| +10 | 10 | 0.0288 | 0.0008 | 0.371 |
| +20 | 20 | 0.0525 | 0.0257 | 0.598 |
| +30 | 30 | 0.0487 | 0.0452 | 0.813 |
| full | 59 | 0.0620 | 0.0659 | 0.997 |

Pooled over all 43 decision states the correlations are `0.595 / 0.752 / 0.898 /
0.999` — higher at later decisions only because less horizon remains to
truncate. The `0.997` and `0.999` self-checks validate the estimator.

A `0.598` ceiling means a +20 label, however many tapes it averages, cannot rank
cells better than Spearman `≈0.55`. That is the setting the configuration
default carries (`DEFAULT_NMCC_PI_BRANCH_HORIZON = 20`, which v25 used), and
`BranchValuer._continue` truncates bare — it returns `clock.accumulated - start`
with no `gamma^k V(s_{+k})` tail. The v26 curricula's choice of `0`, the full
horizon, is the right one; the default is not.

## 4. How the branch budget should be divided

A branch costs about `tapes x window` simulator timesteps per cell, and a
decision costs `cells x tapes x window`. All three are configurable
(`nmccPiTapes`, `nmccPiBranchHorizon`, `nmccPiMaxBranches` /
`nmccPiExhaustiveDecisions`), and all three trade against each other at constant
cost. Expected Spearman against the terminal cell effect, first decision:

| window | 1 tape | 2 tapes | 3 tapes | 5 tapes | 8 tapes |
| --- | --- | --- | --- | --- | --- |
| +10 | 0.335 @ 10 | 0.341 @ 20 | 0.335 @ 30 | 0.337 @ 50 | 0.339 @ 80 |
| +20 | 0.495 @ 20 | 0.520 @ 40 | 0.531 @ 60 | 0.539 @ 100 | 0.545 @ 160 |
| +30 | 0.549 @ 30 | 0.632 @ 60 | 0.667 @ 90 | 0.703 @ 150 | 0.725 @ 240 |
| full | 0.636 @ 59 | 0.751 @ 118 | 0.805 @ 177 | 0.863 @ 295 | 0.900 @ 472 |

Horizon first, then tapes. At equal cost, `+30 x 2` (`0.632`) beats `+20 x 3`
(`0.531`), and `full x 1` at 59 steps matches `+30 x 2` at 60. Past the full
horizon every further step should buy a tape: the full-horizon row is the cost
frontier at every budget above 59 steps per branch. Adding tapes to a truncated
window is close to wasted — `+20` gains `0.050` of Spearman going from 1 tape to
8, because the ceiling, not the noise, is binding.

The third axis is how many cells to branch, and it has a ceiling of its own. The
v25 prior correlates `0.338` with the terminal cell effect, and the truly best
cell sits at prior rank `7.7` of `17` on average: it is inside the prior's top 6
only `44%` of the time and its top 8 only `56%`. So `nmccPiMaxBranches = 6` caps
the attainable top-1 rate at `0.44` whatever the label's quality.

Putting the three together, at a fixed cost per decision:

| budget | best allocation | top-1 | regret (cell-effect SD) |
| --- | --- | --- | --- |
| ~500 steps | 9 cells x 1 tape x full | 0.335 | 0.582 |
| ~1000 steps | 9 cells x 2 tapes x full | 0.429 | 0.416 |
| ~1500 steps | 9 cells x 3 tapes x full | 0.474 | 0.336 |
| ~2000 steps | 12 cells x 3 tapes x full | 0.501 | 0.300 |
| ~3000 steps | 10 cells x 5 tapes x full | 0.543 | 0.242 |

And what the current settings buy:

| setting | cost per decision | top-1 | regret |
| --- | --- | --- | --- |
| v26 first decision: all 17 cells x 1 tape x full | 1003 | 0.359 | 0.551 |
| v25: 6 cells x 1 tape x +20 | 120 | 0.257 | 0.805 |
| **same v26 budget, 9 cells x 2 tapes x full** | **1062** | **0.429** | **0.416** |
| same budget, 6 cells x 3 tapes x full | 1062 | 0.409 | 0.434 |

Exhaustive branching on a single tape is dominated. Spending a quarter of the
cells on a second tape cuts expected regret by `25%` at the same simulator cost,
and the interior optimum is `9–12` cells rather than all of them or six.

## 5. Why this is the convergence problem

The project's difficulty is stated as the impossibility of separating system
self-evolution from the decision-maker's effect. The measurements locate that
precisely: with one tape and the full horizon the label's rank agreement with
the truth it is meant to represent is `0.73` pooled over decisions and `0.64` at
the first decision, where the regret is. The label, not the network and not the
credit-assignment machinery downstream of it, is where the information is lost.
No refit schedule, deployment gate or ensemble penalty can recover ordering that
the target never contained; a gate that asks whether the fitted model beats the
base policy *on that label* is measuring agreement with a `0.64`-quality
reference.

That also explains a pattern in the earlier audits. The `value_lcb` gate opens
on three consecutive positive lower-confidence gains, and a model of moderate
skill produces a positive statistic often enough to pass; the statistic is
computed against labels whose own agreement with the truth is `0.64`, so the
gate's discrimination is bounded by the label's, independent of `z` or the
number of consecutive updates.

## 6. Recommended changes

Ordered by measured effect per unit of work. These are design recommendations;
no model code was changed for this document.

1. **Re-divide the branch budget: fewer cells, more tapes, full horizon.**
   Replace exhaustive first-decision branching with the prior's top `9–12`
   cells (the executed and base actions always included, as now) and set
   `nmccPiTapes = 2` at the exhaustive decisions. Cost-neutral against the
   present configuration; expected regret `0.551 -> 0.416` cell-effect standard
   deviations. If the budget can grow, `3` tapes at `9–12` cells reaches
   `0.336`.
2. **Never truncate a branch bare.** Either keep `nmccPiBranchHorizon = 0`, or,
   if truncation is needed for cost, add the bootstrap term `gamma^k V(s_{+k})`
   from the value head to `BranchValuer._continue`. The missing `40%` of the
   ordering at +20 is exactly what the critic is there to supply; discarding it
   is the one choice with no upside. Change the configuration default from `20`
   to `0` so a curriculum that omits the key is not silently capped at
   Spearman `0.55`.
3. **Raise `nmccPiMaxBranches` from 6 to at least 9** at later decisions, or
   accept that the top-1 rate there cannot exceed `0.44`. Six candidate cells
   was calibrated against cost, not against the prior's recall.
4. **Improve the candidate prior before improving the label.** Its correlation
   with the terminal cell effect is `0.338`; every subset scheme inherits that
   as a ceiling. A prior at `0.6` would let a 6-cell subset reach the recall a
   12-cell subset reaches today, halving the branch cost.
5. **Do not pursue component re-weighting or a single-component label.**
   Measured, negative: `+0.021` Spearman at best, with a lower top-1 rate and a
   changed target. The reward's own weights already place the signal correctly.
6. **Report the label's rank quality as a first-class training diagnostic.** One
   extra tape at a small sample of decision states per epoch yields a direct
   estimate of the label's Spearman against a held-out tape. Without it, a run
   cannot distinguish a learner that is failing to fit from a target that cannot
   be fitted — which is the ambiguity this whole investigation had to resolve
   offline.

## Artifacts

- `exposure_signal_decomposition.py` → `runs/exposure_signal_decomposition_20260927.json`
- `exposure_label_estimators.py` → `runs/exposure_label_estimators_20260927.json`
- `exposure_budget_allocation.py` → `runs/exposure_budget_allocation_first_20260927.json`,
  `runs/exposure_budget_allocation_all_20260927.json`
- `exposure_component_signal.py` → `runs/exposure_component_states_20260927.pkl`,
  `runs/exposure_component_signal_20260927.json`
