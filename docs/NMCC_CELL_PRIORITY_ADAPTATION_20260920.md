# NMCC adapted to the cell-priority action space: Stage 0, Variant A, and the measured causal signal

> Superseded on 2026-09-20 by
> `docs/HYBRID_NMCC_IMPLEMENTATION_AND_BACKTEST_20260920.md`. This document is
> retained as the Stage-0 design and estimator record; its learner-side
> implementation status no longer describes model version 21.

## Material Passport

- Origin: Claude (Cowork) implementation and experiment, made directly in this
  repository at the user's explicit instruction
- Origin Mode: `code` + `experiment` — `RLBridge.py` was modified;
  `CounterfactualBranch.py`, `nmcc_testbed.py`, `nmcc_paired_experiment.py`
  and two test modules were added; the paired-effect experiment was executed
  and its report committed to `runs/nmcc_paired_report.json`
- Origin Date: 2026-09-20
- Verification Status: `SIMULATOR LAYER VERIFIED; LEARNER LAYER UNVERIFIED`.
  Every claim in sections 2–4 was produced by executing the real dynamics
  code. Section 5 (the `RLBridge` integration) could not be executed here:
  torch is not installable in either environment reachable from this tool
  (PyPI and `download.pytorch.org` are both outside the account's egress
  allowlist), so the then-current `tests/test_nmcc_integration.py` (retired in
  v27) and the backtest had to be
  run in the pinned `rlevacuation` environment before the integration is
  treated as working.
- Version Label: `nmcc_cell_priority_variant_a_v1`

## 1. The question this answers

The brief was: the brute-force approach would be to evolve the system with and
without a decision over the same `n` timesteps and difference the two, but the
system is stochastic and that would be expensive.

Both concerns are correct, and both have precise answers.

**On stochasticity.** The paired comparison does not require the system to be
deterministic or predictable. It requires the two branches to experience the
*same* disturbances. Writing `G(a, U)` for the return after action `a` under
disturbance realization `U`,

    Var[G(a, U) - G(0, U)] = Var[G(a)] + Var[G(0)] - 2 Cov[G(a), G(0)],

so when both branches ride the same fire, the same person-level casualty
shocks and the same panic draws, the covariance approaches the variances and
the difference variance collapses. The stochasticity is not removed; it is
shared, and then it cancels. This is common random numbers (Kleinman, Spall
and Naiman 1999) used as a control variate, and it is exactly what NMCC's
structural noise tape is for.

**On cost.** The expensive version is the one that re-runs whole episodes.
Three properties reduce it to roughly a factor of two:

1. *Bounded branch horizon.* The counterfactual runs for `L` timesteps, not to
   the end of the episode; the critic closes the remaining gap through
   `gamma^L (V(s_L^a) - V(s_L^0))`. `L` is a bias/variance knob.
2. *One baseline serves every action.* `WAIT` does not depend on which cell
   was chosen, so a single no-deployment branch is a valid control variate for
   all `number_of_cells` actions at once. Cost is `2x`, not `|C|x`. This is the
   property that makes the scheme tractable at full grid resolution, and it is
   a direct benefit of the cell-priority action space: there are far fewer
   distinct actions to reason about than under the retired exact-candidate
   design.
3. *The factual branch is already being simulated.* Setting `L` to the
   deployment interval makes the acted branch coincide with the trajectory the
   episode lives anyway, so only the `WAIT` branch is extra work.

Measured overhead for the snapshot/restore machinery, on the real dynamics:

| population | capture | restore | one timestep | branch (L=5) | overhead vs. simulating L steps |
|---|---|---|---|---|---|
| 200 | 10.7 ms | 10.7 ms | 25.6 ms | 151 ms | 1.18x |
| 800 | 34.8 ms | 36.0 ms | 73.1 ms | 471 ms | 1.29x |
| 3000 | 120 ms | 147 ms | 288 ms | 1672 ms | 1.16x |

Snapshot cost is linear in population (~40 us per pedestrian) because the road
graph and its routing caches are shared rather than copied. A training episode
with counterfactual credit therefore costs about `2.2x` a baseline episode.

Set against the measured variance reduction, that is a large net win: the
number of samples needed to resolve an effect to a given precision scales with
its variance, so a 99% variance reduction buys roughly two orders of magnitude
in sample efficiency for the per-decision effect estimate, against a 2.2x cost
per episode.

## 2. What was missing, and what is now in place

The simulator was already most of the way to a valid counterfactual, which is
worth stating plainly because it changes the size of the job:

- `PedestrianDatabase._hazard_uniform`, `_panic_uniform` and
  `_panic_susceptibility_uniform` are already counter-based splitmix64 draws
  keyed by immutable identifiers (episode seed, mechanism, pedestrian id,
  timestep, channel). They are invariant to iteration order and to how many
  people another branch has already evacuated. `U^cas`, `U^panic` and `U^move`
  therefore needed no work at all.
- Hazard evolution consumes a sequential `numpy` generator, but
  `HazardDatabase._spreadUpdateStochastic` reads only hazard state and cell
  states — never shelters, never pedestrians. Hazard is exogenous, so
  capturing and restoring the bit-generator state reproduces `U^H` exactly.
  This is asserted on the live object graph by
  `assert_hazard_is_action_independent` rather than assumed.

What did not exist: snapshot/restore, the branch runner, and the estimator.
`CounterfactualBranch.py` adds them.

The snapshot deliberately splits the world in two. Branch-varying state
(`pedDS`, `hazardDS`, `shelterDS`, `cellTracker`, `forceTracker`) is deep-copied
through a single shared memo, which preserves every internal alias — a cell's
reference to a shelter still points at *that branch's* shelter. Static
infrastructure (the graph, every `Node` and `Edge`, and the routing-tree caches,
which are pure functions of the map and a target) is shared, because copying it
would dominate the cost and a cache entry computed in one branch is exactly the
entry the other would have computed. The mutable flow fields that happen to
live on those shared node and edge objects (`nodeFlow`, `edgeFlow`, congestion
occupancy, density and speed ratio) are captured separately as flat arrays.

## 3. Stage-0 validity, verified

NMCC calls these prerequisites rather than niceties, because training on
invalid twins manufactures high-confidence false causal labels — a worse
failure than the noisy gradients it set out to fix. All are executed in
`tests/test_counterfactual_branch.py` against the real dynamics (10 tests,
4.8 s):

- **Factual replay from a restored snapshot is bitwise identical.** Any mutable
  state escaping the snapshot would show up here.
- **A snapshot is reusable.** `restore` copies on the way out as well as in, so
  the first branch cannot mutate the stored state and silently invalidate every
  later comparison.
- **A branch does not disturb the live episode.** The state digest and shelter
  count are unchanged after a paired branch runs.
- **Branch order does not change either branch.** Running `WAIT` first and the
  deployment second reproduces the opposite order exactly.
- **Hazard is action-independent.** Installing a shelter leaves the fire front
  identical.
- **The fast outcome read matches the authoritative one.** `outcome_snapshot`
  skips the `O(cells x pedestrians)` candidate feature block; it is pinned
  against `RegionalObservationBuilder` step by step so it cannot drift.
- **The `WAIT` baseline is constant across cells.** This is what keeps the
  policy gradient unbiased: a baseline that varied with the chosen cell would
  bias the update rather than de-noise it.

## 4. The measured signal

`nmcc_paired_experiment.py` is NMCC's minimal first experiment. At each
decision epoch it snapshots the simulator and records `A[r, c]`, the return
over the branch horizon after deploying in cell `c` under noise tape `r`, and
`W[r]`, the return with no deployment under the same tape. Two estimators are
then formed from *exactly the same simulations*, so the comparison isolates the
pairing and nothing else:

    paired:    D[r, c] = A[r, c] - W[r]
    unpaired:  A[r, c]        (PPO's own baseline V(s) is constant in the action,
                               so it shifts the estimate without changing spread)

Representative run (6x6 grid, 400 pedestrians, 8 noise tapes, 6 cells per
epoch, L=6, five decision epochs; full report in
`runs/nmcc_paired_report.json`):

| quantity | value |
|---|---|
| between-cell signal SD (how much the choice matters) | 0.0167 |
| paired estimator SD | 0.00064 |
| unpaired estimator SD | 0.0080 |
| variance reduction | **99.36%** |
| signal-to-noise, paired | **54.4** |
| signal-to-noise, unpaired | 2.19 |
| single-sample cell-ranking recovery, paired | **+0.984** |
| single-sample cell-ranking recovery, unpaired | +0.457 |
| mean effect shift between estimators | z = 0.00 |

The last two rows are the ones that answer "can the agent pick up useful
signal". Variance reduction on a scalar is not the same as a usable learning
signal, so the harness also asks the decision-relevant question directly:
on-policy training does not get many matched samples of a state — it sees the
state once, takes one action, and lives one noise realization. So each cell is
scored under *its own* tape and the resulting ordering is compared against the
tape-averaged truth. The paired estimator recovers the true ordering almost
perfectly (Spearman +0.984); the unpaired estimator, carrying whatever the fire
happened to do on its draw, lands near +0.457.

The mean effect is unchanged between the two estimators (z = 0.00). Pairing
reduced variance without buying it with bias, which is the second gate.

For scale: the 2026-09-19 validation measured a per-episode total-return SD of
0.2226 against a held-out RL-minus-heuristic difference of -0.00596 — the
quantity of interest was roughly forty times smaller than the noise it was
embedded in, which is why the policy converged to agreement with its own prior
rather than to an improvement over it. The paired estimator's noise floor is
0.00064, which resolves an effect of that size comfortably.

Two honest caveats. First, these numbers come from the synthetic grid city in
`nmcc_testbed.py`, not from State College; the mechanisms are the real ones but
the map is not, and the experiment should be re-run on a real map before the
numbers are quoted in the paper. Second, some late-episode epochs show a true
effect of exactly zero (every remaining cell equivalent because the evacuation
has stalled beyond the branch horizon). The paired estimator correctly reports
zero there while the unpaired estimator reports pure noise — which is itself an
illustration of the failure mode, but it does inflate the pooled variance-
reduction figure. Per-epoch numbers are in the report.

## 5. The learner-side integration (Variant A) — written, not yet executed

NMCC's Variant A is exact paired PPO: for each sampled action, run one matched
`WAIT` branch and use the exact counterfactual advantage. NMCC is explicit that
Variant C (the full hybrid with a learned residual world model) "should be
attempted only after Variant A demonstrates a material reduction in advantage
variance". Section 4 is that demonstration, so Variant A is what was
implemented and the learned residual and dueling causal critic are deliberately
*not* built yet.

`RLBridge` gains four constructor options, all inert by default:

    counterfactual_credit=False          # master switch
    counterfactual_horizon=None          # defaults to the deployment interval
    counterfactual_weight=1.0            # 1.0 replaces GAE where a branch exists
    counterfactual_intervention_cost=0.0 # c(a)

When enabled, at each decision epoch and *before* the executor installs
anything, `_collect_wait_baseline` snapshots the simulator, runs the
no-deployment branch for `L` steps, scores it, and restores. When the pending
decision closes,

    A_CF = (R_a - R_0) + gamma^L (V(s_L^a) - V(s_L^0)) - c(a)

is attached to the transition. Both bootstrap values are read with the
*decision epoch's* recurrent state, since both branches share history up to
that point; conditioning them on different pasts would make them
incomparable. `_blend_counterfactual_advantages` then standardizes each
estimator separately before combining them — they are on different scales by
construction, and combining them raw would let whichever is larger dominate the
step size for reasons unrelated to which is more informative.

With the flag clear, every one of these paths is guarded and training is
bit-for-bit what it was. `MODEL_VERSION` is deliberately **not** bumped: the
network architecture is unchanged, so existing checkpoints stay loadable.

**Historical note.** This v21 hybrid counterfactual-credit path was retired in
v27 together with its dedicated integration test. Current exact branching and
fitted-value control are covered by `tests/test_nmcc_policy_improvement.py` and
`tests/test_nmcc_pi_torch.py`.

## 6. What to run next, in order

1. `python -m unittest discover -s tests -v` in the `rlevacuation` environment.
   This covers the previously unverified cell-priority change as well as the
   new modules.
2. `python nmcc_paired_experiment.py --output runs/nmcc_paired_synthetic.json`
   to reproduce section 4 locally, then the same measurement against a real
   `Core` on State College. The numbers in section 4 should be replaced with
   the real-map numbers before they are used in the paper.
3. A short training run with `counterfactual_credit=True` against a matched run
   with it off, on identical scenario seeds. The quantity to watch is not the
   reward curve first but `nmcc_counterfactual_advantage_sd` against
   `nmcc_gae_advantage_sd` in the diagnostics: the advantage variance is what
   this change targets, and it should drop by the order of magnitude section 4
   predicts. If it does not, the integration is wrong and the reward curve
   would not tell you why.
4. Only then the paired backtest against the heuristic benchmarks.

## 7. What is deliberately not built

- **The learned residual world model and dueling causal critic** (NMCC
  Variants B and C). These are Stage 2/3 work and depend on a paired dataset
  that does not exist yet. Building them before Variant A is confirmed in a
  real environment would be speculative, and NMCC's own staging says not to.
- **`WAIT` as a policy action.** The counterfactual baseline needs `WAIT` as a
  *branch*, not as an action the policy may select, and the branch is enough
  for the entire variance reduction. Adding `WAIT` to the action space is a
  separate change that interacts with the capacity-token P0 item, and it was
  kept out of scope so this change stays reviewable.
- **Keyed rewriting of the hazard generator.** Snapshot/restore of the
  generator state is sufficient given the verified exogeneity of hazard
  evolution. Rewriting it into keyed form would be belt-and-braces hardening
  against a future change that made hazard read crowd state; if that ever
  happens, `assert_hazard_is_action_independent` is the test that will catch
  it.
