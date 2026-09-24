# Companion note to Natural-Momentum Counterfactual Control: retrospective evidence, a formal invariance cross-reference, and a Stage-0.5 interim baseline

## Material Passport

- Origin: Claude (Cowork) design consultation, external to the `academic-research-suite/experiment-agent` pipeline
- Origin Mode: `plan` only — no code, config, or run artifacts were changed to produce this document
- Origin Date: 2026-09-20
- Verification Status: `UNVERIFIED` (companion analysis; every quantitative claim below is either (a) recomputed directly from numbers already published in `docs/CREDIT_ASSIGNMENT_VALIDATION_RESULTS_20260919.md`, or (b) a literature-grounded inference, clearly marked)
- Version Label: `nmcc_companion_note_v1`

## 0. Why this document exists, and what changed while writing it

This analysis was drafted independently, starting from the same evidence base as everyone else working on this project: `docs/MDP_AND_OPTIMIZATION_DESIGN.md`, `docs/CREDIT_ASSIGNMENT_VALIDATION_PROTOCOL_20260919.md`, `docs/CREDIT_ASSIGNMENT_VALIDATION_RESULTS_20260919.md`, and `docs/RL_SIGNAL_CURRICULUM_PROPOSAL_20260919.md`. Its first draft proposed a CRN-coupled candidate-versus-reference-action difference reward as a fundamental fix for the same problem the user's brief describes as being "too hard to differentiate global variation (system self-evolution) with variation caused by actions."

While preparing to commit that draft, `docs/NATURAL_MOMENTUM_COUNTERFACTUAL_CONTROL_20260920.md` (NMCC) was found already registered in this repository, dated the same day, as a `P0` blocker in `TODO.md`. NMCC is a substantially more complete treatment of exactly this problem: a structural causal decomposition (`H`, `C`, `P`, `M` state blocks), a formal natural-momentum/intervention-residual factorization, a keyed structural noise tape, a dueling causal critic `Q = V_wait + D`, counterfactual-advantage PPO, explicit identifiability conditions, and a staged `N0`–`N8` ablation ladder with concrete promotion gates (including a variance-ratio gate that is precisely the right quantity to be tracking). It is well grounded in the literature it cites (Dietterich, Trimponias & Chen on exogenous-state MDPs; Mesnard et al. on hindsight/counterfactual credit assignment; the COCOA paper; Kleinman, Spall & Naiman on common random numbers; Hafner et al. on world models).

This document does not restate or compete with NMCC. NMCC should remain the registered plan. What follows is three things NMCC's own document does not already contain, offered as a companion note:

1. quantitative evidence, computable **today** from data already in `runs/`, that the causal-identification premise NMCC is built on is very likely to hold in this simulator — a form of low-cost pre-registration confidence before investing in the Stage-0 build;
2. a formal cross-reference that gives NMCC's dueling-critic invariance argument (currently justified only for the static-argmax case) a general multi-step proof from the classical reward-shaping literature;
3. a "Stage-0.5" interim step — pretraining `V_wait` from independent (unpaired) WAIT-continuation rollouts — that can start the moment the `WAIT` action itself lands, without waiting for the harder snapshot/restore and structural-noise-tape items later in NMCC's own Stage 0 checklist.

## 1. Retrospective evidence for NMCC's central hypothesis, from data already in hand

NMCC's promotion gate for its minimal first experiment (Section "Minimal first experiment") is that common-noise paired candidate-versus-`WAIT` effects show materially lower variance than independent-noise pairs, formalized as the variance ratio

```
Var[G(a,U) - G(WAIT,U)] / Var[G(a,U)]
```

with an initial bar of at least a 50% reduction (Section "Primary metrics and promotion gates").

This project already has a data point for a closely related quantity, computed without running any new experiment. The 2026-09-19 credit-assignment validation reports two numbers that bracket this ratio:

| Quantity | Value | Source |
|---|---:|---|
| `Var[G(single policy, U)]`, proxied by per-episode total-return SD across raw unpaired training rollouts | `0.2226` SD → `Var ≈ 0.0496` | "Reward components" table |
| `Var[G(RL,U) - G(heuristic,U)]`, proxied by the held-out RL-minus-heuristic return-difference SD under matched (CRN-paired) hazard/pedestrian trajectories | SD ≈ `0.028`–`0.048` → `Var ≈ 0.0008`–`0.0023`, back-calculated from the reported 95% CI `[-0.02746, 0.01076]` | "Held-out paired backtest" table |

The implied variance ratio is roughly `0.0008–0.0023 / 0.0496 ≈ 1.6%–4.6%`, i.e. a **95%+ reduction** — well past NMCC's 50% bar. This is not the same estimand NMCC's `N2` will measure (this compares two different *policies* — RL and the heuristic — under shared noise, not one *policy's own action* against `WAIT` under shared noise; the held-out comparison also uses only 8 scenario seeds × 3 checkpoints, and is at the episode level rather than the single-decision level NMCC's minimal experiment targets). It should therefore be read as prior evidence in the same direction, not as a substitute for running NMCC's own 200-snapshot experiment — but it is a meaningfully large, already-observed effect, in exactly the mechanism NMCC's premise depends on (shared exogenous randomness dominates unpaired variance; matching it removes most of that variance). It is worth recording in whichever report documents NMCC's Stage-0 results, as a reason the team had non-trivial prior confidence the minimal experiment would pass before running it.

## 2. A general invariance proof for the dueling causal critic

NMCC justifies that subtracting the same `WAIT` baseline from every action's value does not change the optimal action, via the algebraic identity

```
arg max_a Q(b_t, a) = arg max_a [Q(b_t, a) - Q(b_t, WAIT)]
```

This is correct but is a single-decision (static argmax) argument. The stronger and more directly relevant guarantee — that using `V_wait` as a *running* baseline inside a multi-step, discounted or undiscounted return, exactly as `A_CF` in NMCC's "Counterfactual-advantage PPO" section does, leaves the *entire optimal policy* unchanged, not just the ranking at one decision — is a known, previously proven result: potential-based reward shaping (Ng, Harada & Russell, *Policy invariance under reward transformations: Theory and application to reward shaping*, ICML 1999). Their theorem states that for any potential function `Φ(s)`, transforming the reward as

```
r'(s, a, s') = r(s, a, s') + γΦ(s') − Φ(s)
```

leaves the optimal policy of the MDP unchanged, for *any* `Φ`, including one estimated from data. Setting `Φ = V_wait` (NMCC's natural-momentum value head) makes NMCC's causal-advantage construction a direct instance of this theorem, extended across the whole trajectory rather than one decision. This is worth adding as a citation in NMCC's own "Identifiability and failure conditions" or "Why this is stronger than reward shaping" section: it converts "we believe subtracting a shared baseline is fine" into "this is a proven special case of an established invariance theorem," which is a stronger claim to make in whatever confirmatory writeup eventually reports this work, and it correctly frames NMCC's real departure from classical shaping — the *causal, CRN-identified* estimation of `Φ = V_wait` and of the residual `D`, rather than the invariance property itself, which is not new.

## 3. A Stage-0.5 interim step: pretrain `V_wait` from independent WAIT-continuation rollouts

NMCC's Stage 0 bundles several prerequisites together: the `WAIT` action and complete interval accounting, equal shelter-capacity tokens, the keyed structural noise tape, full simulator snapshot/restore, and a proof that a restored-and-replayed factual trajectory is bitwise identical to the original. All of that is required before any *paired* branch (`N2` and beyond) can be trusted. It is not, however, required to get a first, useful, if noisier, estimate of `V_wait(b_t)` — the expected future outcome vector if no further shelter is installed from state `b_t` onward.

`V_wait` can be estimated by ordinary independent Monte Carlo: once the `WAIT` action exists (already scoped as the very first Stage-0/`E1` item in both `RL_SIGNAL_CURRICULUM_PROPOSAL_20260919.md` and NMCC), run many episodes under the trivial "always `WAIT`" policy from randomized initial configurations — city, population, hazard count, and grid resolution — exactly as NMCC's own Stage 1 already recommends randomizing for `M0`'s coverage, and fit a regression against the same four-component outcome vector the existing critic already predicts. This step needs **no** noise-tape keying and **no** snapshot/restore, because it never compares two branches from the same state — it only needs many independent samples of one fixed, non-learning policy, which is the easiest possible object to estimate well: no nonstationarity, no small-batch casualty-head instability, and stratifiable by hazard dose at whatever sample size is affordable, exactly as `docs/CREDIT_ASSIGNMENT_VALIDATION_RESULTS_20260919.md`'s own "next experiment" list already calls for (item 1: "balance training scenario generation over prespecified hazard-dose/risk strata").

Once available, `V_wait` can be plugged into the *current* recurrent PPO's `_component_gae_targets` in `RLBridge.py` as classical potential-based shaping (Section 2) — replacing or blending with the on-policy learned critic as the GAE baseline — while the harder parts of Stage 0 (noise-tape keying, snapshot/restore, bitwise-replay proof) are still being built and verified. This is explicitly a stopgap, not a substitute for NMCC's eventual `M0`: NMCC's natural-momentum model is distributional, multi-horizon, and conditioned on the full predicted hazard trajectory, which this simple regression is not. Its only purpose is to test, cheaply and early, whether a better (lower-variance, stationary) baseline alone moves the needle on the specific failure named in the 09-19 results — the casualty-head's late-training instability under a nonstationary, small-batch target — before committing to the full `N3`–`N6` build. If it does not help, that is itself informative: it would suggest the dominant problem is closer to `N2`'s realized-noise term than to `N0`'s baseline-quality term, sharpening where the heavier engineering effort should go first.

Suggested placement on the existing ladder: between `N0` and `N1`, as `N0.5`, gated the same way as the others (held-out return, casualty tail, and — the metric most relevant here — component critic explained variance and casualty-head loss stability across rollout blocks, directly comparable to the numbers already reported for `N0` in `CREDIT_ASSIGNMENT_VALIDATION_RESULTS_20260919.md`).

## 4. A sequencing note across the three registered `P0` items

`TODO.md`'s `P0` section currently lists three items — equal-capacity-token enforcement, NMCC Stage 0 validation, and the staged RL signal curriculum — as separate bullets. Reading them together, they share literally overlapping engineering: NMCC's Stage 0 explicitly requires "implement equal shelter-capacity tokens" and "add `WAIT` and complete interval accounting," and the curriculum proposal's `E1` is "`WAIT`, intervention cost, and hard physical masks," with its own equal-capacity-token contract as a documented prerequisite. Implementing these as one coordinated engineering slice — one `WAIT` action, one capacity-token contract, one set of interval-accounting tests — rather than three independently-branched efforts would avoid the realistic risk of two slightly different `WAIT`/capacity-token implementations landing in different work and needing reconciliation later. This is purely a project-management observation, not a technical one, but worth stating given how much shared surface area the three `P0` items have.

## 5. Appendix — how initial shelters are currently selected (answering the Policy-1 sub-question directly)

Two distinct mechanisms exist in the code and should not be conflated:

1. **The common baseline shelters present at `t=0` for every policy and benchmark** (`ShelterDatabase.initShelter()`): a pure round-robin sweep over grid cells in row-major order, taking whichever candidate is first in that cell's OSM-derived candidate list (`candidates.pop(0)`) until `initVol` shelters are placed. No capacity, demand, or hazard criterion is used at this stage — it is a spatial placement rule only, and its within-cell tie-break is whatever order the OSM candidate extraction produced, not an explicit ranking.
2. **The lower-level site-selection rule used every time any policy (RL or any heuristic) opens a *new* shelter in a chosen region** (`ShelterDatabase._candidate_index()` / `newShelter()`): once a region is chosen, the implementation deterministically installs the **maximum-capacity** feasible candidate in that region, tie-broken by OSM identifier. This rule is shared identically by RL and every benchmark, so it is not a source of RL-vs-heuristic asymmetry.
3. **The `initial_only` static benchmark policy** (`docs/BENCHMARK_MODEL_PROTOCOL.md`) is a third, separate thing: it predeploys the entire *dynamic* deployment budget at time zero, also in "deterministic row-major candidate order," specifically as a non-optimized anticipative-timing diagnostic — the document explicitly notes that `static_greedy` (the demand-aware static comparator) is the substantive static OR baseline, not `initial_only`.

Because rule (1) is common to every policy and every benchmark, it does not bias the RL-vs-heuristic comparison. It is worth flagging as a separate, lower-priority finding: since the common `t=0` baseline is placed without regard to demand or hazard exposure, every episode may start from an already-suboptimal coverage configuration, which could inflate early casualties/exposure in a way that is common across arms (not confounding the comparison) but adds to the total outcome variance every policy — and every NMCC branch — must contend with. If reducing that shared variance floor is ever desired, `predeployStaticDemandGreedy()` (already implemented, currently used only for the `static_greedy` benchmark's own predeployment) could be reused for the common baseline as a separate, clearly-labeled experiment — out of scope here, noted only because it surfaced while answering the question.

## 6. Scope and honesty notes

- No code, configuration, checkpoint, or run artifact was modified to produce this document.
- Section 1's variance-ratio estimate mixes two related but distinct estimands (RL-vs-heuristic under shared noise, versus NMCC's own action-vs-`WAIT` under shared noise) and a small sample (8 held-out scenarios × 3 checkpoints); it is supportive prior evidence, not a substitute for NMCC's own registered minimal experiment.
- Section 2's invariance claim is exact only for the tabular/expected-value case proved by Ng, Harada & Russell; with a function-approximated, finite-sample `V_wait`, the practical benefit is a bias/variance trade-off that must still be verified empirically, exactly as NMCC's own promotion gates already require.
- Section 3's `V_wait` pretraining is explicitly a stopgap for the period before NMCC's full Stage 0/1 is built and verified, not a claim that it is an adequate substitute for `M0`.
- This note takes NMCC's structural framework, PPO, the recurrent-GNN architecture, and the four-component critic decomposition as given; it does not re-argue for or against any of them.
