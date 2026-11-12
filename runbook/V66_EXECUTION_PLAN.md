# EvacuationRL v66 execution and result generation plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan, with implementation verification
- Origin Date: 2026-11-12, America/Los_Angeles
- Verification Status: implementation and preflight verified; scientific gates pending
- Version Label: v66_validity_plan_1

## Current state and scope

The requested code changes are applied only in /Users/huali/Desktop/RLEvacuationCollab-v66, on branch codex/v66-validity based on f3edfb2e3f8ce23b44066878e77fbc75c8731c85. The reviewed code is frozen under tag v66.0-validity; specs/curricula are recorded in the following setup commit. The original /Users/huali/Desktop/RLEvacuationCollab checkout keeps its frozen source and data; only its LOG.md and TODO.md receive measured status entries.

Execution was authorized on 2026-11-12: start G0 identity and smoke, then continue under the manual gates, with 20-minute measured progress updates. Local unit tests, compilation, spec generation, and preflight planning are complete. Runtime state is recorded separately under runs/_logs and the v66 results STATUS.md. The legacy v65 PPO shard remains preserved at1190/2000 and is not relaunched during v66 evaluation.

Source: RUN_MANUAL_V66_VALIDITY_20261112_1.pdf, all 21 pages, byte-identical to the package manual. SHA256 a731cfde879c06a80f91b2608f5e8e58271bac1c7ebb6c8390c61d065d7a3cae. All 63 package-manifest entries verified. Site-loss revision 4 patch SHA256 433dd4e69edbaa33ad671ab28b83eec0ff34fbbfc3affaa01ea81963582adc88. V66 patch SHA256 56178e23372d85c8aa35666bf54a8c015fa1f1c3cf9b86bb4046bb9e38374eb2.

## Implemented changes and validation

| Change | Behavior |
|---|---|
| Candidate pool | Optional deterministic spatial_maximin ordering; legacy_row_major default retained |
| Initial shelters | Optional demand_greedy placement on planned population; legacy default retained |
| Planner shortlist | Optional enriched observation at planner decision opportunities; raw default retained |
| Defer prior | Optional below_sites mode propagated through config, bridge, and actor; zero default retained |
| Departure delay | Dedicated counter-based Weibull stream, waiting/movement filtering, and wait-aware safety/arrival calculations; mean zero default retained |
| Frozen checkpoints | Explicit --allow-dynamics-change records changed source files, changes episode identity, and guards the clone's paper_v65 campaign |
| Curricula | Six scenario keys plus actorPriorDeferMode accepted for Path B |

Applied the supplied patches without ad hoc source edits. Copied supplied tests and env_v66.sh unchanged. Prepared APFS copies of cache and the three completed NMCC shards; the interpreter is the existing campaign venv. Generated 26 Path A specs and 25 Path B specs; both SPECS_SHA256.txt files match the shipped lists exactly. Path A alone contains v66_identity. Both Path B curricula match the supplied bytes.

- Supplied validity unit tests: 18/18 PASS, zero skips.
- Existing targeted regression tests: 90/90 PASS, actual exit 0.
- Modified dynamics/evaluator/test files compile.
- Job planning: v66_identity 40 unique/0 stored; v66_smoke 6/0; v66_scaling 32/0.
- Smoke preflight provenance lists exactly the eight dynamics files required by Section 2.6.
- G0 identity, smoke execution, G1, G2, G3: PENDING. Unit tests and dry plans do not establish those scientific gates.

Logs are under the clone's runs/_logs. Stage 0 starts with a fresh plan when execution is authorized; the current stored counts are preparation observations.

## Fixed design and statistical declarations

Design overrides are candidateSamplingMode=spatial_maximin, initialShelterMode=demand_greedy, plannerShortlistObservation=enriched. Frozen final v65 learned checkpoints retain the zero defer prior. Only the designated iteration-zero arms and Path B training use below_sites. Departure delay is enabled only by the designated sensitivity arms.

G1 uses v66_main tapes 21-80: PASS iff t(rl - heuristic) <= -2 AND mean(rl - objective_greedy) <= 0. A failure requires Path B. It is not a reason to alter settings or rerun for a favorable outcome. If Path B completes before submission, it is primary regardless of Path A's result.

Primary sample: 60 tapes per training city, 120 paired city-tapes; learned values averaged over seeds within tape. Secondary samples: 1-20 and 1-80. Differences are first minus second. Report all eight declared contrasts:

    P1 rl:heuristic
    P2 rl:hazard_weighted
    P3 objective_greedy:heuristic
    P4 objective_greedy:hazard_weighted
    P5 objective_greedy:static_greedy
    P6 objective_greedy:initial_only
    P7 objective_greedy:objective_greedy_legacy
    P8 rl:objective_greedy

Use the supplied equal-city macro mean/SE, two-sided normal-reference Holm family at 5%, and per-seed values plus seed-and-tape bootstrap intervals for P1, P2, P8. Final checkpoints stay primary. Validation-selected checkpoints are secondary; choose minimum mean J across 20 validation tapes per seed, breaking ties toward the later iteration. Record the manual's exact declarations in LOG.md before the first Stage 1 episode, unchanged.

## Execution order after authorization

Use bash from the clone and source only runbook/env_v66.sh. Never source runbook/env.sh there: it changes directory to the original campaign. One panel at a time, no training/evaluator overlap. The manual's worker counts are planned configurations, subject to a fresh resource preflight; do not silently change scientific overrides or hot-add workers.

| Order | Panels / action | Workers plain/heavy | Required exit condition and outputs |
|---|---|---|---|
| 0 | v66_identity, then v66_smoke | 2/1 each | G0: ten cross-identities PASS on 40 legacy episodes; then 6/6 smoke, iteration-zero identity PASS, exact eight-file provenance |
| 1a | tune_deadline_main, v66_main | 2/0 | Tuning collected; complete 1,920-row main sample; Claim 1 three tape ranges, G1, people-level outcomes, old/new strip maps |
| Decision | G1 | no compute | PASS continues Path A; FAIL makes Path B required. Report the full gate table either way |
| 1b | main, then g2_seed1 | 2/1, then 2/0 | Complete heavy search/planner and ablation rows; main_secondary analysis; g2_seed1 reuses main |
| 2 | v66_iter0 | 2/1 | G2 identities for all three seeds; four-way index/learned/search decomposition and defer counts |
| 3 | v66_zero_shot, then zero_shot | 2/0 each | Seattle/Chicago, 20 tapes each, no retraining; per-city transfer table |
| 4 | v66_sensitivity, sim_checks, v66_sim_checks_depart | 2/0 each | Six settings; cost contrasts and people-level outcomes; density captures and progress for base/delay validation |
| 5 | v66_scaling | 1/0 | 32 total episodes; 8 expected to reuse the 2,500-person main configuration after it exists, 24 new; timing/outcomes |
| 6 | v66_planner | 1/1 | Raw versus enriched shortlist analysis and timing; optional full-search arm excluded by default |
| 7 | v66_ckpt_select, learning_curve | 2/0 each | Validation learning curve, per-seed selection rule; v66_selected only if selected iteration differs from 40 |
| 8 | tune_deadline_calm, gate1_calm, gate1_compound, capacity_sweep, kappa_sweep, plugin_check, site_loss | 2/1 for the six standard panels; site_loss 2/0 | Complete re-issued exhibits; site-loss identity versus new main; site-loss results --check; paper results --check |
| 9 | v66_ppo | 2/0 | Blocked until the correct three PPO sources are complete and required preceding PPO gates pass; G3 three cross-identities; paired PPO contrasts |

Run main before v66_iter0 because the latter expects its learned-search episodes from main. This follows the detailed stage dependency rather than treating the overview's day/night schedule as a mandatory order. A planned reuse count is verified from the fresh plan and exact job keys; it is never asserted as completed reuse in advance.

The old v65 Chicago planner skip and failed v65 iteration-zero identity remain historical results. They do not constitute v66 G0/G2 results. The v66 transfer panel is a new design and panel, not a rerun of the old B3 planner.

Optional accessibility_deficit, full-search, site-loss recourse, and rollout-recourse arms are not automatically selected. If later selected, retain their supplied pilot/stop rules. Never relaunch completed v65 panels.

## Gate commands and result generation

The following are reviewable commands, not a script that has been executed:

    cd /Users/huali/Desktop/RLEvacuationCollab-v66
    source runbook/env_v66.sh
    mkdir -p "$AN"

Stage 0, after the fresh plans and successful collection:

    REF="$CAMPAIGN_REPO/runs/paper_v65/panels/main/evaluation_episode_summary.csv"
    for a in objective_greedy objective_greedy_legacy heuristic static_greedy initial_only random 'rl#1' 'rl#2' 'rl#3' 'rl_early_search#1'; do
      chk crossidentity --csv-a "$(csv v66_identity)" --arm-a "$a" --csv-b "$REF" --arm-b "$a"
    done | tee "$AN/g0_identity.txt"
    chk identity --csv "$(csv v66_smoke)" --arm-a objective_greedy --arm-b 'rl_iter000_bs#1'

G0 requires ten actual PASS results, complete costs and decision lists, no episode failure, and 6/6 smoke. A v66_wait return or printed timing PASS alone is insufficient. Record the actual launcher/collector exits and inspect durable rows, manifest, claims, and failures before advancing.

Stage 1:

    ana claim1 --csv "$(csv v66_main)" --tapes 21:80 --gate --out "$AN/claim1_test"
    ana claim1 --csv "$(csv v66_main)" --tapes 1:20 --out "$AN/claim1_main20"
    ana claim1 --csv "$(csv v66_main)" --tapes 1:80 --out "$AN/claim1_all"
    ana strips --csv "$(csv v66_main)" --arms random objective_greedy heuristic static_greedy rl --png "$AN/strips_v66.png" --out "$AN/strips_v66"
    ana strips --csv "$CAMPAIGN_REPO/runs/paper_v65/panels/fu_b2_claim1_ext/evaluation_episode_summary.csv" --arms random objective_greedy heuristic static_greedy rl --png "$AN/strips_v65.png" --out "$AN/strips_v65"
    ana contrasts --csv "$(csv main)" --tapes 1:20 --pairs rl_early_search:rl rl_early_search:objective_greedy belief_rollout:objective_greedy rl_early_search:belief_rollout rl_forecast_ablated:rl rl_precommit:rl deadline_greedy_tuned:objective_greedy --out "$AN/main_secondary"

The supplied analysis tool's FAIL message mentions "Stage 3"; the retraining procedure is Section 6, Path B. Preserve its output verbatim and use the actual G1 rule above.

Stage 2:

    for k in 1 2 3; do
      chk identity --csv "$(csv v66_iter0)" --arm-a objective_greedy --arm-b "rl_iter000_bs#$k"
    done | tee "$AN/g2_identity.txt"
    ana contrasts --csv "$(csv v66_iter0)" --tapes 1:20 --pairs rl_iter000:objective_greedy rl_iter000_bs:objective_greedy rl_iter000_bs_search:objective_greedy rl:objective_greedy rl_early_search:rl rl_early_search:rl_iter000_bs_search --out "$AN/iter0_search"
    ana outcomes --csv "$(csv v66_iter0)" --out "$AN/iter0_outcomes"

G2 FAIL is reported with differing tapes/defer counts and permits continuation; it never permits an exact-identity claim.

Stages 3-6:

    ana claim1 --csv "$(csv v66_zero_shot)" --tapes 1:20 --out "$AN/zero_shot"
    ana sensitivity --csv "$(csv v66_sensitivity)" --out "$AN/sensitivity"
    ana outcomes --csv "$(csv v66_sensitivity)" --out "$AN/sensitivity_outcomes"
    chk timing --csv "$(csv v66_scaling)" | tee "$AN/scaling_timing.txt"
    ana outcomes --csv "$(csv v66_scaling)" --out "$AN/scaling_outcomes"
    ana contrasts --csv "$(csv v66_planner)" --tapes 1:5 --pairs belief_rollout:belief_rollout_raw rl_early_search:belief_rollout belief_rollout:objective_greedy --out "$AN/planner"
    chk timing --csv "$(csv v66_planner)" >> "$AN/planner.md"

Scaling tokens are fixed at 120/300/600/1200 for populations 1000/2500/5000/10000, surplus 20%; 9*token >= population in every arm. Departure-delay rows are out-of-distribution policy tests. Validation must retain density captures, progress, person-minute jam-density shares, clearance times, and cumulative-safe curves; do not infer calibration against observed evacuations.

Stage 7:

    ana contrasts --csv "$(csv v66_ckpt_select)" --tapes 1:10 --pairs rl_iter000:objective_greedy rl_iter005:objective_greedy rl_iter010:objective_greedy rl_iter015:objective_greedy rl_iter020:objective_greedy rl_iter025:objective_greedy rl_iter030:objective_greedy rl_iter035:objective_greedy rl_iter040:objective_greedy --out "$AN/ckpt_select"

If required, generate v66_selected with the three validation-selected iterations using the supplied generator, freeze that additional spec, run it, and analyze tapes 21-80. Its additional spec hash is expected; do not overwrite the original shipped lists or select using test outcomes.

Stage 8, after every required collected panel exists:

    "$PY" scripts/site_loss_identity.py --site-loss "$(csv site_loss)" --reference "$(csv main)"
    "$PY" scripts/site_loss_results.py --campaign "$SL_CAMPAIGN" --out "$SL_CAMPAIGN/results" --check
    "$PY" scripts/paper_results.py --campaign "$CAMPAIGN" --training $TRAINING --out "$CAMPAIGN/paper_results" --check

The frozen paper-results generator requires city_build_profile.json in the campaign root unless --city-profile is supplied. Preparation copies the existing five-city profile, byte-identical, into both new campaign roots. Its historical build timings remain historical metadata, not newly measured v66 benchmarks. The supplied profile SHA256 is 577e7760384c7fa6b1c3c66bdac15e1a4591fcd936072a1d09c89c5643072979.

Require actual exit 0, all 20 exhibits OK and every column audit OK. Retain all nonzero attempts. Missing exhibits are reported; no --expect-missing, source edit, or invented data. Inspect every PNG/PDF and the supporting CSV ranges. The old Figure4 safe-share values above 1 remain an unresolved content flag; this patch does not fix that plotting/measurement issue. Preserve any recurrence and report it for a supplied correction, without clipping or renormalizing.

Stage 9, only after its prerequisites:

    for a in objective_greedy heuristic static_greedy; do
      chk crossidentity --csv-a "$(csv v66_ppo)" --arm-a "$a" --csv-b "$(csv v66_main)" --arm-b "$a"
    done | tee "$AN/g3_identity.txt"
    ana contrasts --csv "$(csv v66_ppo)" "$(csv v66_main)" --tapes 21:80 --pairs rl_ppo:objective_greedy rl_ppo:heuristic rl:rl_ppo --out "$AN/ppo"

Pass both CSVs to preserve tape matching. G3 FAIL stops Stage 9.

## PPO prerequisite and Path B branch

Path A Stage 9 is currently blocked. Original PPO seed1 was intentionally stopped at 1190/2000 (E40/P1150), not complete; seeds2/3 do not exist. Its manifest still says running because it was preserved, while the user-stop evidence and zero processes establish USER_STOPPED. Do not copy that shard as a completed paper source. The legacy B6c/B6e checks remain unrun. No legacy restart is authorized by this code/preparation request.

After a later explicit instruction, legacy PPO would resume the same p980 seed1 with --resume, complete its final --expected 2000 --final --require-ppo gate and checkpoint minimum, then follow B6c, sequential Mac seeds2/3 and B6e. No concurrent cloud/ACCESS work is authorized.

If G1 requires Path B, use config/curricula_v66 and new names in the clone:

    C=config/curricula_v66/state_college_spokane_regime_v66_2500_paper.json
    v66_train paper_v66_seed1 "$C" 100 1
    chk training --dir runs/paper_v66_seed1 --expected 200 --final

Only after seed1 is complete and its owned processes end, run seed2, then seed3 with policy indices 2,3 and the same counts. Train the three new PPO shards sequentially under ppo_paper_v66.json, 1000 episodes per city, expected 2000 and --final --require-ppo. These are fresh training runs, never resumes of v65 weights.

Then source env_v66.sh with V66_PATH=B, use the already-prepared config/panels/v66b and separate paper_v66b / paper_v66b_site_loss stores, run the Path B smoke, repeat Stages 1-9, and cross-identify the seven plain arms against Path A. G0 legacy identity is run once in Path A, not repeated. Final Path B learned rows become primary if complete before submission; Path A becomes candidate-pool transfer evidence.

## Resources, elapsed-time estimates, and stopping

- V66 execution and 20-minute progress updates are now authorized; each launch remains subject to gate, integrity, process, and resource checks.
- On a later execution request, verify a fresh scoped process inventory, source/spec hashes, memory_pressure, swap trend, disk, power/sleep state, and the relevant resources.csv before launch.
- On this 16-GiB Mac, preserve the prior safeguards: no evaluator/trainer overlap; sequential training seeds; stop for current persistent swap growth or two free-memory readings below 10%, and for the manual's red pressure or sleep condition. Verify exact owner/argv/cwd/PPID/PGID before individually signaling own stage processes. Never group-signal or touch unrelated projects. Preserve rows and partial progress; no manual claim clearing.
- A resumed panel uses the identical launch after confirmed resource recovery and zero equivalent compute. Training resumes only the identical shard with --resume. Record holds separately; preserve raw sleep/resource-inflated times.
- Unit/G0 or smoke failure stops new v66 runs; G1 failure branches to retraining; G2 failure is a measured result; G3 failure stops Stage 9. A needed code/spec fix is reported for a new supplied revision, not silently applied after results exist.
- Historical manual estimates: Path A about 47 compute hours, roughly 3 calendar days with 2 plain/1 heavy worker; about 46 hours for parallel NMCC seeds on ACCESS, or roughly 6 Mac days for NMCC alone. These are not measured v66 forecasts and exclude the currently unresolved legacy PPO prerequisite.
- The budget table quotes about 138+30 Mac training hours for Path B. Thirty PPO hours is especially uncertain: the stopped v65 seed1 had already consumed over 11 hours of launch-clock time at 59.5%, including recorded holds. Do not present the manual's aggregate as a completion promise.
- After authorized execution, update stage ETAs from current-stage completed episode samples, with sample counts, resource phase, confidence, and America/Los_Angeles windows. Do not reuse the old v65 full-workflow ETA, extrapolate E onto P, discard outliers, or slide an unchanged forecast with elapsed time.

## Stage handoff and completion

After each completed stage, copy only collected CSVs, panel manifests, actual gate/analysis outputs, and logs into the original project's Claude outputs/v66_validity/results/<stage>/, with MANIFEST_SHA256.txt. Never copy episode stores. Keep analysis artifact paths/progress references usable in the clone. STATUS.md records one line per stage with start/end/gate/anomalies. Negative findings, manual skips, and incomplete stages remain explicit.

Final products: new campaign panel CSV/manifests; Claim 1 tables and maps; iteration/search decomposition; transfer/sensitivity/scaling/planner/learning/PPO tables; 20 paper exhibits; site-loss results; raw validation captures and provenance. No publication/push/upload or unrelated project action is included.
