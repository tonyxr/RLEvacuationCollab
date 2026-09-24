# Administrator-facing shelter deployment MDP and optimization design

## Research objective

The control problem is to select a fixed number of additional shelter sites
over time so that as many evacuees as possible reach safety, casualties are
avoided, evacuation is fast, and exposure to the stochastic hazard is low. At
each decision epoch the controller names one regional cell to prioritize. A
shared, deterministic lower layer -- maximum remaining capacity in that cell,
tie-broken by OSM identifier -- resolves the specific building installed
there. This rule is identical for the RL policy and every heuristic
benchmark, so a policy comparison isolates cell-prioritization behavior, not
building-level tie-breaking. (Revision, Model version 18: the action space
was moved from exact-candidate selection to this cell-priority design after
the exact-candidate space proved too high a decision complexity to converge
reliably; see the "Action" section below and the companion note
`CELL_PRIORITY_ACTION_SPACE_20260920.md`.)

## Decision process

Let simulator timesteps be \(t=0,\ldots,H\), and let deployment decision
epochs be \(\tau_0,\tau_1,\ldots\), separated by the configured action interval
\(K\) when another feasible deployment is actually made. If the budget is
exhausted or every candidate is temporarily unsafe, the last action remains
open until a later executed action or the physical episode terminal. The
implemented sequence is

\[
x_{\tau_k}\;\xrightarrow{h}\;o_k
\;\xrightarrow{\pi_\theta}\;a_k
\;\xrightarrow{\text{simulate until next executed action or terminal}}\;
(r_k,x_{\tau_{k+1}},o_{k+1}).
\]

The simulator first commits movement, hazard effects, shelter admissions, and
casualties at a timestep boundary. It then forms \(o_k\). Action \(a_k\) is
installed immediately and receives only outcomes in
\((\tau_k,\tau_{k+1}]\). Pre-action outcomes are never credited to the new
action. The process is semi-Markov at deployment epochs because one decision
transition spans multiple simulator timesteps.

### State

The Markov simulator state \(x_t\) contains all variables needed for the next
transition:

- every active evacuee's location, route, speed, group size, panic state, and
  hazard-response state;
- the road graph and current link occupancies and congestion;
- each hazard's source, location, age, intensity, spread state, wind field, and
  random-stream state;
- every installed shelter's site, capacity, flow, and availability;
- remaining candidate sites, deployment budget, clock, and cumulative outcome
  ledger.

Given \(x_t\), an action, and the simulator's keyed stochastic shocks, the
distribution of \(x_{t+1}\) does not depend on earlier states. This is the
state used to justify the Markov transition model; it is not presented to an
administrator as a giant feature vector.

### Observation and formal classification

The policy sees \(o_k=h(x_{\tau_k})\), an operational summary obtainable from
incident dashboards, traffic monitoring, hazard maps, and the shelter
registry. It does not see individual latent panic states or future hazard
draws. Consequently, the simulator is Markov, but the controller faces a
finite-horizon POMDP and a semi-Markov process at deployment epochs. Calling
the policy input the full Markov state would be inaccurate.

### Spatial discretization and variable region-node count

The grid dimensions \(X\) and \(Y\) are experiment inputs rather than fixed
properties of the policy. A condition therefore contains \(M=XY\) region
nodes. The chosen dimensions and boundaries remain fixed within an episode and
within a training/evaluation condition so that a node keeps the same physical
meaning throughout a trajectory, but \(X\), \(Y\), and hence \(M\), are varied
between the registered resolution conditions.

The projected road-network extent is divided into the requested number of
rectangular cells under either of two registered modes:

- `equal_area`: x and y boundaries are equally spaced in the simulator's local
  metre coordinates, so all cells have equal area.
- `node_density_adaptive`: each axis boundary is an empirical quantile of the
  OSM walking-network node coordinates. Dense coordinate ranges therefore have
  narrower intervals and finer spatial resolution.

Both modes use the same region-feature schema, row-major identifier rule,
four-neighbour edge-construction rule, and candidate-to-region assignment rule
at a given \(X\) and \(Y\). The number of rows in the observation changes with
\(M\), but the width and meaning of each row do not. The GNN applies shared
node encoders, shared message-passing weights, permutation-invariant pooling,
and a shared candidate scorer; it therefore does not learn a separate
parameter vector for each region and its trainable parameter count does not
depend on \(M\). Region identity must not be represented by a learned row-major
index. If position is used, it is computed from the cell's normalized physical
centroid and, for unequal adaptive cells, its normalized physical area.

The adaptive construction is axis-aligned and separable; it is not a quadtree
and does not guarantee identical two-dimensional node counts. It uses only the
fixed road graph, never realized pedestrian or hazard outcomes, so it does not
leak scenario information. OSM road-node density is an infrastructure
resolution proxy, not resident population density.

The partition mode, \(X\), \(Y\), minimum interval constraint, complete x/y
edge arrays, and edge SHA-256 digest are stored in run metadata. Model version
18 checkpoints also carry the partition contract and the action-space contract
(cell-priority vs. the retired exact-candidate design) and fail on a different
mode, grid shape, or action space. The architecture is reusable across dimensions, but checkpoints
are not shared in the confirmatory experiment: every grid-resolution or
partition-mode accuracy comparison trains a separate policy for each
preprocessing condition. This prevents a resolution-induced observation shift
from being mistaken for a treatment effect. A single mixed-resolution policy
would be a separate transfer experiment and would require packed variable-size
graph batches and an explicit cross-resolution training distribution.

### Operational observation graph

Every dynamic policy receives the same immutable `RegionalObservation`. It is
an administrator-readable graph with three per-region layers, one global
dashboard vector, and one row per regional cell -- exactly one action slot per
cell, whatever the number of raw candidate buildings the cell holds. Each
symbol below denotes one feature. The computation column states how the code
produces it without hiding several quantities inside an opaque learned state.

#### Pedestrian layer

| Notation | Feature | Computation | Decision value |
|---|---|---|---|
| \(n_i\) | active population | Active group-weighted population in region \(i\), divided by initialized population. | Locates unresolved demand. |
| \(m_i\) | mobility delay | One minus mean current speed divided by configured maximum speed; zero in an empty region. | Detects congestion or hazard-impaired movement. |
| \(t_i\) | mean remaining route time | Group-weighted remaining route distance divided by a speed floor, averaged over non-panicked routed evacuees, then divided by episode duration. | Separates nearby demand from people facing long trips. |
| \(l_i\) | long-route population | Share of active people whose estimated remaining route time exceeds the larger of one deployment interval and one quarter of the episode duration. | Makes persistent accessibility deficits explicit. |
| \(w_i^s\) | stable wellness | Share of active people neither affected nor panicked. | Identifies demand that can be rerouted normally. |
| \(w_i^e\) | exposed wellness | Share affected by the hazard but not panicked. | Identifies people already experiencing hazard effects. |
| \(w_i^p\) | panicked wellness | Share currently panicked. | Separates behaviorally unstable demand from route-responsive demand. |

The three wellness shares are mutually exclusive and sum to one in an occupied
region. They are zero in an empty region. Individual identities are not passed
to the policy.

#### Hazard layer

| Notation | Feature | Computation | Decision value |
|---|---|---|---|
| \(d_i\) | current danger | Current simulator danger level in region \(i\), normalized to \([0,1]\). | Measures current exposure and casualty risk. |
| \(f_i\) | forecast danger | Deterministic propagation of the current front over one deployment interval using the simulator's spread probabilities and wind direction; no future random draws are consumed. | Prevents a currently quiet but soon hazardous site from appearing safe. |
| \(q_i\) | source proximity | One minus distance from the region centroid to the nearest active hazard source, divided by the study-area diagonal. | Supplies a stable warning while the stochastic front is sparse. |

The forecast is an operational nowcast from the current hazard map, not access
to realized future hazards. Wind and spread are observable scenario inputs.

#### Physical-environment and shelter layer

| Notation | Feature | Computation | Decision value |
|---|---|---|---|
| \(c_i\) | uncommitted installed capacity | Capacity neither occupied nor reserved by en-route pedestrians in region \(i\), divided by initialized population. | Avoids treating promised slots as available twice. |
| \(u_i\) | shelter commitment | Admitted plus reserved population divided by installed capacity in the region; zero without an installed shelter. | Distinguishes genuinely unused from already-promised capacity. |
| \(b_i\) | deployable capacity | Capacity of the best currently available candidate in the region, divided by initialized population. | Shows how much relief can be added. |
| \(k_i\) | candidate availability | Remaining candidate count in the region divided by the episode's total remaining candidate count. | Exposes where placement options remain. |
| \(g_i\) | road-node share | Walking-network nodes in the region divided by network nodes in the study area. | Represents local network support and partition density. |
| \(x_i\) | east position | East coordinate of the physical region centroid, normalized to the study extent. | Preserves relative pedestrian, shelter, and hazard geometry. |
| \(y_i\) | north position | North coordinate of the physical region centroid, normalized to the study extent. | Preserves relative pedestrian, shelter, and hazard geometry. |
| \(a_i\) | region area | Physical region area divided by total study area. | Retains the unequal scale of adaptive cells. |

All region features are recomputed for the selected partition. Counts use fixed
scenario denominators, route times use the fixed episode duration, and
coordinates use fixed study bounds. Thus changing \(M\) changes resolution but
not feature meaning. Empty regions remain in the graph with zero pedestrian
features and their observed hazard and infrastructure features.

#### Global dashboard context

| Notation | Feature | Computation |
|---|---|---|
| \(h\) | time remaining | Remaining timesteps divided by the fixed horizon. |
| \(n\) | active population | Total active population divided by initialized population. |
| \(b\) | deployment budget | Remaining deployments divided by the fixed maximum. |
| \(\rho\) | network load | Initialized demand divided by initialized demand plus physical walking-link storage. |
| \(v\) | wind speed | Wind speed divided by wind speed plus configured maximum walking speed. |
| \(e\) | eastward wind component | East component mapped from \([-1,1]\) to \([0,1]\); calm wind is 0.5. |
| \(z\) | northward wind component | North component mapped from \([-1,1]\) to \([0,1]\); calm wind is 0.5. |
| \(\sigma\) | hazard spread | Mean spread-rate parameter among active hazards. |
| \(\eta_P\) | population/network density | Initialized population \(P\) divided by \(P+G\), where \(G\) is the number of walking-network nodes. This is the bounded transform of people per network node. |
| \(\eta_H\) | hazard-instance load | Number of configured hazard instances \(J\) divided by \(J+1\); this stays bounded without assuming a maximum hazard count. |
| \(\eta_\pi\) | configured panic susceptibility | Fraction of pedestrians susceptible to persistent panic at first qualifying exposure. |
| \(\eta_B\) | deployment-capacity coverage | Maximum dynamic deployment capacity divided by initialized population and clipped to one. With equal capacity tokens this is \(Bc/P\); in legacy site-capacity mode it uses the capacities of the best \(B\) resolved sites. |

The current cumulative safe-completion and casualty counts are not duplicated
in this vector. Their backward-looking rates enter the temporal momentum
vector below, because an administrator can observe recent outcomes while still
being denied future hazard draws.

#### Exact candidate-site table

| Notation | Feature | Computation | Decision value |
|---|---|---|---|
| \(p_j\) | candidate capacity | Site capacity divided by initialized population. | Measures potential service. |
| \(x_j\) | east position | Candidate east coordinate normalized to study bounds. | Distinguishes sites sharing a host region. |
| \(y_j\) | north position | Candidate north coordinate normalized to study bounds. | Distinguishes sites sharing a host region. |
| \(d_j\) | distance from open shelter | Shortest network distance from the candidate to an open shelter with capacity, divided by the study-area diagonal. | Rewards spatial coverage rather than duplication. |
| \(f_j\) | forecast danger | The host region's one-interval forecast danger. | Makes prospective shelter safety directly observable. |
| \(s_j\) | hazard safety margin | Candidate distance from the nearest active hazard source, divided by the study-area diagonal. | Favors physical separation from the source. |
| \(r_j\) | reroutable population | Capacity-capped active population for whom the candidate shortens the remaining route, divided by initialized population. | Measures how many people can benefit. |
| \(\delta_j\) | future risk-time reduction | Capacity-capped saved person-minutes weighted by one plus the greater of current and forecast origin danger, then divided by twice initialized population and episode duration. | Uses one score to prioritize both earlier safety and urgent exposure reduction. |

The candidate table has a stable action slot, OSM node identifier, host region,
features, and mask for every initial site. The OSM identifier is used only to
execute the exact action and is never embedded. Installed, zero-capacity, or
otherwise unavailable sites remain in the table but are masked. A site is also
masked when its forecast danger exceeds the configured
`maximumShelterForecastDanger` (0.6 in the registered profiles). If all sites
are forecast unsafe, deployment pauses instead of forcing a hazardous shelter.
All displayed fractions are clipped to \([0,1]\); there is no running
observation normalization.

#### Graph relations

The spatial relation is the union of bidirectional four-neighbor region edges
and bidirectional road links crossing region boundaries. Its messages are
neighbor means, so changing local degree does not change scale. The route
relation links each occupied origin region and its assigned shelter region in
both directions. Its edge weight is assigned group population divided by
initialized population. Route messages are weighted sums rather than weighted
means so a large assignment remains different from a small one. Two message
layers allow an urgent condition to reach a region up to two relational hops
away while keeping the network compact.

#### Temporal momentum and observation history

Every simulator boundary before termination is encoded by the same GNN and
then passed through one episode-level LSTM. The LSTM state is reset only at a
true physical episode boundary. PPO stores the raw graph observations rather
than detached embeddings, allowing gradients to train both the spatial and
temporal encoders. At decision time the following features make the direction
of change explicit; each rate compares only the current and preceding observed
boundary and is scaled to one deployment interval.

| Notation | Feature | Computation |
|---|---|---|
| \(\dot S\) | safe-completion velocity | New safe completions divided by initialized population and elapsed time. |
| \(\dot C\) | casualty incidence velocity | New casualties divided by initialized population and elapsed time. |
| \(\dot N\) | active-population clearance velocity | Decline in active population share over the observed interval. |
| \(\dot E\) | exposure-reduction velocity | Decline in active-population-weighted current danger. |
| \(\dot T\) | route-time-reduction velocity | Decline in population-weighted normalized remaining route time. |
| \(\dot L\) | long-route-reduction velocity | Decline in the population share currently assigned a long route. |
| \(\dot F\) | forecast-risk-reduction velocity | Decline in active-population-weighted forecast danger. |
| \(\dot\rho\) | network-load-reduction velocity | Decline in the observed network-load share. |
| \(\ell\) | time since deployment | Timesteps since the most recent installed shelter divided by the deployment interval and clipped to one. |

Positive values denote improvement except \(\dot C\), whose adverse direction
is retained explicitly. These trends help distinguish a deteriorating hazard
front from an evacuation already improving without another shelter. They are
observational context, not separate shaping rewards.

### Action

Let \(C\) be the set of regional cells, one action slot per cell. The action is

\[
a_k=i,\qquad i\in C,\quad m_{k,i}=1,
\]

meaning “prioritize cell \(i\).” The executor does not choose among buildings:
it calls the shared deterministic site rule
\[
j^\star(i)=\arg\min_{j\in \mathrm{candidates}(i)}\bigl(-\mathrm{cap}_j,\ \mathrm{OSMID}_j\bigr),
\]
(maximum remaining capacity, OSM identifier as the tie break) and installs
\(j^\star(i)\). This rule is exposed to every policy in advance: slot \(i\)'s
capacity, position, and safety features already describe \(j^\star(i)\), so a
policy's ranking over cells already accounts for what would be built there.
The rule is re-run every decision epoch (the winning building in a cell can
change once an earlier one is installed), but the cell-indexed action table
itself is fixed for the whole episode -- a cell with no remaining candidate
keeps its slot, permanently masked infeasible, rather than shrinking the
table. There is no no-op action: when a decision is due and at least one cell
is feasible, every dynamic policy uses one unit of the common budget. If a due
decision has no forecast-safe candidate, the unused token is retried at the
next simulator boundary instead of being discarded until the next absolute
clock multiple. After an installation, the next interval starts from that
actual deployment time. Because
\(j^\star(\cdot)\) is the same function for RL and every benchmark, the only
way policies can differ is in which cell they prioritize -- which is exactly
the quantity the confirmatory experiment compares across policies as the
evacuation unfolds (see the "Experiment" material in the companion note).

Routing and capacity assignment are one transaction. A route to a shelter is
installed only after capacity is reserved, and occupied plus reserved capacity
may never exceed the shelter capacity. Reassignment first proves that the new
route and slot exist, then atomically transfers the old promise. Admission
consumes the promise; casualty, panic, termination, or shelter loss releases
it. Unreserved walk-ins may use only genuinely uncommitted capacity. When a
new shelter cannot serve every beneficiary, slots are assigned in descending
per-person reduction in the same active-plus-exposure-time proxy used by the
base policy, with stable pedestrian identifiers breaking ties. Weighted
simulation cohorts remain indivisible while travelling; a cohort receives a
route only when its full represented population can be reserved.

### Benchmark policy

The benchmark implements the stated active-population rule over cells:

\[
a_k^{\mathrm{heur}}=\arg\max_{i:\,m_{k,i}=1}N_{k,i},
\]

where \(m_{k,i}\) is the common cell action mask and \(N_{k,i}\) is cell \(i\)'s
active population. Stable cell-table order breaks ties. The benchmark receives
the full common observation, although its prespecified rule deliberately uses
active population only; the building installed once it names a cell is
resolved by the same \(j^\star(\cdot)\) rule used everywhere else.

## Reward

For interval \(k\), define:

- \(\Delta S_k\): new safe completions, including shelter admission and safe planned-destination arrival;
- \(\Delta C_k\): new casualties;
- \(N_i(t)\): active population in region \(i\) at timestep \(t\);
- \(D_i(t)\in[0,1]\): normalized danger;
- \(P\): initialized population;
- \(H\): episode horizon.

Define active evacuation person-time and hazard-exposure person-time over the
post-action interval:

\[
T_k=\sum_{t=\tau_k+1}^{\tau_{k+1}}\sum_iN_i(t),
\qquad
E_k=\sum_{t=\tau_k+1}^{\tau_{k+1}}\sum_iN_i(t)D_i(t),
\]

where \(\tau_{k+1}\) is the next executed-action boundary or \(H\) for
the final action. A scheduled opportunity with no feasible action is not a
transition boundary.

The reward is

\[
r_k=\underbrace{\frac{\Delta S_k}{P}}_{\text{people reaching safety}}
-\underbrace{3\frac{\Delta C_k}{P}}_{\text{casualties}}
-\underbrace{\frac{T_k}{PH}}_{\text{evacuation time}}
-\underbrace{\frac{E_k}{PH}}_{\text{hazard exposure}}.
\]

Thus each safe arrival receives immediate positive credit, each casualty
receives a larger immediate penalty, every person still evacuating accrues a
time cost, and time in more dangerous regions accrues an additional exposure
cost. The intervals are non-overlapping and collectively cover every outcome
after the first executed action. In particular, exhausting the deployment
budget does not censor the final action's later casualties, completions,
evacuation time, or exposure. There is no shelter-utilization, rerouting,
selected-site, or attributed-admission shaping term; training and held-out
evaluation use the same outcome equation.

The casualty coefficient has a structural justification rather than being an
arbitrary shaping parameter. One active person can contribute at most \(H\)
evacuation person-timesteps and \(H\) exposure person-timesteps because
\(D_i(t)\leq1\). A casualty could therefore avoid at most \(2/P\) future time
cost. The casualty penalty of \(3/P\) strictly dominates that maximum avoided
cost; losing the possible safe-completion reward strengthens the dominance
further. Death cannot become an attractive shortcut for reducing the active
population.

The scientific episode objective is undiscounted (\(\gamma=1\)). Its completion
and casualty increments telescope, and it equals normalized terminal outcome
minus normalized full-episode hazard-weighted person-time. The registered v28
controller learns from exact paired intervention values; it does not alter or
clip this reward to create a denser proxy.

No terminal bonus, site-criticality reward, rerouting bonus, utilization
objective, potential shaping, exponential moving normalization, or ad hoc
reward clipping is part of the scientific objective.

## Evacuation-process visualization

Visualization is a reporting layer, not part of the MDP. It observes the
simulator only after the outcome ledger and regional decision have been
committed at a timestep boundary; it never changes the observation, reward,
action, random-number stream, routing, or shelter executor.

The primary map experiment fixes seven north-up milestones at minutes
0, 10, 20, 30, 40, 50, and 60. Each panel uses the exact
OpenStreetMap road graph loaded for that run and overlays one red point per
active pedestrian agent, a continuous heatmap of every cell's normalized
danger, hazard sources, initial shelters, dynamically installed shelters, the
exact selected candidate, and its host region. The danger scale is fixed at 0--1 rather
than normalized independently within each panel. RL and heuristic figures use
the same scenario seed, map extent, milestone schedule, and visual encoding.

Every figure is accompanied by source tables for milestone and decision-epoch
pedestrian positions, shelter state, continuous cell danger, and deployment
decisions. A manifest records
the scenario and policy seeds, coordinate convention, OpenStreetMap
attribution, artifact hashes, requested milestones, and captured milestones.
This makes the figure auditable and permits independent re-rendering for the
paper. The background is the OSM-derived road network rather than an online
raster tile, avoiding a hidden network dependency during experiments.

## Stochastic-hazard contract

The configuration fields for casualty, spread, and speed reduction are
interpreted as a mean in percent and a variance in squared percentage points.
For example, `[10, 4]` means a mean of 10% and a standard deviation of 2
percentage points. Each hazard draws one clipped probability for each effect at
initialization; the experiment seed makes these draws reproducible.

Hazard evolution has a dedicated pseudorandom stream. Pedestrian casualty
shocks are counter-based potential outcomes indexed by scenario, timestep, and
pedestrian identifier. This prevents policy-dependent population counts from
advancing a shared global random stream. The backtest records all component
seeds and rejects a matched comparison unless the complete hazard-trajectory
digests are identical across policies.

Cell hazard state (s_i(t)\in\{0,\ldots,5\}) supplies the normalized decision
danger (D_i(t)=s_i(t)/5). This removes unreachable heat/smoke thresholds and
is the exact `[0,1]` variable used in the reward proof. A hazard's sampled
casualty probability is cumulative over 60 minutes of continuous level-5
exposure. It is converted to an exposure-time-consistent per-step probability;
levels 0--3 are nonlethal, level 4 receives half severity, and level 5 receives
full severity. When active hazards overlap, complementary survival
probabilities are multiplied. Speed reduction remains a contemporaneous
state-five effect scaled by s_i(t)/5 and is applied before pedestrian movement.
Heat and smoke remain visualization diagnostics and additive contributions
replace the previous last-writer-wins behavior. Expired hazards cease lethal
and speed effects while residual impacted cells remain costly.

The configured panic level is the susceptibility fraction among pedestrians
when they first encounter danger level 3 or higher. Each individual receives
one keyed Bernoulli trial at that first qualifying exposure; a non-susceptible
person is not retried every minute. Panic is permanent after onset. At every
node decision, a panicked pedestrian independently uses the frozen
highest-occupancy incident edge with probability 0.5 or a random incident edge
with probability 0.5.

## Learning and optimization

The recurrent PPO actor remains available as a registered ablation. Model v28
uses fitted NMCC policy improvement for the deployed controller because the
v25 pilot showed that one-pass score fitting memorized the newest branch states
without improving unseen decisions. The GNN and LSTM are still the complete
function approximator; only the supervision and action-selection layer change.

The legacy actor emits one logit per regional cell. It gathers the graph embedding of
each cell, combines it with the eight slot features (describing whichever
building the shared deterministic rule would install in that cell) and global
context, and applies one scorer shared across all cells. Infeasible cells
are masked before categorical sampling, so the stored probability belongs to
the exact cell executed. A fixed prior ranks cells by active population and
exactly reproduces the benchmark's cell choice when the learned residual is
zero. PPO learns a bounded correction from route burden, wellness, current
and forecast hazard, installed and deployable capacity, the resolved site's
safety and estimated risk-time reduction, relational messages, wind, remaining time, and
budget. The residual can reverse the prior's ranking; the prior is an
optimization aid, not a restriction to the heuristic. Shared parameters permit
variable cell counts without a separate learned parameter for every cell.
Reducing the action space from one row per raw candidate building to one row
per cell is itself part of the credit-assignment remedy: it removes decision
complexity that carried no information relevant to outcomes (choosing among
near-identical buildings within the same cell) while keeping the strategically
meaningful choice -- which part of the map to prioritize -- fully exposed to
learning.

The GNN processes every observed simulator boundary, not only action times.
Its pooled graph context and the explicit momentum vector update a 96-unit
LSTM. At a deployment epoch, the actor combines the current candidate and host-
region embeddings with this recurrent context. This lets the decision depend
on whether evacuation, routes, exposure, and forecast danger have been
improving or deteriorating over the preceding trajectory.

The pedestrian, hazard, and infrastructure layers have separate encoders
before fusion. Spatial and route edges then use separate transformations in two
residual message layers. A learned attentive pool represents distributed need,
while a maximum pool preserves a rare urgent region. A recurrent critic trunk
feeds four independent value heads for safe completion, casualties, evacuation
time, and hazard exposure. Their signed predictions sum exactly to the scalar
diagnostic value. Each head is trained against its own one-step duration-aware
TD(0) target with a raw smooth-L1 loss. The critic and NMCC world heads have an
optimizer partition disjoint from the actor/shared-policy partition, so critic
or auxiliary error cannot directly move policy logits. The actor adds host-
region, candidate, global, and temporal embeddings before producing its
bounded residual logit.

The same shared encoders and scorer are used for every requested grid
dimension. At runtime, graph edges and candidate-to-region indices are rebuilt
from the selected partition. Pooling uses learned attention and maxima rather
than a flattened region vector, so neither actor nor critic has a weight matrix whose
shape depends on \(M\). The primary experiment still batches only transitions
from one resolution and partition mode at a time and trains a separate
checkpoint for each condition; architectural size-flexibility is not treated
as evidence that one learned policy transfers without retraining.

### Model v28 fitted intervention-value controller

At a labelled decision state, candidate branches start from one exact simulator
snapshot and share an independent common-random-number future tape. The first
two deployments are both exhaustively branched and continued to the physical
episode terminal. These are the actions for which a short branch omitted most
outcome variance in the v25 audit. Later deployments may use the configured
short horizon and a stratified candidate subset, but the executed action and
the fixed risk-time-reduction base action are always included.

Every exact labelled episode is retained in a bounded, checkpointed replay
dataset. The split is by complete episode into fitting, early-stopping, and
deployment-gate partitions: no recurrent prefix or stochastic trajectory can
appear in more than one. System-identification
updates own the relational GNN and episode LSTM. During fitted control those
representations are frozen, so a small action-label set cannot overwrite the
learned natural dynamics. The reinitialized intervention ensemble then fits a
wide-and-deep value model over all retained training episodes: its deep path
uses the full GNN/LSTM embeddings, while a low-variance linear path directly
uses the normalized regional pedestrian, hazard, infrastructure, candidate,
scenario, and momentum features. Episode-heldout loss supplies early stopping.
This expanding refit prevents a small newest rollout from erasing older city,
hazard, population, or panic scenarios while retaining nonlinear spatial and
temporal capacity.

Let \(h_b\) be the fixed risk-time-reduction base action's prior logit, \(T\) the
ranking temperature, and \(v\) the value scale. Intervention head \(m\)
predicts the exact physical advantage \(\widehat A_j^{(m)}/v\). Only
within-state differences are identified, so uncertainty is computed on the
paired effect relative to the risk-time-reduction base action \(b\),
\(\Delta_j^{(m)}=(\widehat A_j^{(m)}-\widehat A_b^{(m)})/v\). Its conservative
value is

\[
L_j=\operatorname{mean}_m\Delta_j^{(m)}
-\kappa\operatorname{sd}_m\Delta_j^{(m)}.
\]

This makes \(L_b=0\) exactly and prevents arbitrary ensemble-head offsets from
being mistaken for intervention uncertainty. While the gate is closed, the
controller uses the complete risk-time-reduction prior vector \(h_j\). With the gate open,
the deployed score is \(h_b+(v/T)L_j\): the prior supplies the fallback and a
common score level, while the identified physical effect supplies candidate
ordering. Thus a perfect value fit has exactly the physical-return ordering
instead of counting the base benefit twice. The same
composite score—not the value head alone—is evaluated by a gate set never used
for fitting or early stopping. Only exhaustively branched decisions may certify
deployment.
The selected candidate is compared with the fixed base-policy candidate, and
the correction gate opens only after the lower confidence bound on this paired
return gain is positive for the required number of updates. Until then, the
controller executes the base policy and collects scheduled epsilon-greedy
exploration. Thus a poorly calibrated model cannot degrade the known policy
while it is learning.

The old generic convergence rule is also inapplicable here: training-return
stationarity and PPO's 0.015 KL threshold neither measure fitted-policy progress
nor distinguish scenario noise. V27 convergence is defined by finite complete
training, a nonempty episode-heldout exact set, and a positive paired-gain lower
bound. Confirmatory performance still requires frozen-checkpoint, matched-seed
evaluation with equal realized deployment count and total installed capacity.

Model v23 uses an actor learning rate of 0.0001, critic/world learning rate of
0.0003, 0.10 policy clipping, one actor epoch, four critic epochs, 0.005 initial
entropy coefficient, raw smooth-L1 critic loss, gradient-norm clipping, and a
0.015 actor target KL. Entropy is divided by the logarithm of the current
feasible-action count, so its scale remains comparable as sites are installed
or masked. Before an actor epoch, both actor parameters and actor-optimizer
state are snapshotted. If full-rollout KL exceeds the target, the attempt is
restored, the actor learning rate is reduced, and no accepted-update schedule
advances. Dropout is not used because changing hidden dropout masks between
behavior collection and PPO replay would corrupt the likelihood ratio. Whole
physical episodes, rather than independently shuffled decisions, form the
recurrent minibatches. Each episode is replayed from a zero LSTM state through
every cached predecision graph frame; an episode is never split merely to meet
the nominal minibatch size.

For actor decision \(k\), the component return is the complete semi-Markov
Monte Carlo return

\[
G_k^j=r_k^j+\gamma^{\Delta_k}(1-d_k)G_{k+1}^j.
\]

It contains no critic bootstrap. Its scalar sum is centered and scaled only by
a lagged baseline estimated from prior rollouts in the same city, population,
hazard-count, and within-episode decision-position stratum. The current rollout
is added to the baseline only after its policy attempt is complete. The critic
instead receives the independent label

\[
y_k^j=r_k^j+\gamma^{\Delta_k}(1-d_k)V_{\mathrm{old}}^j(o_{k+1}).
\]

This separation prevents critic error or current-batch centering from
truncating delayed actor credit. The behavior policy is held fixed for a
complete multi-episode rollout and optimization occurs only at the rollout
boundary. The single-city default is eight episodes;
pooled experiments enlarge this to an integer number of complete city blocks
(ten episodes for five cities), preventing city imbalance within a gradient
update. Training checkpoints contain model, separate actor and critic
optimizers, lagged baseline statistics, accepted actor-rollout counters,
action-generator, episode-order generator, episode, update, raw recurrent
observation histories, full partial-rollout state, and an
interface signature containing every PPO hyperparameter. Incompatible
checkpoints fail rather than partially load.

These choices improve numerical conditioning and make the interface
stationary, but they do not guarantee global convergence of neural PPO. The
project treats convergence as an empirical gate requiring prespecified reward,
KL, entropy, and held-out-performance diagnostics across seeds. A checkpoint
that merely loads successfully is not considered converged.

## Outcome definitions

- `safe_completed = shelter_evacuated + arrival`.
- `casualty` is terminal harm.
- `unfinished` means still active when the finite horizon ends and is not relabeled as arrival.
- Mean safe-completion time covers all safe completions; mean shelter-evacuation time is retained as a narrower diagnostic.
- Restricted mean time to safety assigns the episode horizon to casualties and
  unfinished pedestrians. It is the confirmatory time metric because the
  conditional mean among successful pedestrians can misleadingly favor a
  policy that brings very few people to safety.

The population identity checked after every episode is

\[
\text{safe completed}+\text{casualty}+\text{unfinished}=P.
\]

## Material passport

- Direct implementation evidence: `DecisionInterface.py`, `RewardProcessor.py`, `RLBridge.py`, `GNN.py`, `ShelterDatabase.py`, `NetworkOptimization.py`, `EvacuationVisualizer.py`, and the focused tests.
- Model inference: the coefficient-three dominance argument follows from the enforced danger bound and finite horizon.
- Empirical claim boundary: whether RL outperforms the heuristic is not assumed by the design and must be established by held-out matched-seed evaluation.
