# Pedestrian speed and link-congestion contract

## Units and timestep

`maxSpeed` is the pedestrian free-flow walking speed in metres per minute.
The configured value is 64 m/min, or 1.067 m/s (3.84 km/h). Each simulator
transition represents one minute by default (`timeStepMinutes=1.0`), so an
unaffected pedestrian can traverse at most 64 metres in a transition before
congestion is applied. The nominal 60-transition horizon is therefore one
hour, with a maximum uncongested path length of 3.84 km. Primary map panels at
minutes 0, 10, 20, 30, 40, 50, and 60 are ten minutes apart; interpreting
adjacent panels as one-step motion makes movement appear much faster than the
model specifies. The five fixed downtown OSM radii are enlarged to 2--6 km so
longer trips and network-scale congestion remain observable.

The 64 m/min value is below the frequently used Weidmann free-flow reference
of 1.34 m/s. It is retained as a conservative common walking speed rather than
retuned after inspecting policy outcomes.

## Congestion law

All policies use the same congestion transition. At the start of each
10-second integration substep, pedestrians already on a directed OSM edge and
pedestrians waiting to enter their next route edge are counted on one physical
road segment. Opposing directed graph edges with the same endpoints and OSM
way identifier share the same pedestrian density.

For physical link \(e\),

\[
\rho_e = \frac{N_e}{L_e w},
\qquad
\frac{v_e}{v_0} = 1 - \exp\left[-1.913\left(
\frac{1}{\rho_e} - \frac{1}{5.4}\right)\right].
\]

The multiplier is one on an empty link and decreases monotonically with
density. The parameters 1.913 and 5.4 pedestrians/m² are the conventional
Weidmann/Kladek values. The implementation uses one declared effective
walkable width, `w=3.0 m`, because reliable pedestrian clear width is not
available for every OSM street in all five cities.

At or above the modeled jam density, the speed multiplier is regularized at
0.05 rather than zero. This prevents an overloaded entry wave
from becoming an irreversible absorbing gridlock state. The regularization is
not a calibrated behavioral parameter and must be varied in sensitivity
analysis.

The realized movement speed is

\[
v_{i,e,t}=v_i^{\mathrm{free}}
(1-r_{i,t}^{\mathrm{hazard}})
f(\rho_{e,t}),
\]

where juxtaposition denotes multiplication.
Hazard effects are rebuilt each one-minute transition and do not compound
accidentally. Congestion is rebuilt six times per transition. If a pedestrian
reaches another edge, its new link contributes to the next synchronized
10-second density snapshot; travel within a substep uses that substep's frozen
state.

## Synchronization and policy fairness

Link densities are frozen before any pedestrian moves in an integration
substep. No agent receives a less congested link merely because it appears
earlier in a Python container.
Counterflow is combined on the physical segment. RL, the active-population
heuristic, and static predeployment all invoke this same transition after
their shelter choices, so congestion is part of the environment rather than
private policy information.

The regional observation and reward equations are unchanged. Congestion
reduces actual cell speed and delays safe completion, so it is already valued
through safe completions, casualties, and hazard-weighted person-time without
adding another reward coefficient.

## Diagnostics and calibration requirements

Every timestep log now includes mean and minimum congestion speed ratio,
maximum link density, congested population, occupied physical links, internal
substep count, and mean effective walking speed. Episode summaries include
person-minute-weighted mean speed ratio, congested person-minute share,
maximum observed density, and the complete congestion contract.

Before confirmatory execution:

1. Check that plausible demand levels produce both uncongested and congested
   links rather than a universal floor or ceiling.
2. Prespecify sensitivity runs for effective widths 2, 3, and 4 m and minimum
   speed ratios 0.025, 0.05, and 0.10. Include a numerical-integration check at
   5, 10, and 15 seconds on a representative scenario.
3. Do not reuse a policy trained without congestion. The map runner rejects a
   checkpoint whose source manifest lacks the exact congestion, horizon,
   two-minute decision cadence, enlarged map footprint, or shelter-budget
   contract.

## Method sources

- Weidmann, *Transporttechnik der Fussgänger* (ETH Zürich):
  https://www.research-collection.ethz.ch/bitstreams/46577ec8-832f-4303-a9a6-989f7084a130/download
- Review and analytical statement of the Kladek–Newell–Weidmann relation:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8685666/
