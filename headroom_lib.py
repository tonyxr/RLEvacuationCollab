"""Episode runner shared by the calibration and headroom experiments.

Mirrors production ordering exactly: dynamics advance, the step reward is
scored, then at decision epochs (t = 1, 11, 21, ... as in RLBridge) the
observation is built, the policy chooses a cell, and the shared executor
installs it.
"""
import numpy as np
import CounterfactualBranch as CB
import nmcc_testbed as T
from DecisionInterface import (RegionalObservationBuilder, ActivePopulationHeuristic,
                               AccessibilityDeficitHeuristic, RegionalShelterExecutor,
                               estimate_remaining_hazard_weighted_travel_time)
from RewardProcessor import RewardProcessor

FIRST_DECISION = 1


class Episode:
    def __init__(self, core, *, horizon, interval, budget):
        self.core = core
        self.horizon = int(horizon)
        self.interval = int(interval)
        self.budget = int(budget)
        self.population = int(core.initial_population)
        self.builder = RegionalObservationBuilder(core, initial_population=self.population,
                                                  horizon=self.horizon, maximum_deployments=self.budget)
        self.executor = RegionalShelterExecutor(core)
        self.reward_model = RewardProcessor()
        self.t = 0
        self.deployed = 0
        self.ret = 0.0
        self.prev = CB.outcome_snapshot(core)
        self.decisions = []

    def is_decision(self, t):
        off = t - FIRST_DECISION
        return off >= 0 and off % self.interval == 0 and self.deployed < self.budget

    def step_dynamics(self):
        self.t += 1
        CB.advance_one_timestep(self.core)
        cur = CB.outcome_snapshot(self.core)
        self.prev = cur
        return 0.0

    def observe(self):
        return self.builder.build(decision_index=self.deployed, simulation_time=self.t,
                                  remaining_deployments=self.budget - self.deployed)

    def install(self, obs, action):
        from types import SimpleNamespace
        before = estimate_remaining_hazard_weighted_travel_time(
            self.core,
            episode_horizon=self.horizon,
        )
        receipt = self.executor.execute(obs, SimpleNamespace(action_index=int(action)))
        after = estimate_remaining_hazard_weighted_travel_time(
            self.core,
            episode_horizon=self.horizon,
        )
        reward = self.reward_model.evaluate(
            before=obs.outcome,
            after=obs.outcome,
            remaining_risk_time_before=before,
            remaining_risk_time_after=after,
            initial_population=self.population,
            horizon=self.horizon,
        )
        self.ret += float(reward.total)
        self.deployed += 1
        return receipt

    # restore helpers: episode bookkeeping must travel with the simulator state
    def bookkeeping(self):
        return (self.t, self.deployed, self.ret, self.prev, list(self.decisions))

    def set_bookkeeping(self, b):
        self.t, self.deployed, self.ret, self.prev, d = b
        self.decisions = list(d)
        # the builder/executor hold `core`, whose attributes restore() replaces in place


def run(core, policy, *, horizon=60, interval=10, budget=5, rng=None):
    ep = Episode(core, horizon=horizon, interval=interval, budget=budget)
    while ep.t < horizon:
        ep.step_dynamics()
        if ep.is_decision(ep.t):
            obs = ep.observe()
            if obs.has_feasible_action:
                a = policy(ep, obs, rng)
                rec = ep.install(obs, a)
                ep.decisions.append((ep.t, int(rec.executed_cell), int(np.flatnonzero(obs.action_mask).size)))
            else:
                ep.decisions.append((ep.t, -1, 0))
    o = CB.outcome_snapshot(core)
    return {"return": ep.ret, "safe": o.safe_completed, "casualty": o.casualties,
            "unfinished": o.active_population, "deployed": ep.deployed, "decisions": ep.decisions}


# ----------------------------------------------------------------- policies
_H = ActivePopulationHeuristic()

def heuristic(ep, obs, rng):
    return _H.select(obs).action_index

def accessibility(ep, obs, rng):
    centers = np.column_stack([obs.region_east_positions, obs.region_north_positions])
    return AccessibilityDeficitHeuristic(centers).select(obs).action_index

def uniform(ep, obs, rng):
    return int(rng.choice(np.flatnonzero(obs.action_mask)))

def prior_sampling(temperature):
    """The v22/v23 behavior policy at residual = 0: softmax(relative_active / T)."""
    def policy(ep, obs, rng):
        feas = np.flatnonzero(obs.action_mask)
        act = obs.active_by_cell[obs.candidate_cell_indices[feas]].astype(float)
        rel = act / act.max() if act.max() > 0 else np.zeros_like(act)
        z = rel / float(temperature)
        p = np.exp(z - z.max()); p /= p.sum()
        return int(feas[rng.choice(feas.size, p=p)])
    return policy


def build_core(seed, **kw):
    params = dict(grid=20, cell_x=8, cell_y=8, population=800, stop_time=61,
                  spread_rate=(4, 2), casualty_rate=(40, 9), panic_rate=0.5, hazard_count=3,
                  shelter_capacity_token=160)
    params.update(kw)
    return T.build(scenario_seed=int(seed), **params)


# --- coarse dashboard-only comparison policies (no candidate oracle values)
def route_saving(ep, obs, rng):
    """Capacity-capped local demand weighted by local observed hazard."""
    feas = np.flatnonzero(obs.action_mask)
    cells = obs.candidate_cell_indices[feas]
    score = np.minimum(
        obs.active_by_cell[cells], obs.candidate_capacities[feas]
    ) * (1.0 + np.maximum(
        obs.danger_by_cell[cells], obs.candidate_forecast_danger[feas]
    ))
    return int(feas[int(np.argmax(score))])

def reroutable(ep, obs, rng):
    """Capacity-capped active demand in the candidate's own region."""
    feas = np.flatnonzero(obs.action_mask)
    cells = obs.candidate_cell_indices[feas]
    score = np.minimum(
        obs.active_by_cell[cells], obs.candidate_capacities[feas]
    )
    return int(feas[int(np.argmax(score))])
