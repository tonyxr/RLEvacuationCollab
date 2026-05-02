#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 08:50:12 2025

@author: Xiaoru Shi
"""

from typing import Dict
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass
import os
import math

from GNN import EvacPolicy, fit_gnn, grid_edge_index
from RewardProcessor import RewardProcessor, extract_reward_terms
from Cell import Cell

import time

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

    
@dataclass
class Transition:
    obs_ped: torch.Tensor
    obs_haz: torch.Tensor
    obs_inf: torch.Tensor
    action_sh: torch.Tensor
    action_mask: torch.Tensor
    logp_sh: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


class RLBridge:
    def __init__(self, core, mode="full",
                 gamma=0.995, lam=0.97,
                 clip_eps=0.15, lr=3e-4,
                 epochs=8, minibatch_size=8,
                 entropy_coef=0.003, value_coef=0.7,
                 print_every=20, debug=False, reward_interval: int = 1,
                 train_mode: bool = True,
                 target_kl: float = 0.03,
                 value_clip_eps: float = 0.2,
                 shelter_action_interval: int = 5,
                 exploration_rate: float = 0.15,
                 optimizer_name: str = "Adam",
                 max_episode_steps: int = 120,
                 no_reroute_patience: int = 3,
                 deployment_strategy: str = "rl",
                 target_active_shelters: int = 0):
        
        self.core = core
        self.gamma = gamma; self.lam = lam
        self.clip_eps = clip_eps; self.lr = lr
        self.epochs = epochs; self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef; self.value_coef = value_coef
        self.print_every = print_every; self.debug = debug
        self.reward_interval = reward_interval
        self.train_mode = bool(train_mode)
        self.target_kl = max(1e-4, float(target_kl))
        self.value_clip_eps = max(0.01, float(value_clip_eps))
        self.shelter_action_interval = max(1, int(shelter_action_interval))
        self.base_lr = float(lr)
        self.base_entropy_coef = float(entropy_coef)
        self.exploration_rate = min(0.9, max(0.0, float(exploration_rate)))
        self.exploration_floor = 0.05
        self.max_episode_steps = max(1, int(max_episode_steps))
        self.no_reroute_patience = max(1, int(no_reroute_patience))
        self.deployment_strategy = str(deployment_strategy).strip().lower()
        self.target_active_shelters = max(0, int(target_active_shelters))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rew = RewardProcessor(mode=mode)

        # Build policy
        self.nx, self.ny = int(core.cellX), int(core.cellY)
        self.num_cells = self.nx * self.ny

        # feature dims chosen from CAProcessor wires
        self.d_ped = 2       # count, avg_vel
        self.d_haz = 3       # heat, smoke, danger
        self.d_inf = 2       # fulfill, wellness

        self.policy = EvacPolicy(
            d_ped=self.d_ped, d_hazard=self.d_haz, d_infra=self.d_inf,
            embed_dim=32, heads=2,
            action_dim_shelter=self.num_cells + 1,   # +1 = no-op
            force_mlp=True, verbose=False
        ).to(self.device)
        opt_name = str(optimizer_name).strip().lower()
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.lr, weight_decay=1e-4)
        elif opt_name == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=self.lr, alpha=0.99)
        else:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.8, patience=2, min_lr=1e-5
        )

        # Disk checkpoint across replications
        self.ckpt_dir = os.path.join("runs", f"{str(core.address).replace(' ','_')}")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_path = os.path.join(self.ckpt_dir, "policy.pt")
        if os.path.exists(self.ckpt_path):
            try:
                self.policy.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
                if self.debug: print(f"[RLBridge] Loaded policy from {self.ckpt_path}")
            except Exception as e:
                print("[RLBridge] Load failed:", e)

        # graph edges for (nx, ny) grid (used if you later enable GNN)
        self.edge_index_np = grid_edge_index(self.nx, self.ny)
        self.edge_index = torch.as_tensor(self.edge_index_np, dtype=torch.long, device=self.device)

        # per-episode storage
        self.traj: list[Transition] = []
        self.t = 0
        self.reward_ema = 0.0
        self.reward_var_ema = 1.0
        self.reward_norm_beta = 0.98
        self.reward_norm_clip = 3.0
        self.reward_norm_temperature = 2.0
        self.shelter_evac_history = {}
        self.shelter_last_flow = {}
        self.installed_shelter_order = []
        self.shelter_install_step = {}
        self.shelter_rerouted_count = {}
        self.consecutive_no_reroute_installs = 0
        self.stop_new_shelter_install = False
        
    
    @staticmethod
    def _safe_tensor(x: torch.Tensor, clamp: float | None = None) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        if clamp is not None:
            x = torch.clamp(x, min=-float(clamp), max=float(clamp))
        return x

    def _safe_masked_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe_logits = self._safe_tensor(logits, clamp=50.0)
        safe_logits = safe_logits.masked_fill(~mask, -1e9)
        return self._safe_tensor(safe_logits, clamp=1e9)
        
    def _heuristic_pick_cell_idx(self) -> int:
        shelter_grid = getattr(self.core.shelterDS, "shelterByCell", None)
        candidate_grid = getattr(self.core.shelterDS, "shelterCanByCell", None)
        counts = np.asarray(getattr(self.core.cellTracker, "countByCell", None))
        if shelter_grid is None or candidate_grid is None or counts is None:
            return self.num_cells

        best_idx = self.num_cells
        best_ratio = float("-inf")
        for i in range(self.nx):
            for j in range(self.ny):
                if not candidate_grid[i][j]:
                    continue
                idx = i * self.ny + j
                active_evacuees = float(max(0.0, counts[idx]))
                remaining_capacity = 0.0
                for sh in shelter_grid[i][j]:
                    cap = float(max(0.0, getattr(sh, "shelterCap", 0.0)))
                    flow = float(max(0.0, getattr(sh, "shelterFlow", 0.0)))
                    if int(getattr(sh, "status", 0)) == 0:
                        remaining_capacity += max(0.0, cap - flow)
                ratio = active_evacuees / max(1.0, remaining_capacity)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = idx
        return int(best_idx)

    def _random_pick_cell_idx(self, action_mask: torch.Tensor) -> int:
        valid_idx = torch.where(action_mask[0, :self.num_cells])[0]
        if int(valid_idx.numel()) <= 0:
            return self.num_cells
        pick = int(torch.randint(low=0, high=int(valid_idx.shape[0]), size=(1,), device=self.device).item())
        return int(valid_idx[pick].item())

    def _update_shelter_evac_history(self):
        max_window = 15
        shelter_list = getattr(self.core.shelterDS, "shelterList", {})
        for sid, shelter in shelter_list.items():
            flow_now = float(max(0.0, getattr(shelter, "shelterFlow", 0.0)))
            if sid not in self.shelter_last_flow:
                delta = 0.0
            else:
                delta = max(0.0, flow_now - self.shelter_last_flow[sid])
            hist = self.shelter_evac_history.setdefault(sid, deque(maxlen=max_window))
            hist.append(delta)
            self.shelter_last_flow[sid] = flow_now

    def _latest_shelter_utilization_reward(self) -> tuple[float, float]:
        windows = (5, 10, 15)
        utilization = 0.0
        rerouted_arrival_speed_score = 0.0
        latest_ids = list(reversed(self.installed_shelter_order[-3:]))
        for idx, sid in enumerate(latest_ids):
            window = windows[idx]
            hist = self.shelter_evac_history.get(sid)
            if not hist:
                continue
            hist_list = list(hist)
            recent_deltas = hist_list[-window:]
            utilization += float(sum(recent_deltas))

            # Encourage arrivals that happen soon after rerouting to a newly installed shelter.
            # Earlier arrivals produce larger gain via time decay.
            install_step = int(self.shelter_install_step.get(sid, self.t))
            rerouted_count = float(max(0, self.shelter_rerouted_count.get(sid, 0)))
            if rerouted_count <= 0.0:
                continue
            tau = 5.0
            for lookback, delta in enumerate(reversed(recent_deltas)):
                if delta <= 0.0:
                    continue
                step_of_arrival = self.t - lookback
                age = max(0, step_of_arrival - install_step)
                decay = math.exp(-float(age) / tau)
                rerouted_arrival_speed_score += float(delta) * (1.0 + 0.1 * rerouted_count) * decay

        return utilization, rerouted_arrival_speed_score

    # ---- helpers ----
    def _get_obs_tensors(self):
        ct = self.core.cellTracker
        # flatten into (N, D)
        def _flat(a): return torch.as_tensor(np.asarray(a, dtype=np.float32).reshape(self.num_cells))
        ped = torch.stack([
            _flat(ct.countByCell),
            _flat(ct.avgVelocityByCell)
        ], dim=-1)                                     # (N,2)

        haz = torch.stack([
            _flat(ct.heatByCell),
            _flat(ct.smokeByCell),
            _flat(ct.dangerLevelByCell),
        ], dim=-1)                                     # (N,3)

        inf = torch.stack([
            _flat(ct.shelterFulfillByCell),
            _flat(ct.wellnessPenaltyByCell),
        ], dim=-1)                                     # (N,2)
        
        ped = self._safe_tensor(ped, clamp=1e6)
        haz = self._safe_tensor(haz, clamp=1e6)
        inf = self._safe_tensor(inf, clamp=1e6)

        return ped.to(self.device), haz.to(self.device), inf.to(self.device)
    
    def _build_action_mask(self) -> torch.Tensor:
        mask = torch.zeros((1, self.num_cells + 1), dtype=torch.bool, device=self.device)
        if self.core.shelterDS.shelterCanByCell is not None:
            for i in range(self.nx):
                for j in range(self.ny):
                    idx = i * self.ny + j
                    if self.core.shelterDS.shelterCanByCell[i][j]:
                        mask[:, idx] = True
        mask[:, -1] = True  # no-op always valid
        return mask

    def _select_actions(self, gnn_input):
        sh_logits, value = self.policy(gnn_input)  # (1, A), (1,)
        mask = self._build_action_mask()
        masked_logits = self._safe_masked_logits(sh_logits, mask)
        value = self._safe_tensor(value, clamp=1e6)

        sh_dist = torch.distributions.Categorical(logits=masked_logits)
        if self.train_mode:
            decay_steps = max(10.0, 0.4 * float(self.max_episode_steps))
            eps_t = self.exploration_floor + (self.exploration_rate - self.exploration_floor) * math.exp(-float(self.t) / decay_steps)
            if torch.rand((), device=self.device).item() < eps_t:
                valid_idx = torch.where(mask[0])[0]
                pick = int(torch.randint(low=0, high=int(valid_idx.shape[0]), size=(1,), device=self.device).item())
                a_sh = valid_idx[pick].unsqueeze(0)
            else:
                a_sh = sh_dist.sample()
        else:
            a_sh = torch.argmax(masked_logits, dim=-1)
        return a_sh, mask, sh_dist.log_prob(a_sh), value

    @staticmethod
    def _idx_to_cell(idx, ny):
        if idx < 0: return None
        i, j = divmod(int(idx), int(ny))
        return (i, j)

    # ---- main step called by Core each timestep ----
    def step(self) -> Dict[str, float]:
        # Build obs and run policy
        x_ped, x_haz, x_inf = self._get_obs_tensors()          # (N, D)
        g = fit_gnn(x_ped, x_haz, x_inf)                       # batch dim = 1
        a_sh, action_mask, lp_sh, value = self._select_actions(g)

        # Map actions to environment (A-1 = no-op)
        added_sh = 0
        attempted_sh = 0
        installed_capacity = 0.0
        immediate_rerouted_count = 0.0
        shelter_decision = "no-op"
        shelter_gate_open = ((self.t % self.shelter_action_interval) == 0)
        current_active_shelters = int(len(self.core.shelterDS.shelterList))
        shelter_budget_reached = (
            self.target_active_shelters > 0 and current_active_shelters >= self.target_active_shelters
        )
        if shelter_budget_reached:
            a_sh = torch.as_tensor([self.num_cells], dtype=torch.long, device=self.device)
            sh_logits, _ = self.policy(g)
            masked_logits = self._safe_masked_logits(sh_logits, action_mask)
            sh_dist = torch.distributions.Categorical(logits=masked_logits)
            lp_sh = sh_dist.log_prob(a_sh)
            shelter_decision = (
                f"no-op (shelter budget reached: active={current_active_shelters}/"
                f"{self.target_active_shelters})"
            )
        elif self.deployment_strategy in {"none", "initial_only"}:
            a_sh = torch.as_tensor([self.num_cells], dtype=torch.long, device=self.device)
            sh_logits, _ = self.policy(g)
            masked_logits = self._safe_masked_logits(sh_logits, action_mask)
            sh_dist = torch.distributions.Categorical(logits=masked_logits)
            lp_sh = sh_dist.log_prob(a_sh)
            shelter_decision = "no-op (initial shelters only strategy)"
        elif shelter_gate_open and self.deployment_strategy in {"random", "heuristic"}:
            if self.deployment_strategy == "random":
                picked_idx = self._random_pick_cell_idx(action_mask)
                shelter_decision = "random-upper-layer"
            else:
                picked_idx = self._heuristic_pick_cell_idx()
                shelter_decision = "heuristic-upper-layer"
            a_sh = torch.as_tensor([picked_idx], dtype=torch.long, device=self.device)
            sh_logits, _ = self.policy(g)
            masked_logits = self._safe_masked_logits(sh_logits, action_mask)
            sh_dist = torch.distributions.Categorical(logits=masked_logits)
            lp_sh = sh_dist.log_prob(a_sh)
            if picked_idx >= self.num_cells:
                shelter_decision += " no-op (no valid candidate cell)"
        elif shelter_gate_open:
            if int(a_sh.item()) >= self.num_cells:
                fallback_idx = self._heuristic_pick_cell_idx()
                a_sh = torch.as_tensor([fallback_idx], dtype=torch.long, device=self.device)
                sh_logits, _ = self.policy(g)
                masked_logits = self._safe_masked_logits(sh_logits, action_mask)
                sh_dist = torch.distributions.Categorical(logits=masked_logits)
                lp_sh = sh_dist.log_prob(a_sh)
                shelter_decision = "policy-noop overridden to keep shelter budget on track"
            if int(a_sh.item()) >= self.num_cells:
                shelter_decision = "no-op (no valid candidate cell)"
            else:
                attempted_sh = 1
                cell = self._idx_to_cell(int(a_sh.item()), self.ny)
                sid = self.core.shelterDS.newShelter({"cell": cell}, self.core.cellTracker)
                if sid is not None:
                    added_sh = 1
                    rerouted = 0
                    ped_ds = getattr(self.core, "pedDS", None)
                    new_sh = self.core.shelterDS.shelterList.get(sid)
                    installed_capacity = float(max(0.0, getattr(new_sh, "shelterCap", 0.0))) if new_sh is not None else 0.0
                    self.installed_shelter_order.append(sid)
                    if ped_ds is not None and hasattr(ped_ds, "reroute_to_new_shelter_if_closer"):
                        try:
                            rerouted = int(ped_ds.reroute_to_new_shelter_if_closer(new_sh))
                        except Exception:
                            rerouted = 0
                    immediate_rerouted_count = float(max(0, rerouted))
                    self.shelter_install_step[sid] = int(self.t)
                    self.shelter_rerouted_count[sid] = int(max(0, rerouted))
                    if rerouted > 0:
                        self.consecutive_no_reroute_installs = 0
                    else:
                        self.consecutive_no_reroute_installs += 1
                    shelter_decision = f"installed shelter_id={sid} at cell={cell} rerouted={rerouted}"
                else:
                    shelter_decision = f"attempted install at cell={cell} (no candidate available)"
        elif not shelter_gate_open:
            # Force no-op action on non-deployment timesteps.
            a_sh = torch.as_tensor([self.num_cells], dtype=torch.long, device=self.device)
            sh_logits, _ = self.policy(g)
            masked_logits = self._safe_masked_logits(sh_logits, action_mask)
            sh_dist = torch.distributions.Categorical(logits=masked_logits)
            lp_sh = sh_dist.log_prob(a_sh)
            shelter_decision = f"no-op (placement gated; interval={self.shelter_action_interval})"
            
        # Compute reward
        pedRes = self.core.pedDS.result
        self._update_shelter_evac_history()
        if (self.t % self.reward_interval) == 0:
            count_casualty = int(pedRes.get("casualty", 0))
            terms = extract_reward_terms(self.core.cellTracker)
            delayed_new_shelter_evac, rerouted_arrival_speed_score = self._latest_shelter_utilization_reward()
            r = self.rew.rewardMode(
                numCasualties=count_casualty,
                t=self.t,
                wellnessPenaltySum=terms["wellnessPenaltySum"],
                fulfillmentSum=terms["fulfillmentSum"],
                evacuatedTotal=int(pedRes.get("evacuated", 0)),
                totalShelters=len(self.core.shelterDS.shelterList),
                installedShelterCapacityThisStep=installed_capacity,
                delayedNewShelterEvac=delayed_new_shelter_evac,
                reroutedArrivalSpeedScore=rerouted_arrival_speed_score,
                immediateReroutedCount=immediate_rerouted_count,
                strandedCount=int(len(getattr(self.core.pedDS, "pedAgentList", []))),
            )
        else:
            # no new reward signal this step
            r = 0.0
        # Running normalization to improve stochastic training stability.
        delta = float(r) - self.reward_ema
        self.reward_ema += (1.0 - self.reward_norm_beta) * delta
        self.reward_var_ema = self.reward_norm_beta * self.reward_var_ema + (1.0 - self.reward_norm_beta) * (delta * delta)
        reward_scale = math.sqrt(max(self.reward_var_ema, 1e-6))
        r_z = float((float(r) - self.reward_ema) / reward_scale)
        r_z = float(np.clip(r_z, -self.reward_norm_clip, self.reward_norm_clip))
        r_norm = float(np.tanh(r_z / self.reward_norm_temperature))
        
        print(
            f"[RL decision] t={self.t} reward={float(r):.3f} "
            f"arrival={int(pedRes.get('arrival', 0))} "
            f"casualty={int(pedRes.get('casualty', 0))} "
            f"evacuated={int(pedRes.get('evacuated', 0))} "
            f"shelter={shelter_decision}",
            flush=True,
        )

        reward = torch.as_tensor([r_norm], dtype=torch.float32, device=self.device)
        
        done = torch.as_tensor([0.0], dtype=torch.float32, device=self.device)  # episode end flagged by Core

        # Store transition
        self.traj.append(Transition(
            obs_ped=x_ped.detach(), obs_haz=x_haz.detach(), obs_inf=x_inf.detach(),
            action_sh=a_sh.detach(),
            action_mask=action_mask.detach(),
            logp_sh=lp_sh.detach(),
            value=value.detach(), reward=reward.detach(), done=done
        ))
        self.t += 1

        return {
            "reward": float(r),
            "reward_norm": float(r_norm),
            "added_shelters": added_sh,
        }

    # ---- called by Core at episode end ----
    def end_episode(self):
        if not self.traj:
            return
        if not self.train_mode:
            self.traj.clear()
            self.t = 0
            self.shelter_evac_history.clear()
            self.shelter_last_flow.clear()
            self.installed_shelter_order.clear()
            self.shelter_install_step.clear()
            self.shelter_rerouted_count.clear()
            self.consecutive_no_reroute_installs = 0
            self.stop_new_shelter_install = False
            return

        # Build tensors
        rewards = torch.cat([tr.reward for tr in self.traj])            # (T,)
        values  = torch.cat([tr.value  for tr in self.traj]).squeeze(-1) # (T,)
        dones   = torch.cat([tr.done   for tr in self.traj]).squeeze(-1) # (T,)

        with torch.no_grad():
            # GAE-Lambda
            T = rewards.shape[0]
            adv = torch.zeros(T, device=self.device)
            lastgaelam = 0.0
            next_value = torch.tensor(0.0, device=self.device)
            for t in reversed(range(T)):
                nonterminal = 1.0 - float(dones[t].item())
                delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
                lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
                adv[t] = lastgaelam
                next_value = values[t]
            returns = adv + values

        # Normalize advantages
        adv_mean = adv.mean()
        adv_std = adv.std()
        if not torch.isfinite(adv_mean) or not torch.isfinite(adv_std) or float(adv_std.item()) < 1e-8:
            adv = adv - torch.nan_to_num(adv_mean, nan=0.0)
        else:
            adv = (adv - adv_mean) / (adv_std + 1e-8)
        adv = self._safe_tensor(adv, clamp=1e4)
        returns = self._safe_tensor(returns, clamp=1e6)

        # Recompute logprobs for current policy (we need obs compact form)
        obs_ped = torch.stack([tr.obs_ped for tr in self.traj])   # (T, N,2)
        obs_haz = torch.stack([tr.obs_haz for tr in self.traj])   # (T, N,3)
        obs_inf = torch.stack([tr.obs_inf for tr in self.traj])   # (T, N,3)

        acts_sh = torch.stack([tr.action_sh for tr in self.traj]).squeeze(-1)  # (T,)
        masks_sh = torch.stack([tr.action_mask for tr in self.traj]).squeeze(1)  # (T, A)

        old_lp_sh = self._safe_tensor(torch.stack([tr.logp_sh for tr in self.traj]).detach(), clamp=1e4)
        old_values = self._safe_tensor(torch.stack([tr.value for tr in self.traj]).detach().squeeze(-1), clamp=1e6)

        T = obs_ped.shape[0]
        idx = torch.randperm(T, device=self.device)
        mb = max(1, T // self.minibatch_size)
        
        loss_history = []
        entropy_history = []
        approx_kl_history = []
        clipfrac_history = []
        
        stop_early = False
        for _ in range(self.epochs):
            if stop_early:
                break
            for k in range(0, T, mb):
                sel = idx[k:k+mb]              # indices of timesteps in this minibatch
                bs = sel.shape[0]              # minibatch size (number of time steps)
        
                # obs_ped: (T, N, d_ped)  -> (bs, N, d_ped)
                x_p_sel = obs_ped[sel]         # (bs, N, d_ped)
                x_h_sel = obs_haz[sel]         # (bs, N, d_haz)
                x_i_sel = obs_inf[sel]         # (bs, N, d_inf)
        
                # Flatten cells, keep batch, build batch vector:
                flat_p = x_p_sel.reshape(bs * self.num_cells, self.d_ped)
                flat_h = x_h_sel.reshape(bs * self.num_cells, self.d_haz)
                flat_i = x_i_sel.reshape(bs * self.num_cells, self.d_inf)
        
                # batch[i*num_cells : (i+1)*num_cells] = i
                batch_vec = torch.arange(bs, device=self.device).repeat_interleave(self.num_cells)
        
                g = fit_gnn(
                    x_ped=flat_p,
                    x_hazard=flat_h,
                    x_infra=flat_i,
                    edge_index=self.edge_index,
                    batch=batch_vec,
                )
        
                sh_logits, values_now = self.policy(g)   # shapes: (bs, A), (bs,)
                sh_logits = self._safe_masked_logits(sh_logits, masks_sh[sel])
                values_now = self._safe_tensor(values_now, clamp=1e6)
                if not torch.isfinite(sh_logits).all() or not torch.isfinite(values_now).all():
                    if self.debug:
                        print("[RLBridge] Non-finite minibatch outputs detected; skipping minibatch update.")
                    continue
                sh_dist = torch.distributions.Categorical(logits=sh_logits)
        
                lp_sh = sh_dist.log_prob(acts_sh[sel])              # (bs,)
                entropy = sh_dist.entropy().mean()
        
                ratio_sh = torch.exp(lp_sh - old_lp_sh[sel])        # (bs,)
                ratio = ratio_sh                                    # (bs,)
                
                approx_kl = torch.mean(old_lp_sh[sel] - lp_sh).item()
                clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.clip_eps).float()).item()
        
                adv_now = adv[sel]                                  # (bs,)
                surr1 = ratio * adv_now
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_now
                policy_loss = -torch.mean(torch.min(surr1, surr2))
        
                ret_now = returns[sel]                              # (bs,)
                # values_now is already (bs,) after squeeze
                values_now = values_now.squeeze(-1)
                value_pred_clipped = old_values[sel] + torch.clamp(
                    values_now - old_values[sel], -self.value_clip_eps, self.value_clip_eps
                )
                value_loss_unclipped = (values_now - ret_now) ** 2
                value_loss_clipped = (value_pred_clipped - ret_now) ** 2
                value_loss = 0.5 * torch.mean(torch.max(value_loss_unclipped, value_loss_clipped))
        
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                if not torch.isfinite(loss):
                    if self.debug:
                        print("[RLBridge] Non-finite PPO loss detected; skipping minibatch update.")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
        
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                loss_history.append((policy_loss.item(), value_loss.item()))
                entropy_history.append(entropy.item())
                approx_kl_history.append(approx_kl)
                clipfrac_history.append(clipfrac)
                if approx_kl > self.target_kl:
                    stop_early = True
                    break

        # checkpoint
        try:
            torch.save(self.policy.state_dict(), self.ckpt_path)
        except Exception as e:
            print("[RLBridge] Save failed:", e)

        # reset episode storage
        self.traj.clear()
        self.t = 0
        self.shelter_evac_history.clear()
        self.shelter_last_flow.clear()
        self.installed_shelter_order.clear()
        self.shelter_install_step.clear()
        self.shelter_rerouted_count.clear()
        self.consecutive_no_reroute_installs = 0
        self.stop_new_shelter_install = False

        if loss_history:
           pol = float(np.mean([x[0] for x in loss_history]))
           val = float(np.mean([x[1] for x in loss_history]))
           ent = float(np.mean(entropy_history))
           kl = float(np.mean(approx_kl_history))
           cf = float(np.mean(clipfrac_history))
           avg_r = float(rewards.mean().item())
           # Adaptive entropy and clipping for highly stochastic scenarios.
           reward_vol = float(rewards.std().item()) if rewards.numel() > 1 else 0.0
           stochastic_scale = min(2.0, max(0.8, 1.0 + reward_vol))
           self.entropy_coef = min(0.02, max(0.001, self.base_entropy_coef * stochastic_scale))
           if cf > 0.45:
               self.clip_eps = max(0.08, self.clip_eps * 0.9)
           elif cf < 0.15:
               self.clip_eps = min(0.22, self.clip_eps * 1.05)
           if kl > self.target_kl * 1.2:
               for pg in self.optimizer.param_groups:
                   pg["lr"] = max(1e-5, float(pg["lr"]) * 0.85)
           elif kl < self.target_kl * 0.5:
               for pg in self.optimizer.param_groups:
                   pg["lr"] = min(self.base_lr, float(pg["lr"]) * 1.03)
           self.lr_scheduler.step(avg_r)
           current_lr = float(self.optimizer.param_groups[0]["lr"])
           print(
               f"[RL LEARN CHECK] steps={T} avg_reward={avg_r:.4f} "
               f"policy_loss={pol:.4f} value_loss={val:.4f} "
               f"entropy={ent:.4f} approx_kl={kl:.5f} clipfrac={cf:.3f} "
               f"clip_eps={self.clip_eps:.3f} lr={current_lr:.6f}",
               flush=True,
           )