#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 08:50:12 2025

@author: Xiaoru Shi
"""

from typing import Dict
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
                 train_mode: bool = True):
        
        self.core = core
        self.gamma = gamma; self.lam = lam
        self.clip_eps = clip_eps; self.lr = lr
        self.epochs = epochs; self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef; self.value_coef = value_coef
        self.print_every = print_every; self.debug = debug
        self.reward_interval = reward_interval
        self.train_mode = bool(train_mode)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rew = RewardProcessor(mode=mode)

        # Build policy
        self.nx, self.ny = int(core.cellX), int(core.cellY)
        self.num_cells = self.nx * self.ny

        # feature dims chosen from CAProcessor wires
        self.d_ped = 2       # count, avg_vel
        self.d_haz = 3       # heat, smoke, danger
        self.d_inf = 3       # fulfill, guided, wellness

        self.policy = EvacPolicy(
            d_ped=self.d_ped, d_hazard=self.d_haz, d_infra=self.d_inf,
            embed_dim=32, heads=2,
            action_dim_shelter=self.num_cells + 1,   # +1 = no-op
            force_mlp=True, verbose=False
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

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
            _flat(ct.guidanceInterByCell),
            _flat(ct.wellnessPenaltyByCell),
        ], dim=-1)                                     # (N,3)

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
        masked_logits = sh_logits.masked_fill(~mask, -1e9)

        sh_dist = torch.distributions.Categorical(logits=masked_logits)
        if self.train_mode:
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
        shelter_decision = "no-op"
        if int(a_sh.item()) < self.num_cells:
            attempted_sh = 1
            cell = self._idx_to_cell(int(a_sh.item()), self.ny)
            sid = self.core.shelterDS.newShelter({"cell": cell}, self.core.cellTracker)
            if sid is not None:
                added_sh = 1
                shelter_decision = f"installed shelter_id={sid} at cell={cell}"
            else:
                shelter_decision = f"attempted install at cell={cell} (no candidate available)"
            
        # Compute reward
        pedRes = self.core.pedDS.result
        if (self.t % self.reward_interval) == 0:
            count_casualty = int(pedRes.get("casualty", 0))
            terms = extract_reward_terms(self.core.cellTracker)
            r = self.rew.rewardMode(
                numCasualties=count_casualty,
                t=self.t,
                wellnessPenaltySum=terms["wellnessPenaltySum"],
                fulfillmentSum=terms["fulfillmentSum"],
                evacuatedTotal=int(pedRes.get("evacuated", 0)),
                totalShelters=len(self.core.shelterDS.shelterList),
                shelterInstalledThisStep=added_sh,
                shelterInstallAttemptsThisStep=attempted_sh,
            )
        else:
            # no new reward signal this step
            r = 0.0
        
        print(
            f"[RL decision] t={self.t} reward={float(r):.3f} "
            f"casualty={int(pedRes.get('casualty', 0))} "
            f"evacuated={int(pedRes.get('evacuated', 0))} "
            f"shelter={shelter_decision}",
            flush=True,
        )

        reward = torch.as_tensor([r], dtype=torch.float32, device=self.device)
        
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

        return {"reward": float(r), "added_shelters": added_sh}

    # ---- called by Core at episode end ----
    def end_episode(self):
        if not self.traj:
            return
        if not self.train_mode:
            self.traj.clear()
            self.t = 0
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
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Recompute logprobs for current policy (we need obs compact form)
        obs_ped = torch.stack([tr.obs_ped for tr in self.traj])   # (T, N,2)
        obs_haz = torch.stack([tr.obs_haz for tr in self.traj])   # (T, N,3)
        obs_inf = torch.stack([tr.obs_inf for tr in self.traj])   # (T, N,3)

        acts_sh = torch.stack([tr.action_sh for tr in self.traj]).squeeze(-1)  # (T,)
        masks_sh = torch.stack([tr.action_mask for tr in self.traj]).squeeze(1)  # (T, A)

        old_lp_sh = torch.stack([tr.logp_sh for tr in self.traj]).detach()

        T = obs_ped.shape[0]
        idx = torch.randperm(T, device=self.device)
        mb = max(1, T // self.minibatch_size)
        
        loss_history = []
        entropy_history = []
        approx_kl_history = []
        clipfrac_history = []

        for _ in range(self.epochs):
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
                sh_logits = sh_logits.masked_fill(~masks_sh[sel], -1e9)
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
                value_loss = F.mse_loss(values_now, ret_now)
        
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                
                loss_history.append((policy_loss.item(), value_loss.item()))
                entropy_history.append(entropy.item())
                approx_kl_history.append(approx_kl)
                clipfrac_history.append(clipfrac)

        # checkpoint
        try:
            torch.save(self.policy.state_dict(), self.ckpt_path)
        except Exception as e:
            print("[RLBridge] Save failed:", e)

        # reset episode storage
        self.traj.clear()
        self.t = 0

        if loss_history:
           pol = float(np.mean([x[0] for x in loss_history]))
           val = float(np.mean([x[1] for x in loss_history]))
           ent = float(np.mean(entropy_history))
           kl = float(np.mean(approx_kl_history))
           cf = float(np.mean(clipfrac_history))
           avg_r = float(rewards.mean().item())
           print(
               f"[RL LEARN CHECK] steps={T} avg_reward={avg_r:.4f} "
               f"policy_loss={pol:.4f} value_loss={val:.4f} "
               f"entropy={ent:.4f} approx_kl={kl:.5f} clipfrac={cf:.3f}",
               flush=True,
           )