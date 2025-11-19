#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 02:22:30 2025

@author: Xiaoru Shi
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os



#from torch_geometric.nn import GATConv, global_mean_pool
#from torch_geometric.data import Data

_HAS_PYG = False
if os.getenv("EVAC_ENABLE_PYG", "0") == "1":   
    try:
        from torch_geometric.nn import GATConv, global_mean_pool   
        from torch_geometric.data import Data                      
        _HAS_PYG = True
    except Exception:
        _HAS_PYG = False

@dataclass
class GNNInput: 
    x_ped: torch.Tensor # Shape = (N, D_p)
    x_hazard: torch.Tensor # Shape = (N, D_h)
    x_infra: torch.Tensor # Shape = (N, D_i)
    
    edge_index: Optional[torch.Tensor] = None
    batch: Optional[torch.Tensor] = None


def getStateConcat(cellStateGrid) -> torch.Tensor:
    rows = []
    for row in cellStateGrid:
        for cellVec in row:
            rows.append(torch.as_tensor(cellVec, dtype = torch.float32))
    
    # Shape = (num_cells, feature_dim)
    return torch.stack(rows, dim = 0)

def fit_gnn(
        x_ped: torch.Tensor,
        x_hazard: torch.Tensor,
        x_infra: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> GNNInput: 
    
    print("Fit GNN running")
    n = x_ped.size(0)
    if batch is None:
        batch = torch.zeros(n, dtype = torch.long, device = x_ped.device)
        
    return GNNInput(x_ped = x_ped, x_hazard = x_hazard, x_infra = x_infra, edge_index = edge_index, batch = batch)

"""
4-neighbor undirected graph for an nx-by-ny grid of cells
node_id = i*ny + j
"""
def grid_edge_index(nx: int, ny: int) -> np.ndarray:
    def nid(i, j): 
        return i * ny + j
    
    edges = []
    for i in range(nx):
        for j in range(ny):
            u = nid(i, j)
            if i + 1 < nx:
                v = nid(i+1, j)
                edges += [(u, v), (v, u)]
            if j + 1 < ny:
                v = nid(i, j + 1)
                edges += [(u, v), (v, u)]
    
    # return shape (2, E)
    return np.array(edges, dtype = np.int64).T

class _MLPBranch(nn.Module):
    """modular layer in NN"""
    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x, *args, **kwargs):
        return self.net(x)
    
class EvacPolicy(nn.Module):
    """define GNN model components"""
    def __init__(self,
        d_ped: int,
        d_hazard: int,
        d_infra: int,
        embed_dim: int = 32,
        heads: int = 2,
        action_dim_shelter: int = 1,
        action_dim_guidance: int = 1,
        force_mlp: bool = True,
        verbose: bool = False
    ):
        attn_heads = 8
        use_attention = False
        d_ped = 2
        d_hazard = 3
        d_infra = 3
        
        # check dimension of action inputs
        print("Model Step 1", flush = True)
        super().__init__()
        
        # if we are in debugging mode, if so, prinout debugging test outputs
        self.verbose = bool(verbose)
        
        # whether to try torch_geometric GATConv
        self.use_pyg = (_HAS_PYG and not force_mlp)
        
        # whether to run multihead self-attention across cells
        self.use_attention = use_attention

        print("Model Step 2", flush = True)
        
        """IMPORTANT: Defines NN layers"""
        """IMPORTANT: Defines the feature extraction layers (use or not use GNN)"""
        if self.use_pyg:
            # GNN definition, need torch_geometric
            self.gnn_ped = GATConv(d_ped, embed_dim, heads = heads, concat = True)
            self.gnn_hazard = GATConv(d_hazard, embed_dim, heads = heads, concat = True)
            self.gnn_inf = GATConv(d_infra, embed_dim, heads = heads, concat = True)
            self._out_dim = embed_dim * heads
        else:
            # MLP fallback: per-node feedforward
            self.gnn_ped = _MLPBranch(d_ped, embed_dim * heads)
            self.gnn_hazard = _MLPBranch(d_hazard, embed_dim * heads)
            self.gnn_inf = _MLPBranch(d_infra, embed_dim * heads)
            self._out_dim = embed_dim * heads
            
        # ped | hazard | infra
        fused_dim = self._out_dim * 3
        
        """Multi-head self-attention"""
        print("Model Step 3", flush = True)
        # optional and explicitly defined head across all cells
        if self.use_attention:
            self.attn = nn.MultiheadAttention(
                embed_dim = fused_dim, 
                num_heads = attn_heads, 
                batch_first = True
            )
        else:
            self.attn = None
        
        """Fully connected layers for processing"""
        self.fc1 = nn.Linear(fused_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        
        """Bi-variate actor layers (shelter, guidance)"""
        self.actor_shelter = nn.Linear(128, action_dim_shelter)
        self.actor_guidance = nn.Linear(128, action_dim_guidance)
        
        """Critic layer"""
        self.critic = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        """Debugging code on NN architecture setup, if use GNN and self-attention"""
        if self.verbose:
            print(
                f"[EvacPolicy::__init__] use_pyg={self.use_pyg} "
                f"use_attention={self.use_attention} "
                f"action_dims=({action_dim_shelter},{action_dim_guidance})"
            )
    
    """feed forward"""
    def forward(self, g: GNNInput):
        """
        Shapes:
        
        g.x_ped      : (N, d_ped)
        g.x_hazard   : (N, d_hazard)
        g.x_infra    : (N, d_infra)
        g.edge_index : (2, E) or None
        g.batch      : (N,) long, graph id per node. MAY be None, in which case we assume a single graph.
        Returns:
            shelter_logits : (B, action_dim_shelter)
            guidance_logits: (B, action_dim_guidance)
            value          : (B,)  predicted state value(s)
        
        """
        
        if self.verbose:
            print("[EvacPolicy.forward] enter")
        
        """Feature Extraction"""
        print("Forward step 1: Feature extraction", flush = True)
        """encode per-cell features"""
        if self.use_pyg:
            if g.edge_index is None:
                raise RuntimeError("edge_index is required when use_pyg = True")
            # dimension = (N, out_dim)
            ped_emb = F.gelu(self.gnn_ped(g.x_ped, g.edge_index))
            # dimension = (N, out_dim)
            hazard_emb = F.gelu(self.gnn_hazard(g.x_hazard, g.edge_index))
            # dimension = (N, out_dim)
            inf_emb = F.gelu(self.gnn_inf(g.x_infra, g.edge_index))
        else:
            # dimension = (N, out_dim)
            ped_emb = F.gelu(self.gnn_ped(g.x_ped))
            # dimension = (N, out_dim)
            hazard_emb = F.gelu(self.gnn_hazard(g.x_hazard))
            # dimension = (N, out_dim)
            inf_emb = F.gelu(self.gnn_inf(g.x_infra))
        
        # (N, 3*out_dim)
        """Feature Concatnation"""
        print("Forward step 2: Feature Concatnation", flush = True)
        # (num_cells, fused_dim)
        node_feat = torch.cat([ped_emb, hazard_emb, inf_emb], dim = -1)
        
        """Pool per graph to get graph_embed"""
        # batch handling
        
        if g.batch is None:
            if self.use_attention:
                # single graph shape: (1, N, F)
                attn_in = node_feat.unsqueeze(0)
                # (1, N, F)
                attn_out, _ = self.attn(attn_in, attn_in, attn_in)
                # global mean pool across tokens, (1, F)
                graph_embed = attn_out.mean(dim=1)
            else:
                # (1, F)
                graph_embed = node_feat.mean(dim = 0, keepdim = True)
        else:
            # mini-batch of graphs packed in one tensor
            B = int(g.batch.max().item() + 1)
            graphs = []
            for b in range(B):
                # shape = (1, N_b, F)
                xb = node_feat[g.batch == b]
                if xb.numel() == 0:
                    graphs.append(torch.zeros((1, node_feat.size(-1)), 
                                              dtype = node_feat.dtype,
                                              device = node_feat.device))
                    continue
                      
                if self.use_attention:
                    attn_in = xb.unsqueeze(0) # (1, Nb, F)
                    ob, _ = self.attn(attn_in, attn_in, attn_in) # (1, Nb, F)
                    graphs.append(ob.mean(dim = 1)) # (1, F)
                else:
                    # (1, F)
                    graphs.append(xb.mean(dim = 0, keepdim = True))
            # shape = (B, F)
            graph_embed = torch.cat(graphs, dim = 0)
                
        print("Forward step 3", flush = True)
        """Actor block"""
        # shape = (B, 256)
        dense = F.gelu(self.fc1(graph_embed))
        # shape = (B, 128)
        dense = F.gelu(self.fc2(dense))
        
        # shape = (B, action_shelter)
        shelter_logits = self.actor_shelter(dense)
        # shape = (B, action_guidance)
        guidance_logits = self.actor_guidance(dense)
        
        """Critic head"""
        print("Forward step 4", flush = True)
        # shape = (B,)
        value = self.critic(graph_embed).squeeze(-1)
        
        if self.verbose:
            print(
                f"[EvacPolicy.forward] B={graph_embed.size(0)} "
                f"shelter_logits.shape={shelter_logits.shape} "
                f"guidance_logits.shape={guidance_logits.shape} "
                f"value.shape={value.shape}"
            )
        
        return shelter_logits, guidance_logits, value
    
