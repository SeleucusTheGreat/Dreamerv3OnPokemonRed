import os
import glob
import copy
import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_optimizer as optim
from torch.distributions import Independent, Normal, kl_divergence, OneHotCategoricalStraightThrough, OneHotCategorical
from torch.distributions.utils import probs_to_logits

IMAGE_SIZE = 64

torch.set_float32_matmul_precision('high')

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

class TwoHotEncoding:
    def __init__(self, min=-10, max=10, num_bins=255, device="cuda"):
        self.device = device
        self.num_bins = num_bins
        # Linear bins in symlog space
        self.bins = torch.linspace(min, max, num_bins, device=device)
        # The actual values the bins represent
        self.bin_values = symexp(self.bins)

    def encode(self, x):
        """Scalar to Two-Hot Vector (Equation 12)"""
        x = symlog(x)
        x = torch.clamp(x, self.bins[0], self.bins[-1])
        
        # Find the indices of the two closest bins
        pos = (x - self.bins[0]) / (self.bins[1] - self.bins[0])
        low = torch.floor(pos).long()
        high = torch.ceil(pos).long()
        
        # Calculate weights
        weight_high = pos - low
        weight_low = 1.0 - weight_high
        
        # Create the one-hot-like vector
        two_hot = torch.zeros(*x.shape, self.num_bins, device=self.device)
        two_hot.scatter_(-1, low.unsqueeze(-1), weight_low.unsqueeze(-1))
        two_hot.scatter_(-1, high.unsqueeze(-1), weight_high.unsqueeze(-1))
        return two_hot

    def decode(self, logits):
        """Logits to Expected Value (Equation 10)"""
        probs = torch.softmax(logits, dim=-1)
        return torch.sum(probs * self.bin_values, dim=-1, keepdim=True)


class GoalEncoder(nn.Module):
    def __init__(self, ram_dim=14, ram_out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ram_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, ram_out_dim),
            nn.LayerNorm(ram_out_dim)
        )

    def forward(self, ram):
        return self.net(ram)


class GoalPredictor(nn.Module):
    def __init__(self, input_size=768, goal_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, goal_dim)
        )
        
    def forward(self, x):
        return self.net(x)


class TeamEncoder(nn.Module):
    def __init__(self, team_dim=6, team_out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(team_dim, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, team_out_dim),
            nn.LayerNorm(team_out_dim)
        )

    def forward(self, team):
        return self.net(team)


class TeamPredictor(nn.Module):
    def __init__(self, input_size=768, team_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, team_dim)
        )

    def forward(self, x):
        return self.net(x)


class ItemEncoder(nn.Module):
    def __init__(self, item_dim=2, item_out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(item_dim, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, item_out_dim),
            nn.LayerNorm(item_out_dim)
        )

    def forward(self, items):
        return self.net(items)


class ItemPredictor(nn.Module):
    def __init__(self, input_size=768, item_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, item_dim)
        )

    def forward(self, x):
        return self.net(x)


class CuriosityPredictor(nn.Module):
    def __init__(self, input_size=768, num_bins=255):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, num_bins)
        )
        # Initialize final layer near zero for a stable initial categorical distribution
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class DynamicDataNormalizer(nn.Module):
    def __init__(self, device, decay=0.99, min_=1.0, percentileLow=0.05, percentileHigh=0.95):
        super().__init__()
        self._decay = decay
        self._min = torch.tensor(min_, device=device)
        self._percentileLow = percentileLow
        self._percentileHigh = percentileHigh
        
        # Track the smoothed range (S) directly as a buffer
        self.register_buffer("S", torch.zeros((), dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        low = torch.quantile(x, self._percentileLow)
        high = torch.quantile(x, self._percentileHigh)
        batch_range = high - low
        self.S.copy_(self._decay * self.S + (1.0 - self._decay) * batch_range)
        denominator = torch.max(self._min, self.S)
        
        return denominator.detach()


class EnsembleRewardPredictor(nn.Module):
    def __init__(self, inputsize, num_bins=255, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            RewardPredictor(inputsize, num_bins) for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.stack([head(x) for head in self.heads], dim=0)


class RandomCNNEncoder(nn.Module):
    def __init__(self, out_dim=8):
        super().__init__()
        # Small CNN to extract spatial and visual patterns
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2, bias=False) # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2, bias=False) # 32x32 -> 16x16
        self.fc = nn.Linear(16 * 16 * 16, out_dim)
        
        # Initialize randomly and freeze
        for param in self.parameters():
            param.requires_grad = False
            nn.init.normal_(param, mean=0.0, std=1.0)
            
    def forward(self, x):
        # Expects x shape: [B, 3, 64, 64] or [3, 64, 64], float in [0.0, 1.0]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            out = torch.tanh(self.fc(x))
        return out


class RandomCNNEncoder(nn.Module):
    def __init__(self, out_dim=8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2, bias=False) # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2, bias=False) # 32x32 -> 16x16
        self.fc = nn.Linear(16 * 16 * 16, out_dim)
        
        for param in self.parameters():
            param.requires_grad = False
            nn.init.normal_(param, mean=0.0, std=1.0)
            
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            out = torch.tanh(self.fc(x))
        return out


class Buffer(object):
    def __init__(self, device, capacity=800000, actionSize=6, ram_dim=8, item_dim=2, team_level_dim=6, num_envs=4, block_size=128):  
        self.device = device
        self.capacity = capacity
        self.num_envs = num_envs
        self.actionSize = actionSize
        self.ram_dim = ram_dim
        self.item_dim = item_dim
        self.team_level_dim = team_level_dim
        
        self.block_size = block_size
        self.num_blocks = capacity // block_size
        self.capacity = self.num_blocks * block_size
        
        # Allocation
        self.observations = torch.empty((self.capacity, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8, device='cpu')
        self.rams = torch.empty((self.capacity, ram_dim), dtype=torch.float32, device='cpu')
        self.item_counts = torch.empty((self.capacity, item_dim), dtype=torch.float32, device='cpu')
        self.team_levels = torch.empty((self.capacity, team_level_dim), dtype=torch.float32, device='cpu')
        self.actions = torch.empty((self.capacity, actionSize), dtype=torch.float32, device='cpu')
        self.rewards = torch.empty((self.capacity, 1), dtype=torch.float32, device='cpu')
        
        # Trajectory Signature Model (CPU-bound)
        self.encoder = RandomCNNEncoder(out_dim=8).to('cpu')
        self.signature_dim = 5 * 8 # 5 frames * 8 features
        
        # Dynamic Vector Database / Cluster Codebook
        self.bucket_centroids = [] # List of unit-length PyTorch tensors of shape [signature_dim]
        self.bucket_counts = []    # List of block counts per bucket
        self.max_buckets = 512     # Hard cap to bound memory/computational scaling
        self.similarity_threshold = 0.85 # Cosine similarity threshold to define a "new category"
        
        # Index Mapping: links each block index to its assigned bucket ID
        self.block_bucket_ids = torch.full((self.num_blocks,), -1, dtype=torch.int32)
        
        # Buffer cursors
        self.step_accumulator = []
        self.block_index = 0
        self.block_full = False

    @property
    def index(self):
        return self.block_index * self.block_size

    @property
    def full(self):
        return self.block_full

    def _compute_trajectory_signature(self, step_accumulator):
        # Sample 5 frames evenly spaced across the 128-step block
        indices = [0, 31, 63, 95, 127]
        feats = []
        for idx in indices:
            obs = step_accumulator[idx][0]
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device='cpu') / 255.0
            with torch.no_grad():
                feat = self.encoder(obs_tensor)
            feats.append(feat)
            
        # Concatenate features into a joint trajectory signature [1, 40]
        signature = torch.cat(feats, dim=1)
        
        # L2-normalize to compute Cosine Similarity via dot products
        norm = torch.norm(signature, p=2, dim=1, keepdim=True) + 1e-8
        return (signature / norm).squeeze(0)

    def _find_or_create_bucket(self, signature):
        # Base case: first block recorded
        if len(self.bucket_centroids) == 0:
            self.bucket_centroids.append(signature.clone())
            self.bucket_counts.append(0) # Will be incremented on allocation
            return 0
            
        # Stack existing bucket centroids: Shape [K, signature_dim]
        centroids_matrix = torch.stack(self.bucket_centroids)
        
        # Compute Cosine Similarities: Shape [K]
        similarities = torch.matmul(centroids_matrix, signature)
        
        max_sim, best_idx = torch.max(similarities, dim=0)
        best_idx = best_idx.item()
        
        if max_sim >= self.similarity_threshold or len(self.bucket_centroids) >= self.max_buckets:
            # Merge block into best-matching bucket and update centroid using running momentum
            momentum = 0.95
            updated_centroid = momentum * self.bucket_centroids[best_idx] + (1.0 - momentum) * signature
            self.bucket_centroids[best_idx] = updated_centroid / (torch.norm(updated_centroid, p=2) + 1e-8)
            return best_idx
        else:
            # Create a brand new trajectory bucket
            new_idx = len(self.bucket_centroids)
            self.bucket_centroids.append(signature.clone())
            self.bucket_counts.append(0)
            return new_idx

    def add(self, observation, ram, item_count, team_level, action, reward):
        self.step_accumulator.append((observation, ram, item_count, team_level, action, reward))
        if len(self.step_accumulator) == self.block_size:
            self._flush_accumulator()

    def _flush_accumulator(self):
        # 1. First, compute the trajectory signature of the incoming sequence
        signature = self._compute_trajectory_signature(self.step_accumulator)
        
        # 2. Determine where we are writing (always the oldest block, self.block_index)
        write_block_idx = self.block_index
        start_idx = write_block_idx * self.block_size
        
        # 3. If full, perform FIFO eviction on the old block at self.block_index
        if self.block_full:
            old_bucket_id = self.block_bucket_ids[write_block_idx].item()
            if old_bucket_id != -1:
                self.bucket_counts[old_bucket_id] -= 1
                # If the bucket is now empty, remove it completely from the codebook
                if self.bucket_counts[old_bucket_id] <= 0:
                    self._prune_bucket(old_bucket_id)
                    
        # 4. Query/Update Vector Database (after the empty bucket cleanup is done)
        bucket_id = self._find_or_create_bucket(signature)
        
        # 5. Write the new sequence data into the pre-allocated CPU tensors
        for offset, t in enumerate(self.step_accumulator):
            idx = start_idx + offset
            self.observations[idx] = torch.as_tensor(t[0], dtype=torch.uint8)
            self.rams[idx] = torch.as_tensor(t[1], dtype=torch.float32)
            self.item_counts[idx] = torch.as_tensor(t[2], dtype=torch.float32)
            self.team_levels[idx] = torch.as_tensor(t[3], dtype=torch.float32)
            self.actions[idx] = torch.as_tensor(t[4], dtype=torch.float32)
            self.rewards[idx] = torch.as_tensor(t[5], dtype=torch.float32)
            
        # 6. Assign the block's new bucket ID and update counts
        self.block_bucket_ids[write_block_idx] = bucket_id
        
        # Ensure bucket_counts has space for the new bucket index
        while len(self.bucket_counts) <= bucket_id:
            self.bucket_counts.append(0)
            
        self.bucket_counts[bucket_id] += 1
        
        # 7. Advance the FIFO cursor index
        self.block_index += 1
        if self.block_index >= self.num_blocks:
            self.block_full = True
            self.block_index = 0
            
        self.step_accumulator.clear()

    def sample(self, batchSize, sequenceSize):
        valid_blocks = self.num_blocks if self.block_full else self.block_index
        if valid_blocks == 0:
            return None

        num_recent = int(round(batchSize * 0.25))
        num_all = batchSize - num_recent

        sample_indices = []

        if num_recent > 0:
            recent_window_size = min(200, valid_blocks)
            if self.block_full:
                block_starts = (self.block_index - torch.randint(1, recent_window_size + 1, (num_recent, 1))) % self.num_blocks
            else:
                block_starts = torch.randint(max(0, valid_blocks - recent_window_size), valid_blocks, (num_recent, 1))
                
            offsets = torch.randint(0, self.block_size - sequenceSize + 1, (num_recent, 1))
            seq_offsets = torch.arange(sequenceSize).reshape(1, -1)
            
            recent_indices = block_starts * self.block_size + offsets + seq_offsets
            sample_indices.append(recent_indices)

        # Rarity-Prioritized Sampling across dynamically generated vector buckets
        if num_all > 0:
            valid_bucket_ids = self.block_bucket_ids[:valid_blocks]
            
            counts_tensor = torch.tensor(self.bucket_counts, dtype=torch.float32)
            counts = counts_tensor[valid_bucket_ids]
            
            weights = 1.0 / (torch.sqrt(counts) + 1e-5)
            
            block_starts = torch.multinomial(weights, num_all, replacement=True).reshape(-1, 1)
            
            offsets = torch.randint(0, self.block_size - sequenceSize + 1, (num_all, 1))
            seq_offsets = torch.arange(sequenceSize).reshape(1, -1)
            
            all_indices = block_starts * self.block_size + offsets + seq_offsets
            sample_indices.append(all_indices)

        sampleIndex = torch.cat(sample_indices, dim=0).long()

        obs_cpu = self.observations[sampleIndex]
        obs_float = obs_cpu.to(self.device, non_blocking=True).float() / 255.0

        sample = {
            "observations": obs_float,
            "rams":         self.rams[sampleIndex].to(self.device, non_blocking=True), 
            "item_counts":  self.item_counts[sampleIndex].to(self.device, non_blocking=True),
            "team_levels":  self.team_levels[sampleIndex].to(self.device, non_blocking=True),
            "actions":      self.actions[sampleIndex].to(self.device, non_blocking=True),
            "rewards":      self.rewards[sampleIndex].to(self.device, non_blocking=True),
            "index":        sampleIndex.to(self.device, non_blocking=True)
        }
        return sample

    def save(self, path):
        valid_limit = self.capacity if self.block_full else self.block_index * self.block_size
        checkpoint = {
            'observations': self.observations[:valid_limit].cpu(),
            'rams': self.rams[:valid_limit].cpu(),
            'item_counts': self.item_counts[:valid_limit].cpu(),
            'team_levels': self.team_levels[:valid_limit].cpu(),
            'actions': self.actions[:valid_limit].cpu(),
            'rewards': self.rewards[:valid_limit].cpu(),
            'bucket_centroids': [c.cpu() for c in self.bucket_centroids],
            'bucket_counts': self.bucket_counts,
            'block_bucket_ids': self.block_bucket_ids.cpu(),
            'block_index': self.block_index,
            'block_full': self.block_full,
            'capacity': self.capacity,
            'block_size': self.block_size
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        
        if 'bucket_centroids' in checkpoint:
            saved_obs = checkpoint['observations']
            valid_limit = saved_obs.shape[0]
            
            self.observations[:valid_limit].copy_(saved_obs)
            self.rams[:valid_limit].copy_(checkpoint['rams'])
            self.item_counts[:valid_limit].copy_(checkpoint['item_counts'])
            self.team_levels[:valid_limit].copy_(checkpoint['team_levels'])
            self.actions[:valid_limit].copy_(checkpoint['actions'])
            self.rewards[:valid_limit].copy_(checkpoint['rewards'])
            
            self.bucket_centroids = checkpoint['bucket_centroids']
            self.bucket_counts = checkpoint['bucket_counts']
            self.block_bucket_ids.copy_(checkpoint['block_bucket_ids'])
            self.block_index = checkpoint['block_index']
            self.block_full = checkpoint['block_full']
            
            print(f"[*] Loaded Dynamic Trajectory Codebook Buffer: {valid_limit // self.block_size} blocks across {len(self.bucket_centroids)} unique buckets.")
        else:
            # Fallback parsing legacy checkpoints: rebuild codebook centroids on the fly
            saved_obs = checkpoint['observations']
            saved_rams = checkpoint['rams']
            saved_item_counts = checkpoint.get('item_counts', torch.zeros((saved_obs.shape[0], 2), dtype=torch.float32))
            saved_team_levels = checkpoint.get('team_levels', torch.zeros((saved_obs.shape[0], 6), dtype=torch.float32))
            saved_actions = checkpoint['actions']
            saved_rewards = checkpoint['rewards']
            
            old_index = checkpoint['index']
            old_full = checkpoint['full']
            old_capacity = saved_obs.shape[0]

            if old_full:
                obs_ordered = torch.cat((saved_obs[old_index:], saved_obs[:old_index]), dim=0)
                rams_ordered = torch.cat((saved_rams[old_index:], saved_rams[:old_index]), dim=0)
                item_ordered = torch.cat((saved_item_counts[old_index:], saved_item_counts[:old_index]), dim=0)
                team_ordered = torch.cat((saved_team_levels[old_index:], saved_team_levels[:old_index]), dim=0)
                actions_ordered = torch.cat((saved_actions[old_index:], saved_actions[:old_index]), dim=0)
                rewards_ordered = torch.cat((saved_rewards[old_index:], saved_rewards[:old_index]), dim=0)
                total_valid = old_capacity
            else:
                obs_ordered = saved_obs[:old_index]
                rams_ordered = saved_rams[:old_index]
                item_ordered = saved_item_counts[:old_index]
                team_ordered = saved_team_levels[:old_index]
                actions_ordered = saved_actions[:old_index]
                rewards_ordered = saved_rewards[:old_index]
                total_valid = old_index

            num_importable_blocks = min(total_valid // self.block_size, self.num_blocks)
            for b in range(num_importable_blocks):
                start = b * self.block_size
                end = start + self.block_size
                
                self.observations[start:end].copy_(obs_ordered[start:end])
                self.rams[start:end].copy_(rams_ordered[start:end])
                self.item_counts[start:end].copy_(item_ordered[start:end])
                self.team_levels[start:end].copy_(team_ordered[start:end])
                self.actions[start:end].copy_(actions_ordered[start:end])
                self.rewards[start:end].copy_(rewards_ordered[start:end])
                
                # Reconstruct temporary block accumulator for signature generation
                temp_accum = []
                for step_idx in range(start, end):
                    temp_accum.append((self.observations[step_idx], self.rams[step_idx], self.item_counts[step_idx], self.team_levels[step_idx], self.actions[step_idx], self.rewards[step_idx]))
                
                signature = self._compute_trajectory_signature(temp_accum)
                bucket_id = self._find_or_create_bucket(signature)
                self.block_bucket_ids[b] = bucket_id
                self.bucket_counts[bucket_id] += 1
                
            self.block_index = num_importable_blocks % self.num_blocks
            self.block_full = (num_importable_blocks == self.num_blocks)
            print(f"[*] Reconstructed codebook from flat checkpoint: {num_importable_blocks} blocks mapped to {len(self.bucket_centroids)} visual categories.")
    
    def print_diagnostics(self):
        """Displays visual cluster distribution, buffer density, and rare state preservation metrics."""
        valid_blocks = self.num_blocks if self.block_full else self.block_index
        total_steps = valid_blocks * self.block_size
        capacity_pct = (valid_blocks / self.num_blocks) * 100
        
        num_buckets = len(self.bucket_centroids)
        if num_buckets == 0:
            print("\n" + "="*60)
            print("  [Buffer Diagnostics] Replay Buffer is currently EMPTY.")
            print("="*60 + "\n")
            return
            
        counts_tensor = torch.tensor(self.bucket_counts, dtype=torch.float32)
        
        # Calculate cluster density statistics
        total_assigned_blocks = int(counts_tensor.sum().item())
        avg_blocks_per_bucket = counts_tensor.mean().item()
        
        # Count empty and active buckets
        non_empty_buckets = int((counts_tensor > 0).sum().item())
        empty_buckets = num_buckets - non_empty_buckets
        
        # Sort bucket sizes
        sorted_counts, sorted_indices = torch.sort(counts_tensor, descending=True)
        
        print("\n" + "="*60)
        print("          DIVERSITY-BALANCED REPLAY BUFFER DIAGNOSTICS")
        print("="*60)
        print(f"  Buffer Fill Ratio       : {total_steps:,} / {self.capacity:,} steps ({capacity_pct:.2f}%)")
        print(f"  Stored Blocks           : {valid_blocks:,} / {self.num_blocks:,} blocks")
        print(f"  Active Vector Buckets   : {num_buckets} / {self.max_buckets} categories")
        print(f"  Populated Categories    : {non_empty_buckets} (Pruned/Empty: {empty_buckets})")
        print(f"  Average Blocks/Category : {avg_blocks_per_bucket:.2f}")
        print("-"*60)
        
        # Top 5 largest categories (visual sequences currently dominating the environment)
        print("  Top 5 Largest Categories (Redundant visual states):")
        top_k = min(5, num_buckets)
        for i in range(top_k):
            b_id = sorted_indices[i].item()
            b_cnt = int(sorted_counts[i].item())
            b_pct = (b_cnt / max(1, valid_blocks)) * 100
            print(f"    - Category #{b_id:03d} : {b_cnt:4d} blocks ({b_pct:5.2f}% of buffer footprint)")
            
        print("-"*60)
        
        # Bottom 5 smallest active categories (visually unique states saved from eviction)
        print("  Bottom 5 Smallest Active Categories (Rare/Protected states):")
        active_indices = sorted_indices[sorted_counts > 0]
        active_counts = sorted_counts[sorted_counts > 0]
        bot_k = min(5, len(active_counts))
        for i in range(bot_k):
            idx_from_end = -(i + 1)
            b_id = active_indices[idx_from_end].item()
            b_cnt = int(active_counts[idx_from_end].item())
            b_pct = (b_cnt / max(1, valid_blocks)) * 100
            print(f"    - Category #{b_id:03d} : {b_cnt:4d} blocks ({b_pct:5.2f}% of buffer footprint)")
            
        print("="*60 + "\n")

    def _prune_bucket(self, bucket_id):
        # 1. Remove the empty centroid from our codebook
        self.bucket_centroids.pop(bucket_id)
        # 2. Remove the count tracker from our list
        self.bucket_counts.pop(bucket_id)
        
        # 3. For any block that pointed to this deleted bucket, reset its index to -1
        mask_same = (self.block_bucket_ids == bucket_id)
        self.block_bucket_ids[mask_same] = -1
        
        # 4. Any block pointing to a bucket index greater than bucket_id shifts down by 1
        mask_greater = (self.block_bucket_ids > bucket_id)
        self.block_bucket_ids[mask_greater] -= 1


class RecurrentModel(nn.Module):
    def __init__(self, recurrentSize=4096, latentSize=64*64, actionSize=3):
        super().__init__()
        self.recurrentSize = recurrentSize
        self.latentSize = latentSize
        self.actionSize = actionSize
        self.linear = nn.Linear(latentSize + actionSize, recurrentSize)
        self.norm = nn.LayerNorm(recurrentSize)
        self.act = nn.SiLU()
        self.recurrent = nn.GRUCell(recurrentSize, recurrentSize)

    def forward(self, PreviousRecurrentState, PreviousLatentState, PreviousAction):
        fullstate = torch.cat((PreviousLatentState, PreviousAction), -1) 
        fullstate = self.act(self.norm(self.linear(fullstate)))         
        return self.recurrent(fullstate, PreviousRecurrentState)  


class PriorNet(nn.Module):
    def __init__(self, recurrentSize=512, rows=16, cols=16):
        super().__init__()
        self.recurrentSize = recurrentSize
        self.latentSize = rows * cols
        self.rows = rows
        self.cols = cols
        self.trasform = nn.Sequential(
            nn.Linear(recurrentSize, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, rows * cols)
        )

    def forward(self, RecurrentState):
        rawLogits = self.trasform(RecurrentState)
        rawProbabilities = rawLogits.view(-1, self.rows, self.cols).softmax(-1)

        confusion = torch.ones_like(rawProbabilities) / self.cols 
        probabilities = (0.99) * rawProbabilities + 0.01 * confusion 

        logits = probs_to_logits(probabilities) 
        distribution = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)  
        sample = distribution.rsample().view(-1, self.latentSize) 

        return sample, logits


class PosteriorNet(nn.Module):
    def __init__(self, inputSize=1536, rows=16, cols=16):
        super().__init__()
        self.inputSize = inputSize
        self.rows = rows
        self.cols = cols
        self.latentSize = rows * cols
        self.trasform = nn.Sequential(
            nn.Linear(inputSize, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, self.latentSize)
        )
    def forward(self, InputState):
        rawLogits = self.trasform(InputState)
        rawProbabilities = rawLogits.view(-1, self.rows, self.cols).softmax(-1) 

        confusion = torch.ones_like(rawProbabilities) / self.cols 
        probabilities = (0.99) * rawProbabilities + 0.01 * confusion 

        logits = probs_to_logits(probabilities) 
        distribution = Independent(OneHotCategoricalStraightThrough(logits=logits, validate_args=False), 1)  
        sample = distribution.rsample().view(-1, self.rows * self.cols) 

        return sample, logits 


class RewardPredictor(nn.Module):
    def __init__(self, inputsize, num_bins=255):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(inputsize, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, num_bins)
        )
        nn.init.zeros_(self.transform[-1].weight)
        nn.init.zeros_(self.transform[-1].bias)

    def forward(self, x):
        return self.transform(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels), # LayerNorm equivalent for images
            nn.ELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels)
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderImage(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), output_size=1024, depth=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, depth, kernel_size=4, stride=2, padding=1), # 64x64 -> 32x32
            nn.ELU(),
            ResBlock(depth),
            nn.Conv2d(depth, depth*2, kernel_size=4, stride=2, padding=1), # 32x32 -> 16x16
            nn.ELU(),
            ResBlock(depth*2),
            nn.Conv2d(depth*2, depth*4, kernel_size=4, stride=2, padding=1), # 16x16 -> 8x8
            nn.ELU(),
            ResBlock(depth*4),
            nn.Conv2d(depth*4, depth*8, kernel_size=4, stride=2, padding=1), # 8x8 -> 4x4
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(depth*8 * 4 * 4, output_size), 
            nn.LayerNorm(output_size)
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, input_size=768, depth=32):
        super().__init__()
        self.depth = depth
        self.linear = nn.Linear(input_size, depth * 8 * 4 * 4) 
        
        self.net = nn.Sequential(
            ResBlock(depth * 8),
            nn.ConvTranspose2d(depth * 8, depth * 4, 4, stride=2, padding=1), # 4x4 -> 8x8
            nn.ELU(),
            ResBlock(depth * 4),
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, stride=2, padding=1), # 8x8 -> 16x16
            nn.ELU(),
            ResBlock(depth * 2),
            nn.ConvTranspose2d(depth * 2, depth, 4, stride=2, padding=1),     # 16x16 -> 32x32
            nn.ELU(),
            nn.ConvTranspose2d(depth, 3, 4, stride=2, padding=1)              # 32x32 -> 64x64
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.depth * 8, 4, 4) 
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, action_dim, device, concatenated_dim=768):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.concatenated_dim = concatenated_dim
        
        self.net = nn.Sequential(
            nn.Linear(concatenated_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, action_dim)
        )
        
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, x):
        raw_logits = self.net(x)
        probs = torch.softmax(raw_logits, dim=-1)
        uniform = torch.ones_like(probs) / self.action_dim
        mixed_probs = 0.99 * probs + 0.01 * uniform
        dist = OneHotCategorical(probs=mixed_probs)
        action_onehot = dist.sample()
        logprobs = dist.log_prob(action_onehot)
        entropy = dist.entropy()

        return action_onehot, logprobs, entropy


class Critic(nn.Module):
    def __init__(self, inputSize, bins=255):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(inputSize, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, bins)
        )
        nn.init.zeros_(self.transform[-1].weight)
        nn.init.zeros_(self.transform[-1].bias)

    def forward(self, x):
        return self.transform(x)


class Dreamer:
    def __init__(self,device, obs_dim,envs, 
                action_dim=6, recurrent_dim=512, rows=64, cols=64, latent_dim=256, 
                number_of_sequences=32, steps_per_sequence=64, seed=42, buffer_size=10000, 
                ram_dim=14, team_dim=6, item_dim=2, curiosity_scale=0.05):

        # Data
        self.device = device
        self.action_dim = action_dim
        self.recurrent_dim = recurrent_dim 
        self.rows = rows
        self.cols = cols
        self.latent_dim = rows*cols 
        self.concatenated_dim = recurrent_dim + self.latent_dim 
        self.total_num_episodes = 0
        self.total_num_steps = 0
        self.total_num_updates = 0
        
        self.ram_out_dim = 256
        self.team_out_dim = 128
        self.item_out_dim = 128
        self.enconder_output_size = 1024 + self.ram_out_dim + self.team_out_dim + self.item_out_dim
        
        self.entropy_scale = 0.001
        self.seed = seed
        self.number_of_sequences = number_of_sequences 
        self.steps_per_sequence = steps_per_sequence
        self.buffer_capacity = buffer_size
        self.dynamicDataNormalizer = DynamicDataNormalizer(self.device)
        self.curiosityDynamicDataNormalizer = DynamicDataNormalizer(self.device)
        self.curiosity_scale = curiosity_scale
        self.envs = envs 
        self.scaler = torch.amp.GradScaler('cuda')

        # Models
        self.two_hot = TwoHotEncoding(device=self.device)
        self.recurrentModel = RecurrentModel(self.recurrent_dim, self.latent_dim, self.action_dim).to(self.device)
        self.posteriorNet = PosteriorNet(self.enconder_output_size + self.recurrent_dim, self.rows, self.cols).to(self.device) 
        self.priorNet = PriorNet(self.recurrent_dim,self.rows,self.cols).to(self.device)
        self.rewardPredictor = EnsembleRewardPredictor(self.concatenated_dim, num_heads=3).to(self.device)
        self.curiosityPredictor = CuriosityPredictor(self.concatenated_dim).to(self.device)
        self.projector = nn.Linear(self.concatenated_dim, 1024).to(self.device) 
        self.actor = Actor(self.action_dim, self.device, self.concatenated_dim).to(self.device)
        self.decoder = Decoder(input_size=self.concatenated_dim).to(self.device)
        self.decoderOptimizer = torch.optim.Adam(self.decoder.parameters(), lr=2e-4)
        
        # Reward Critic
        self.critic = Critic(self.concatenated_dim).to(self.device)
        self.ema_critic = copy.deepcopy(self.critic)
        for param in self.ema_critic.parameters():
            param.requires_grad = False
        
        # Curiosity Critic
        self.curiosity_critic = Critic(self.concatenated_dim).to(self.device)
        self.ema_curiosity_critic = copy.deepcopy(self.curiosity_critic)
        for param in self.ema_curiosity_critic.parameters():
            param.requires_grad = False
            
        self.critic_ema_decay = 0.98
        self.image_encoder = EncoderImage().to(self.device)
        
        self.goal_encoder = GoalEncoder(ram_dim=ram_dim, ram_out_dim=self.ram_out_dim).to(self.device)
        self.team_encoder = TeamEncoder(team_dim=team_dim, team_out_dim=self.team_out_dim).to(self.device)
        self.item_encoder = ItemEncoder(item_dim=item_dim, item_out_dim=self.item_out_dim).to(self.device)
        
        self.goalPredictor = GoalPredictor(self.concatenated_dim, goal_dim=ram_dim).to(self.device)
        self.teamPredictor = TeamPredictor(self.concatenated_dim, team_dim=team_dim).to(self.device)
        self.itemPredictor = ItemPredictor(self.concatenated_dim, item_dim=item_dim).to(self.device)

        # Buffer
        self.buffer = Buffer(
            device=self.device,
            capacity=self.buffer_capacity,
            actionSize=self.action_dim,
            ram_dim=ram_dim,
            item_dim=item_dim,
            team_level_dim=team_dim
        )

        # WorldModelParameters (Includes Curiosity Predictor)
        self.worldModelParameters = (
            list(self.recurrentModel.parameters()) + 
            list(self.posteriorNet.parameters()) + 
            list(self.priorNet.parameters()) + 
            list(self.rewardPredictor.parameters()) + 
            list(self.image_encoder.parameters()) +   
            list(self.goal_encoder.parameters()) +    
            list(self.team_encoder.parameters()) +    
            list(self.item_encoder.parameters()) +    
            list(self.projector.parameters()) + # Swapped from self.decoder
            list(self.goalPredictor.parameters()) +
            list(self.teamPredictor.parameters()) +
            list(self.itemPredictor.parameters()) +
            list(self.curiosityPredictor.parameters())
        )

        # Optimizers
        self.worldModelOptimizer = torch.optim.Adam(self.worldModelParameters, lr=2e-4) 
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=4e-5)      
        self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)    
        self.curiosityCriticOptimizer = torch.optim.Adam(self.curiosity_critic.parameters(), lr=1e-4)

        # Statistics
        self.num_episodes = 0
        self.num_steps = 0
        self.num_updates = 0

    def sample_batch(self, batchSize, sequenceSize):
        return self.buffer.sample(batchSize, sequenceSize)

    def computeLambdaValues(self, rewards, values, continues, lambda_=0.95):
        returns = torch.zeros_like(rewards)
        bootstrap = values[:, -1]
        for i in reversed(range(rewards.shape[-1])):
            returns[:, i] = rewards[:, i] + continues[:, i] * ((1 - lambda_) * values[:, i+1] + lambda_ * bootstrap) 
            bootstrap = returns[:, i]
        return returns

    def TrainWorldModel(self, batch_data):
        self.worldModelOptimizer.zero_grad(set_to_none=True)
        
        obs_flat = batch_data["observations"].flatten(0, 1)
        ram_flat = batch_data["rams"].flatten(0, 1)
        team_flat = batch_data["team_levels"].flatten(0, 1)
        item_flat = batch_data["item_counts"].flatten(0, 1)
        
        encoded_images = self.image_encoder(obs_flat).view(self.number_of_sequences, self.steps_per_sequence, -1)
        encoded_goals = self.goal_encoder(ram_flat).view(self.number_of_sequences, self.steps_per_sequence, -1)   
        encoded_team = self.team_encoder(team_flat / 100.0).view(self.number_of_sequences, self.steps_per_sequence, -1)
        encoded_items = self.item_encoder(item_flat / 10.0).view(self.number_of_sequences, self.steps_per_sequence, -1)
        
        previous_recurrent_state = torch.zeros(self.number_of_sequences, self.recurrent_dim, device=self.device) 
        previous_latent_state = torch.zeros(self.number_of_sequences, self.latent_dim, device=self.device) 
        
        recurrent_states = []
        priors_logits =[]
        posteriors_logits = []
        posteriors =[]

        for t in range(1, self.steps_per_sequence): 
            recurrent_state = self.recurrentModel(previous_recurrent_state, previous_latent_state, batch_data["actions"][:, t-1])
            _ , prior_logits = self.priorNet(recurrent_state) 
            posterior_input = torch.cat((recurrent_state, encoded_images[:, t], encoded_goals[:, t], encoded_team[:, t], encoded_items[:, t]), dim=-1)
            posterior, posterior_logits = self.posteriorNet(posterior_input)

            recurrent_states.append(recurrent_state)
            priors_logits.append(prior_logits)
            posteriors_logits.append(posterior_logits)
            posteriors.append(posterior)

            previous_recurrent_state = recurrent_state
            previous_latent_state = posterior
        
        recurrent_states = torch.stack(recurrent_states, dim=1) 
        priors_logits = torch.stack(priors_logits, dim=1) 
        posteriors_logits = torch.stack(posteriors_logits, dim=1) 
        posteriors = torch.stack(posteriors, dim=1) 
        full_states = torch.cat((recurrent_states, posteriors), -1) 

        # ----------------------------------------------------
        # R2-DREAMER: BARLOW TWINS REPRESENTATION ALIGNMENT
        # ----------------------------------------------------
        flat_states = full_states.view(-1, self.concatenated_dim)
        
        # Project recurrent state-space model's composite states: shape [B * (T-1), 1024]
        k = self.projector(flat_states) 
        
        # Retrieve target image encoder outputs for steps t=1..T-1 and detach: shape [B * (T-1), 1024]
        e = encoded_images[:, 1:].reshape(-1, 1024).detach()

        # Normalize across the batch dimension
        k_mean = k.mean(dim=0)
        k_std = k.std(dim=0) + 1e-5
        k_norm = (k - k_mean) / k_std

        e_mean = e.mean(dim=0)
        e_std = e.std(dim=0) + 1e-5
        e_norm = (e - e_mean) / e_std

        # Compute cross-correlation matrix
        N_samples = k.size(0)
        C = (k_norm.T @ e_norm) / N_samples

        # Barlow Twins Loss Components
        invariance_loss = ((torch.diagonal(C) - 1) ** 2).sum()
        off_diag = C.clone()
        off_diag.fill_diagonal_(0)
        redundancy_loss = (off_diag ** 2).sum()

        alpha = 5e-4       # Redundancy loss scale 
        beta_BT = 0.05     # Barlow Twins overall loss scale 
        
        bt_loss = invariance_loss + alpha * redundancy_loss
        scaled_bt_loss = beta_BT * bt_loss

       # ----------------------------------------------------
        # LATENT CURIOSITY TARGET (Cosine Distance)
        # ----------------------------------------------------
        # Normalized step-wise cosine distance [0.0, 2.0]
        k_unit = F.normalize(k, p=2, dim=-1)
        e_unit = F.normalize(e, p=2, dim=-1)
        step_bt_error = 1.0 - (k_unit * e_unit).sum(dim=-1)
        step_bt_error = step_bt_error.view(self.number_of_sequences, self.steps_per_sequence - 1)


        with torch.no_grad():
            # Find the 20th percentile error in the batch to act as the explored baseline
            baseline = torch.quantile(step_bt_error, 0.95)
            
            # Subtract baseline and clamp to 0.0
            rectified_error = torch.clamp(step_bt_error - baseline, min=0.0)
            transformed_error = rectified_error * 0.2

        # Train Curiosity Predictor on transformed latent prediction error
        pred_curiosity_logits = self.curiosityPredictor(full_states.detach())
        with torch.no_grad():
            target_curiosity = self.two_hot.encode(transformed_error.detach())
            
        curiosity_loss = -torch.mean(torch.sum(target_curiosity * torch.log_softmax(pred_curiosity_logits, dim=-1), dim=-1)) * 100.0

        reward_logits_ensemble = self.rewardPredictor(full_states) # [num_heads, B, T-1, num_bins]
        
        with torch.no_grad():
            target_rewards = self.two_hot.encode(batch_data["rewards"][:, :-1].squeeze(-1)) # [B, T-1, num_bins]
        
        # Calculate cross-entropy loss averaged over all ensemble heads
        reward_loss = 0.0
        for h in range(self.rewardPredictor.num_heads):
            logits = reward_logits_ensemble[h]
            reward_loss += -torch.mean(torch.sum(target_rewards * torch.log_softmax(logits, dim=-1), dim=-1)) * 100
        reward_loss = reward_loss / self.rewardPredictor.num_heads 

        # kl loss
        prior_distribution = Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1)
        prior_distribution_SG = Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1)
        posterior_distribution = Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1)
        posterior_distribution_SG = Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1)

        # goal loss
        pred_goal = self.goalPredictor(full_states)
        target_goal = batch_data["rams"][:, 1:]
        goal_loss = F.binary_cross_entropy_with_logits(pred_goal, target_goal)*100

        # team level loss (predicting scaled level targets [0, 1])
        pred_team = self.teamPredictor(full_states)
        target_team = batch_data["team_levels"][:, 1:] / 100.0
        team_loss = F.mse_loss(pred_team, target_team) * 100.0

        # item count loss (predicting scaled count targets [0, 1])
        pred_items = self.itemPredictor(full_states)
        target_items = batch_data["item_counts"][:, 1:] / 10.0
        item_loss = F.mse_loss(pred_items, target_items) * 100.0

        # kl loss
        prior_loss = kl_divergence(posterior_distribution_SG, prior_distribution)
        posterior_loss = kl_divergence(posterior_distribution, prior_distribution_SG)
        freeNats = torch.full_like(prior_loss, 1)
        prior_loss = 1 * torch.maximum(prior_loss, freeNats)
        posterior_loss = 0.1 * torch.maximum(posterior_loss, freeNats)
        kl_loss = (prior_loss + posterior_loss).mean()

        # TOTAL LOSS (Includes Curiosity Predictor loss)
        world_model_loss = scaled_bt_loss + kl_loss + goal_loss + reward_loss + team_loss + item_loss + curiosity_loss

        # Standard FP32 Backward pass
        world_model_loss.backward()
        nn.utils.clip_grad_norm_(self.worldModelParameters, 10, norm_type=2)
        self.worldModelOptimizer.step()


        # --- SEPARATE VISUALIZATION DECODER TRAINING (Detached gradients) ---
        self.decoderOptimizer.zero_grad(set_to_none=True)
        detached_states = full_states.detach().view(-1, self.concatenated_dim)
        recon_imgs = self.decoder(detached_states)
        target_imgs = batch_data["observations"][:, 1:].flatten(0, 1)
        decoder_loss = F.mse_loss(recon_imgs, target_imgs)
        decoder_loss.backward()
        self.decoderOptimizer.step()


        metrics = {
            "world_model_loss": world_model_loss.item(),
            "reconstruction_loss": scaled_bt_loss.item(), 
            "reward_loss": reward_loss.item(), 
            "kl_loss": kl_loss.item(),
            "goal_loss": goal_loss.item(),
            "team_loss": team_loss.item(),
            "item_loss": item_loss.item(),
            "curiosity_loss": curiosity_loss.item(),
            "decoder_loss": decoder_loss.item()
        }

        return full_states.view(-1, self.concatenated_dim).detach(), metrics
    
    def Dream(self, full_state, batch_data=None, horizon=25):
        self.actorOptimizer.zero_grad(set_to_none=True)
        self.criticOptimizer.zero_grad(set_to_none=True)
        self.curiosityCriticOptimizer.zero_grad(set_to_none=True)

        full_states = [full_state.detach()]
        log_probabilities = []
        entropies = []
        actions_stack = [] 

        curr_state = full_state.detach()
        recurrent_state, latent_state = torch.split(curr_state, [self.recurrent_dim, self.latent_dim], -1)

        # --- IMAGINATION LOOP ---
        for i in range(horizon):
            action, logprob, entropy = self.actor(curr_state)
            with torch.no_grad():
                recurrent_state = self.recurrentModel(recurrent_state, latent_state, action)
                latent_state, _ = self.priorNet(recurrent_state)
            
            curr_state = torch.cat((recurrent_state, latent_state), -1) 
            
            full_states.append(curr_state)
            log_probabilities.append(logprob)
            entropies.append(entropy)
            actions_stack.append(action) 

        full_states = torch.stack(full_states, dim=1)                 
        log_probabilities = torch.stack(log_probabilities, dim=1)     
        entropies = torch.stack(entropies, dim=1)                     
        actions_stack = torch.stack(actions_stack, dim=1)             

        # Predict rewards using the predicted and encoded goals
        with torch.no_grad():
            imagined_steps = full_states[:, 1:]  # [B, H, concatenated_dim]
            reward_logits_ensemble = self.rewardPredictor(imagined_steps) # [num_heads, B, H, num_bins]
            
            # Decode values for each head
            decoded_rewards = []
            for h in range(self.rewardPredictor.num_heads):
                logits = reward_logits_ensemble[h]
                decoded_rewards.append(self.two_hot.decode(logits).squeeze(-1))
            decoded_rewards = torch.stack(decoded_rewards, dim=0) # [num_heads, B, H]
            mean_rewards = decoded_rewards.mean(dim=0)
            predicted_rewards = mean_rewards

        # Predict curiosity values for imagined steps
        with torch.no_grad():
            curiosity_logits = self.curiosityPredictor(imagined_steps)
            predicted_curiosity = self.two_hot.decode(curiosity_logits).squeeze(-1) # [B, H]

        imagined_states = full_states.detach()
        
        # Extrinsic reward values
        critic_logits_all = self.critic(imagined_states)
        online_values = self.two_hot.decode(critic_logits_all).squeeze(-1) # [B, H+1]

        # Curiosity intrinsic reward values
        curiosity_critic_logits_all = self.curiosity_critic(imagined_states)
        online_curiosity_values = self.two_hot.decode(curiosity_critic_logits_all).squeeze(-1) # [B, H+1]

        # Lambda values 
        with torch.no_grad():
            continues = torch.full_like(predicted_rewards, 0.999) 
            lambda_values = self.computeLambdaValues(predicted_rewards, online_values, continues) # [B, H]
            curiosity_lambda_values = self.computeLambdaValues(predicted_curiosity, online_curiosity_values, continues) # [B, H]
        
        # Advantages
        denominator = self.dynamicDataNormalizer(lambda_values)
        advantages = (lambda_values - online_values[:, :-1].detach()) / denominator
        
        curiosity_denominator = self.curiosityDynamicDataNormalizer(curiosity_lambda_values)
        curiosity_advantages = (curiosity_lambda_values - online_curiosity_values[:, :-1].detach()) / curiosity_denominator
        
        # Combine extrinsic and intrinsic exploration advantages
        combined_advantages = advantages  #+ self.curiosity_scale * curiosity_advantages

        # --- ACTOR LOSS ---
        actor_loss_per_step = combined_advantages.detach() * log_probabilities + self.entropy_scale * entropies
        actor_loss = -torch.mean(torch.mean(actor_loss_per_step, dim=1))

        # --- REWARD CRITIC LOSS ---
        critic_logits_to_train = critic_logits_all[:, :-1]
        
        # Two-hot encode the lambda values as target
        target_values_two_hot = self.two_hot.encode(lambda_values.detach())
        
        # Cross entropy loss between predicted logits and two-hot encoded lambda returns
        critic_loss_main = -torch.mean(
            torch.sum(target_values_two_hot * torch.log_softmax(critic_logits_to_train, dim=-1), dim=-1)
        )

        # --- EMA REGULARIZER ---
        with torch.no_grad():
            ema_critic_logits = self.ema_critic(imagined_states[:, :-1])
            ema_probs = torch.softmax(ema_critic_logits, dim=-1)
            
        # KL divergence between EMA distribution (Anchor) and Online distribution
        critic_log_probs = torch.log_softmax(critic_logits_to_train, dim=-1)
        ema_log_probs = torch.log_softmax(ema_critic_logits, dim=-1)
        
        # KL(P || Q) = sum(P * (log P - log Q))
        critic_ema_reg = torch.mean(
            torch.sum(ema_probs * (ema_log_probs - critic_log_probs), dim=-1)
        )

        # Total Critic Loss
        critic_loss = critic_loss_main  + critic_ema_reg

        # --- CURIOSITY CRITIC LOSS ---
        curiosity_critic_logits_to_train = curiosity_critic_logits_all[:, :-1]
        target_curiosity_values_two_hot = self.two_hot.encode(curiosity_lambda_values.detach())
        
        curiosity_critic_loss_main = -torch.mean(
            torch.sum(target_curiosity_values_two_hot * torch.log_softmax(curiosity_critic_logits_to_train, dim=-1), dim=-1)
        )

        # --- CURIOSITY EMA REGULARIZER ---
        with torch.no_grad():
            ema_curiosity_critic_logits = self.ema_curiosity_critic(imagined_states[:, :-1])
            ema_curiosity_probs = torch.softmax(ema_curiosity_critic_logits, dim=-1)
            
        curiosity_critic_log_probs = torch.log_softmax(curiosity_critic_logits_to_train, dim=-1)
        ema_curiosity_log_probs = torch.log_softmax(ema_curiosity_critic_logits, dim=-1)
        
        curiosity_critic_ema_reg = torch.mean(
            torch.sum(ema_curiosity_probs * (ema_curiosity_log_probs - curiosity_critic_log_probs), dim=-1)
        )
        
        curiosity_critic_loss = curiosity_critic_loss_main + curiosity_critic_ema_reg

        # Update Reward Critic
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1, norm_type=2) 
        self.criticOptimizer.step()

        # Update Curiosity Critic
        curiosity_critic_loss.backward()
        nn.utils.clip_grad_norm_(self.curiosity_critic.parameters(), 1, norm_type=2)
        self.curiosityCriticOptimizer.step()

        # Update Actor Second
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1, norm_type=2) 
        self.actorOptimizer.step()

        # Update EMA critic parameters
        with torch.no_grad():
            for param, ema_param in zip(self.critic.parameters(), self.ema_critic.parameters()):
                ema_param.data.copy_(self.critic_ema_decay * ema_param.data + (1.0 - self.critic_ema_decay) * param.data)
            for param, ema_param in zip(self.curiosity_critic.parameters(), self.ema_curiosity_critic.parameters()):
                ema_param.data.copy_(self.critic_ema_decay * ema_param.data + (1.0 - self.critic_ema_decay) * param.data)

        # --- METRICS COLLECTION ---
        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "curiosity_critic_loss": curiosity_critic_loss.item(),
            "entropies": entropies.mean().item(),
            "log_probabilities": log_probabilities.mean().item(),
            "advantages": advantages.mean().item(),
            "curiosity_advantages": curiosity_advantages.mean().item(),
            "critic_values": online_values.mean().item(), 
            "curiosity_critic_values": online_curiosity_values.mean().item(),
        }

        # Select the trajectory that exceeds the expectation of both reward and exploration critics
        trajectory_advantages = combined_advantages.sum(dim=1) 
        
        best_dream_idx = torch.argmax(trajectory_advantages).item()
        best_dream_data = (
            trajectory_advantages[best_dream_idx].item(),  
            full_states[best_dream_idx].detach().cpu(),
            predicted_rewards[best_dream_idx].detach().cpu(),
            lambda_values[best_dream_idx].detach().cpu(),       
            actions_stack[best_dream_idx].detach().cpu(),
            combined_advantages[best_dream_idx].detach().cpu(),
            "MaxAdv"                                        
        )

        rand_dream_idx = torch.randint(0, trajectory_advantages.shape[0], (1,)).item()
        rand_dream_data = (
            trajectory_advantages[rand_dream_idx].item(), 
            full_states[rand_dream_idx].detach().cpu(),
            predicted_rewards[rand_dream_idx].detach().cpu(),
            lambda_values[rand_dream_idx].detach().cpu(),       
            actions_stack[rand_dream_idx].detach().cpu(),
            combined_advantages[rand_dream_idx].detach().cpu(),
            "Random"
        )
        
        # Track the trajectory with the highest predicted curiosity
        trajectory_curiosity = predicted_curiosity.sum(dim=1)
        cur_dream_idx = torch.argmax(trajectory_curiosity).item()
        cur_dream_data = (
            trajectory_curiosity[cur_dream_idx].item(), # Using total curiosity as the sorting metric
            full_states[cur_dream_idx].detach().cpu(),
            predicted_rewards[cur_dream_idx].detach().cpu(),
            lambda_values[cur_dream_idx].detach().cpu(),       
            actions_stack[cur_dream_idx].detach().cpu(),
            combined_advantages[cur_dream_idx].detach().cpu(),
            "MaxCuriosity"
        )
        
        return metrics, best_dream_data, rand_dream_data, cur_dream_data

    @torch.no_grad()
    def Play_the_game(self, number_of_episodes_per_env=1, epsilon=0.05):
        num_envs = len(self.envs)
        episodes_completed = [0] * num_envs
        scores =[]
        current_rewards = [0.0] * num_envs

        local_buffers = [[] for _ in range(num_envs)]

        recurrent_state = torch.zeros((num_envs, self.recurrent_dim), device=self.device)
        latent_state = torch.zeros((num_envs, self.latent_dim), device=self.device)
        action = torch.zeros((num_envs, self.action_dim), device=self.device)

        observations = []
        rams = []
        team_levels = []
        item_counts = []

        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            rams.append(np.array(info["milestones"], dtype=np.float32))
            team_levels.append(np.array(info["team_levels"], dtype=np.float32))
            item_counts.append(np.array(info["item_counts"], dtype=np.float32))

        while min(episodes_completed) < number_of_episodes_per_env:
            
            obs_tensor = (torch.from_numpy(np.array(observations)).float() / 255.0).to(self.device)
            ram_tensor = torch.from_numpy(np.array(rams)).float().to(self.device)
            team_tensor = torch.from_numpy(np.array(team_levels)).float().to(self.device)
            item_tensor = torch.from_numpy(np.array(item_counts)).float().to(self.device)

            encoded_img = self.image_encoder(obs_tensor)
            encoded_goal = self.goal_encoder(ram_tensor) 
            encoded_team = self.team_encoder(team_tensor / 100.0)
            encoded_item = self.item_encoder(item_tensor / 10.0)

            recurrent_state = self.recurrentModel(recurrent_state, latent_state, action)
            posterior_input = torch.cat((recurrent_state, encoded_img, encoded_goal, encoded_team, encoded_item), -1)
            latent_state, _ = self.posteriorNet(posterior_input)

            action_onehot, _, _ = self.actor(torch.cat((recurrent_state, latent_state), -1))
            action = action_onehot
            
            action_idxs = torch.argmax(action, dim=-1).cpu().numpy()
            actions_for_buffer = action.cpu().numpy().astype(np.float32)

            for i, env in enumerate(self.envs):
                if episodes_completed[i] < number_of_episodes_per_env:
                    
                    next_observation, reward, terminated, truncated, next_info = env.step(action_idxs[i])
                    done = terminated or truncated

                    self.total_num_steps += 1
                    current_rewards[i] += reward
                    next_ram = np.array(next_info["milestones"], dtype=np.float32)
                    next_team = np.array(next_info["team_levels"], dtype=np.float32)
                    next_item = np.array(next_info["item_counts"], dtype=np.float32)

                    local_buffers[i].append((
                        observations[i].copy(), 
                        rams[i].copy(), 
                        item_counts[i].copy(),   # Moved here (size 2)
                        team_levels[i].copy(),   # Moved here (size 6)
                        actions_for_buffer[i].copy(), 
                        reward
                    ))

                    observations[i] = next_observation
                    rams[i] = next_ram
                    team_levels[i] = next_team
                    item_counts[i] = next_item

                    if done:
                        episode_reward = current_rewards[i] 
                        
                        # Sequential additions are handled automatically by the block buffer
                        for transition in local_buffers[i]:
                            self.buffer.add(*transition)

                        local_buffers[i].clear()

                        scores.append(episode_reward)
                        self.total_num_episodes += 1
                        episodes_completed[i] += 1

                        if episodes_completed[i] < number_of_episodes_per_env:
                            next_obs, next_info = env.reset()
                            observations[i] = next_obs
                            rams[i] = np.array(next_info["milestones"], dtype=np.float32)
                            team_levels[i] = np.array(next_info["team_levels"], dtype=np.float32)
                            item_counts[i] = np.array(next_info["item_counts"], dtype=np.float32)
                            current_rewards[i] = 0.0
                            
                            recurrent_state[i] = torch.zeros(self.recurrent_dim, device=self.device)
                            latent_state[i] = torch.zeros(self.latent_dim, device=self.device)
                            action[i] = torch.zeros(self.action_dim, device=self.device)

                        for i in range(num_envs):
                            if local_buffers[i]:
                                for transition in local_buffers[i]:
                                    self.buffer.add(*transition)
                                local_buffers[i].clear()

        for i in range(num_envs):
            if local_buffers[i]:
                for transition in local_buffers[i]:
                    self.buffer.add(*transition)
                local_buffers[i].clear()

        return round(sum(scores) / len(scores), 2) if len(scores) > 0 else 0.0

    def saveCheckpoints(self, path):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        checkpoint = {
            # Models
            'recurrentModel': self.recurrentModel.state_dict(),
            'posteriorNet': self.posteriorNet.state_dict(),
            'priorNet': self.priorNet.state_dict(),
            'rewardPredictor': self.rewardPredictor.state_dict(),
            'curiosityPredictor': self.curiosityPredictor.state_dict(),
            'image_encoder': self.image_encoder.state_dict(),
            'goal_encoder': self.goal_encoder.state_dict(),
            'team_encoder': self.team_encoder.state_dict(),
            'item_encoder': self.item_encoder.state_dict(),
            'goalPredictor': self.goalPredictor.state_dict(),
            'teamPredictor': self.teamPredictor.state_dict(),
            'itemPredictor': self.itemPredictor.state_dict(),
            'projector': self.projector.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'ema_critic': self.ema_critic.state_dict(),
            'curiosity_critic': self.curiosity_critic.state_dict(),
            'ema_curiosity_critic': self.ema_curiosity_critic.state_dict(),
            'dynamicDataNormalizer': self.dynamicDataNormalizer.state_dict(),
            'curiosityDynamicDataNormalizer': self.curiosityDynamicDataNormalizer.state_dict(),
            # Independent Visualization Decoder
            'decoder': self.decoder.state_dict(),
            'decoderOptimizer': self.decoderOptimizer.state_dict(),
            
            # Optimizers
            'worldModelOptimizer': self.worldModelOptimizer.state_dict(),
            'actorOptimizer': self.actorOptimizer.state_dict(),
            'criticOptimizer': self.criticOptimizer.state_dict(),
            'curiosityCriticOptimizer': self.curiosityCriticOptimizer.state_dict(),
            
            # Progress
            'total_num_episodes': self.total_num_episodes,
            'total_num_steps': self.total_num_steps,
            'total_num_updates': self.total_num_updates,
        }
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

        # Handle Buffer
        buffer_path = os.path.join(directory, "replay_buffer.buffer") if directory else "replay_buffer.buffer"
        print(f"Saving replay buffer...")
        self.buffer.save(buffer_path)
        
        return 0

    def loadCheckpoints(self, path=None):
        if path is None:
            checkpoint_files = glob.glob("checkpoints/pokemon_model_R*_G*.pt") 
            if not checkpoint_files:
                path = 'model.pt'
            else:
                try:
                    # Load the one with the highest gradient steps
                    path = max(checkpoint_files, key=lambda x: int(x.split('_G')[-1].split('.pt')[0]))
                except:
                    path = max(checkpoint_files, key=os.path.getctime)

        if not os.path.exists(path):
            print(f"No checkpoint found at {path}, starting from scratch.")
            return 0

        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load Models
            self.recurrentModel.load_state_dict(checkpoint['recurrentModel'])
            self.posteriorNet.load_state_dict(checkpoint['posteriorNet'])
            self.priorNet.load_state_dict(checkpoint['priorNet'])
            self.rewardPredictor.load_state_dict(checkpoint['rewardPredictor'])
            self.image_encoder.load_state_dict(checkpoint['image_encoder'])
            self.goal_encoder.load_state_dict(checkpoint['goal_encoder'])
            self.team_encoder.load_state_dict(checkpoint['team_encoder'])
            self.item_encoder.load_state_dict(checkpoint['item_encoder'])
            self.projector.load_state_dict(checkpoint['projector'])
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.ema_critic.load_state_dict(checkpoint['ema_critic'])
            self.goalPredictor.load_state_dict(checkpoint['goalPredictor'])
            self.teamPredictor.load_state_dict(checkpoint['teamPredictor'])
            self.itemPredictor.load_state_dict(checkpoint['itemPredictor'])
            self.dynamicDataNormalizer.load_state_dict(checkpoint['dynamicDataNormalizer'])

            # Optional / Newly Added Curiosity Modules to keep older checkpoints loading smoothly
            self.curiosityPredictor.load_state_dict(checkpoint['curiosityPredictor'])
            self.curiosity_critic.load_state_dict(checkpoint['curiosity_critic'])
            self.ema_curiosity_critic.load_state_dict(checkpoint['ema_curiosity_critic'])
            self.curiosityDynamicDataNormalizer.load_state_dict(checkpoint['curiosityDynamicDataNormalizer'])

            if 'decoder' in checkpoint:
                self.decoder.load_state_dict(checkpoint['decoder'])
            if 'decoderOptimizer' in checkpoint:
                self.decoderOptimizer.load_state_dict(checkpoint['decoderOptimizer'])

            # Load Optimizers
            self.worldModelOptimizer.load_state_dict(checkpoint['worldModelOptimizer'])
            self.actorOptimizer.load_state_dict(checkpoint['actorOptimizer'])
            self.criticOptimizer.load_state_dict(checkpoint['criticOptimizer'])
            self.curiosityCriticOptimizer.load_state_dict(checkpoint['curiosityCriticOptimizer'])
            
            # Load Progress Counters
            self.total_num_episodes = checkpoint.get('total_num_episodes', 0)
            self.total_num_steps = checkpoint.get('total_num_steps', 0)
            self.total_num_updates = checkpoint.get('total_num_updates', 0)
            
            print(f"Loaded weights and progress ({self.total_num_updates} updates) from: {path}")

            # Load Buffer
            directory = os.path.dirname(path)
            buffer_path = os.path.join(directory, "replay_buffer.buffer") if directory else "replay_buffer.buffer"
            if os.path.exists(buffer_path):
                print(f"Loading replay buffer...")
                self.buffer.load(buffer_path)
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

        return 0
        
    @torch.no_grad()
    def visualize_single_dream(self, best_states, best_rewards, best_values, best_actions, best_advantages=None, title_prefix="Dream"):
        # Calculate predicted curiosities for each imagined state step
        curiosity_logits = self.curiosityPredictor(best_states.to(self.device))
        curiosities = self.two_hot.decode(curiosity_logits).squeeze(-1).cpu()

        # Reconstruct state frames using our detached decoder
        decoded_imgs = self.decoder(best_states.to(self.device)).clamp(0.0, 1.0).cpu() # [horizon, 3, 64, 64]

        horizon = best_states.shape[0]
        action_names = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]
        action_icons = {"UP": "▲ UP", "DOWN": "▼ DN", "LEFT": "◀ LT", "RIGHT": "▶ RT", "A": "A", "B": "B"}
        
        # Grid layout: Row 0 is the decoded visual frame, Row 1 is metrics
        fig, axes = plt.subplots(2, horizon, figsize=(horizon * 2.2, 5.5), facecolor='#0d1117', dpi=120,
                                 gridspec_kw={'height_ratios': [1.5, 1]})
        
        total_reward = best_rewards.sum().item() if best_rewards is not None else 0
        fig.suptitle(f'{title_prefix} (total reward = {total_reward:.2f})', 
                     color='#58a6ff', fontsize=14, fontweight='bold', y=0.98)
        
        if horizon == 1:
            axes = np.expand_dims(axes, axis=1)

        for i in range(horizon):
            ax_img = axes[0, i]
            ax_info = axes[1, i]
            
            # --- Row 0: Draw the Decoded Image ---
            img_np = decoded_imgs[i].permute(1, 2, 0).numpy()
            ax_img.imshow(img_np)
            ax_img.axis('off')
            
            for spine in ax_img.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#30363d')
                spine.set_linewidth(1.0)
            
            # --- Row 1: Draw the Information Text ---
            ax_info.set_xlim(0, 1)
            ax_info.set_ylim(0, 1)
            ax_info.axis('off')
            ax_info.set_facecolor('#161b22')
            
            for spine in ax_info.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#30363d')
                spine.set_linewidth(0.5)
            
            if i < horizon - 1:
                step_reward = best_rewards[i].item()
                step_value = best_values[i].item()
                action_idx = torch.argmax(best_actions[i]).item()
                action_str = action_names[action_idx] if action_idx < len(action_names) else str(action_idx)
                icon = action_icons.get(action_str, action_str)
                
                r_col = '#3fb950' if step_reward > 0 else ('#ff7b72' if step_reward < 0 else '#8b949e')
                
                y_pos = 0.88
                ax_info.text(0.5, y_pos, f'{icon}', ha='center', va='center',
                            fontsize=12, fontweight='bold', color='#58a6ff', transform=ax_info.transAxes)
                y_pos -= 0.20
                ax_info.text(0.5, y_pos, f'r:{step_reward:+.2f}', ha='center', va='center',
                            fontsize=11, color=r_col, fontfamily='monospace', transform=ax_info.transAxes)
                y_pos -= 0.20
                ax_info.text(0.5, y_pos, f'v:{step_value:.2f}', ha='center', va='center',
                            fontsize=11, color='#c9d1d9', fontfamily='monospace', transform=ax_info.transAxes)
                
                # Display predicted curiosity value
                step_curiosity = curiosities[i].item()
                y_pos -= 0.20
                ax_info.text(0.5, y_pos, f'c:{step_curiosity:.3f}', ha='center', va='center',
                            fontsize=11, color='#d1f1a5', fontfamily='monospace', transform=ax_info.transAxes)
                
                if best_advantages is not None:
                    adv = best_advantages[i].item()
                    adv_col = '#3fb950' if adv >= 0 else '#ff7b72'
                    y_pos -= 0.20
                    ax_info.text(0.5, y_pos, f'A:{adv:+.2f}', ha='center', va='center',
                                fontsize=11, fontweight='bold', color=adv_col, fontfamily='monospace',
                                transform=ax_info.transAxes)
            else:
                if i < len(best_values):
                    step_value = best_values[i].item()
                    step_value_str = f"v:{step_value:.2f}"
                else:
                    step_value_str = "v: N/A"

                ax_info.text(0.5, 0.78, 'END', ha='center', va='center',
                            fontsize=12, fontweight='bold', color='#8b949e', transform=ax_info.transAxes)
                ax_info.text(0.5, 0.53, step_value_str, ha='center', va='center',
                            fontsize=11, color='#c9d1d9', fontfamily='monospace', transform=ax_info.transAxes)
                
                # Display final step curiosity value
                step_curiosity = curiosities[i].item()
                ax_info.text(0.5, 0.28, f'c:{step_curiosity:.3f}', ha='center', va='center',
                            fontsize=11, color='#d1f1a5', fontfamily='monospace', transform=ax_info.transAxes)
            
        plt.subplots_adjust(top=0.85, bottom=0.05, hspace=0.15)
        plt.show()