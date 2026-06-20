import os
import glob
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, kl_divergence, OneHotCategoricalStraightThrough, OneHotCategorical
from torch.distributions.utils import probs_to_logits

from PokemonRedEnv import LTM_REWARD_DIM, LTM_MAP_DIM

IMAGE_SIZE = 64

torch.set_float32_matmul_precision('high')


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def _mlp2(in_dim, out_dim, hidden=1024, act=nn.ELU, norm_out=True):
    """Two-layer MLP (one hidden layer). `hidden` controls the MLP width."""
    layers = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), act(), nn.Linear(hidden, out_dim)]
    if norm_out:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


def _mlp3(in_dim, out_dim, hidden=1024, act=nn.SiLU):
    """Three-hidden-layer MLP (four Linear layers). `hidden` controls the MLP width."""
    d = hidden
    return nn.Sequential(
        nn.Linear(in_dim, d), nn.LayerNorm(d), act(),
        nn.Linear(d, d), nn.LayerNorm(d), act(),
        nn.Linear(d, d), nn.LayerNorm(d), act(),
        nn.Linear(d, out_dim),
    )


# ==========================================================
# SCALAR ENCODINGS
# ==========================================================
class TwoHotEncoding:
    """Symlog two-hot encoding (DreamerV3) for value/reward estimation."""
    def __init__(self, min=-10, max=10, num_bins=255, device="cuda"):
        self.device = device
        self.num_bins = num_bins
        self.bins = torch.linspace(min, max, num_bins, device=device)
        self.bin_values = symexp(self.bins)

    def encode(self, x):
        x = symlog(x)
        x = torch.clamp(x, self.bins[0], self.bins[-1])
        pos = (x - self.bins[0]) / (self.bins[1] - self.bins[0])
        low = torch.clamp(torch.floor(pos).long(), 0, self.num_bins - 2)
        high = low + 1
        weight_high = pos - low
        weight_low = 1.0 - weight_high
        two_hot = torch.zeros(*x.shape, self.num_bins, device=self.device)
        two_hot.scatter_(-1, low.unsqueeze(-1), weight_low.unsqueeze(-1))
        two_hot.scatter_(-1, high.unsqueeze(-1), weight_high.unsqueeze(-1))
        return two_hot

    def decode(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return torch.sum(probs * self.bin_values, dim=-1, keepdim=True)


# ==========================================================
# IMAGE ENCODER / DECODER
# ==========================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),  # LayerNorm equivalent for images
            nn.ELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderImage(nn.Module):
    def __init__(self, output_size=1024, depth=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, depth, kernel_size=4, stride=2, padding=1), nn.ELU(), ResBlock(depth),                 # 64 -> 32
            nn.Conv2d(depth, depth * 2, kernel_size=4, stride=2, padding=1), nn.ELU(), ResBlock(depth * 2),     # 32 -> 16
            nn.Conv2d(depth * 2, depth * 4, kernel_size=4, stride=2, padding=1), nn.ELU(), ResBlock(depth * 4), # 16 -> 8
            nn.Conv2d(depth * 4, depth * 8, kernel_size=4, stride=2, padding=1), nn.ELU(),                      # 8 -> 4
            nn.Flatten(),
            nn.Linear(depth * 8 * 4 * 4, output_size),
            nn.LayerNorm(output_size),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    """Visualization decoder (trained separately on detached states)."""
    def __init__(self, input_size, depth=32):
        super().__init__()
        self.depth = depth
        self.linear = nn.Linear(input_size, depth * 8 * 4 * 4)
        self.net = nn.Sequential(
            ResBlock(depth * 8),
            nn.ConvTranspose2d(depth * 8, depth * 4, 4, stride=2, padding=1), nn.ELU(),  # 4 -> 8
            ResBlock(depth * 4),
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, stride=2, padding=1), nn.ELU(),  # 8 -> 16
            ResBlock(depth * 2),
            nn.ConvTranspose2d(depth * 2, depth, 4, stride=2, padding=1), nn.ELU(),      # 16 -> 32
            nn.ConvTranspose2d(depth, 3, 4, stride=2, padding=1),                        # 32 -> 64
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.depth * 8, 4, 4)
        return self.net(x)


# ==========================================================
# AUXILIARY ENCODERS (state -> feature) AND PREDICTORS (latent -> state)
# ==========================================================
class TeamItemEncoder(nn.Module):
    """Combined party-levels + item-counts encoder (single encoder, feature #3)."""
    def __init__(self, in_dim=8, out_dim=256, hidden=1024):
        super().__init__()
        self.net = _mlp2(in_dim, out_dim, hidden=hidden)

    def forward(self, x):
        return self.net(x)


class TeamItemPredictor(nn.Module):
    """Combined party-levels + item-counts predictor (single decoder, feature #3)."""
    def __init__(self, input_size, out_dim=8, hidden=1024):
        super().__init__()
        self.net = _mlp2(input_size, out_dim, hidden=hidden, act=nn.SiLU, norm_out=False)

    def forward(self, x):
        return self.net(x)


class LongTermMemoryEncoder(nn.Module):
    """Whole-game long-term reward-memory (event flags) encoder (feature #2)."""
    def __init__(self, in_dim=LTM_REWARD_DIM, out_dim=512, hidden=1024):
        super().__init__()
        self.net = _mlp2(in_dim, out_dim, hidden=hidden)

    def forward(self, x):
        return self.net(x)


class LongTermMemoryPredictor(nn.Module):
    """Whole-game long-term reward-memory predictor (multi-label logits)."""
    def __init__(self, input_size, out_dim=LTM_REWARD_DIM, hidden=1024):
        super().__init__()
        self.net = _mlp2(input_size, out_dim, hidden=hidden, act=nn.SiLU, norm_out=False)
        # Init flags off so nothing is hallucinated initially.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -5.0)

    def forward(self, x):
        return self.net(x)


class LongTermMapEncoder(nn.Module):
    """Whole-game long-term map-memory encoder (feature #1: map vector for all the game)."""
    def __init__(self, in_dim=LTM_MAP_DIM, out_dim=256, hidden=1024):
        super().__init__()
        self.net = _mlp2(in_dim, out_dim, hidden=hidden)

    def forward(self, x):
        return self.net(x)


class LongTermMapPredictor(nn.Module):
    """Whole-game long-term map-memory predictor (multi-label logits)."""
    def __init__(self, input_size, out_dim=LTM_MAP_DIM, hidden=1024):
        super().__init__()
        self.net = _mlp2(input_size, out_dim, hidden=hidden, act=nn.SiLU, norm_out=False)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -5.0)

    def forward(self, x):
        return self.net(x)


# ==========================================================
# REWARD / CURIOSITY HEADS
# ==========================================================
class RewardPredictor(nn.Module):
    def __init__(self, inputsize, num_bins=255, mlp_dim=1024):
        super().__init__()
        self.transform = _mlp3(inputsize, num_bins, hidden=mlp_dim, act=nn.SiLU)
        nn.init.zeros_(self.transform[-1].weight)
        nn.init.zeros_(self.transform[-1].bias)

    def forward(self, x):
        return self.transform(x)


class EnsembleRewardPredictor(nn.Module):
    """Disagreement-friendly ensemble of reward heads (kept from the working project)."""
    def __init__(self, inputsize, num_bins=255, num_heads=3, mlp_dim=1024):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            RewardPredictor(inputsize, num_bins, mlp_dim=mlp_dim) for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.stack([head(x) for head in self.heads], dim=0)


class CuriosityPredictor(nn.Module):
    """Learned curiosity head (two hidden layers); biased toward 0 at init."""
    def __init__(self, input_size, num_bins=255, mlp_dim=1024):
        super().__init__()
        d = mlp_dim
        self.net = nn.Sequential(
            nn.Linear(input_size, d), nn.LayerNorm(d), nn.SiLU(),
            nn.Linear(d, d), nn.LayerNorm(d), nn.SiLU(),
            nn.Linear(d, num_bins),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            # Symlog two-hot: value 0 sits at the center bin, so bias toward it.
            self.net[-1].bias[num_bins // 2] = 10.0

    def forward(self, x):
        return self.net(x)


# ==========================================================
# RSSM CORE
# ==========================================================
class RecurrentModel(nn.Module):
    def __init__(self, recurrentSize=4096, latentSize=64 * 64, actionSize=6):
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
    def __init__(self, recurrentSize=512, rows=16, cols=16, mlp_dim=1024):
        super().__init__()
        self.recurrentSize = recurrentSize
        self.latentSize = rows * cols
        self.rows = rows
        self.cols = cols
        self.trasform = _mlp3(recurrentSize, rows * cols, hidden=mlp_dim, act=nn.SiLU)

    def forward(self, RecurrentState):
        rawLogits = self.trasform(RecurrentState)
        rawProbabilities = rawLogits.view(-1, self.rows, self.cols).softmax(-1)
        confusion = torch.ones_like(rawProbabilities) / self.cols
        probabilities = 0.99 * rawProbabilities + 0.01 * confusion
        logits = probs_to_logits(probabilities)
        distribution = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        sample = distribution.rsample().view(-1, self.latentSize)
        return sample, logits


class PosteriorNet(nn.Module):
    def __init__(self, inputSize=1536, rows=16, cols=16, mlp_dim=1024):
        super().__init__()
        self.inputSize = inputSize
        self.rows = rows
        self.cols = cols
        self.latentSize = rows * cols
        self.trasform = _mlp3(inputSize, self.latentSize, hidden=mlp_dim, act=nn.SiLU)

    def forward(self, InputState):
        rawLogits = self.trasform(InputState)
        rawProbabilities = rawLogits.view(-1, self.rows, self.cols).softmax(-1)
        confusion = torch.ones_like(rawProbabilities) / self.cols
        probabilities = 0.99 * rawProbabilities + 0.01 * confusion
        logits = probs_to_logits(probabilities)
        distribution = Independent(OneHotCategoricalStraightThrough(logits=logits, validate_args=False), 1)
        sample = distribution.rsample().view(-1, self.rows * self.cols)
        return sample, logits


# ==========================================================
# ACTOR / CRITIC
# ==========================================================
class Actor(nn.Module):
    def __init__(self, action_dim, device, concatenated_dim=768, mlp_dim=1024):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.concatenated_dim = concatenated_dim
        self.net = _mlp3(concatenated_dim, action_dim, hidden=mlp_dim, act=nn.SiLU)
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        raw_logits = self.net(x)
        probs = torch.softmax(raw_logits, dim=-1)
        uniform = torch.ones_like(probs) / self.action_dim
        mixed_probs = 0.99 * probs + 0.01 * uniform
        dist = OneHotCategorical(probs=mixed_probs)
        action_onehot = dist.sample()
        return action_onehot, dist.log_prob(action_onehot), dist.entropy()


class Critic(nn.Module):
    def __init__(self, inputSize, bins=255, mlp_dim=1024):
        super().__init__()
        self.transform = _mlp3(inputSize, bins, hidden=mlp_dim, act=nn.SiLU)
        nn.init.zeros_(self.transform[-1].weight)
        nn.init.zeros_(self.transform[-1].bias)

    def forward(self, x):
        return self.transform(x)


# ==========================================================
# UTILITIES
# ==========================================================
class DynamicDataNormalizer(nn.Module):
    def __init__(self, device, decay=0.99, min_=1.0, percentileLow=0.05, percentileHigh=0.95):
        super().__init__()
        self._decay = decay
        self._min = torch.tensor(min_, device=device)
        self._percentileLow = percentileLow
        self._percentileHigh = percentileHigh
        self.register_buffer("S", torch.zeros((), dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        low = torch.quantile(x, self._percentileLow)
        high = torch.quantile(x, self._percentileHigh)
        self.S.copy_(self._decay * self.S + (1.0 - self._decay) * (high - low))
        return torch.max(self._min, self.S).detach()


# ==========================================================
# REPLAY BUFFER
# ==========================================================
class Buffer(object):
    def __init__(self, device, capacity=800000, actionSize=6,
                 ltm_reward_dim=LTM_REWARD_DIM, ltm_map_dim=LTM_MAP_DIM,
                 item_dim=2, team_level_dim=6, num_envs=4):
        self.device = device
        self.capacity = capacity
        self.num_envs = num_envs
        self.observations = torch.empty((capacity, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8, device='cpu')
        self.ltm_rewards = torch.empty((capacity, ltm_reward_dim), dtype=torch.uint8, device='cpu')
        self.ltm_maps = torch.empty((capacity, ltm_map_dim), dtype=torch.uint8, device='cpu')
        self.item_counts = torch.empty((capacity, item_dim), dtype=torch.float32, device='cpu')
        self.team_levels = torch.empty((capacity, team_level_dim), dtype=torch.float32, device='cpu')
        self.actions = torch.empty((capacity, actionSize), dtype=torch.float32, device='cpu')
        # Reward is stored as two streams (summed only when an env total is needed):
        #   sparse_rewards   -> memory-tied events (LTM-gated in the dream)
        #   standard_rewards -> non-memory events (healing, level-ups)
        self.sparse_rewards = torch.empty((capacity, 1), dtype=torch.float32, device='cpu')
        self.standard_rewards = torch.empty((capacity, 1), dtype=torch.float32, device='cpu')
        # Curiosity is stored as two streams (summed only when a total is needed):
        #   sparse_curiosities -> map-transition bonus (map-gated in the dream)
        #   tile_curiosities   -> per-tile discovery bonus (+0.01), added ungated
        self.sparse_curiosities = torch.empty((capacity, 1), dtype=torch.float32, device='cpu')
        self.tile_curiosities = torch.empty((capacity, 1), dtype=torch.float32, device='cpu')

        self.index = 0
        self.full = False

    def add(self, observation, ltm_reward, ltm_map, item_count, team_level, action,
            sparse_reward, standard_reward, sparse_curiosity, tile_curiosity):
        self.observations[self.index] = torch.as_tensor(observation, dtype=torch.uint8)
        self.ltm_rewards[self.index] = torch.as_tensor(ltm_reward, dtype=torch.uint8)
        self.ltm_maps[self.index] = torch.as_tensor(ltm_map, dtype=torch.uint8)
        self.item_counts[self.index] = torch.as_tensor(item_count, dtype=torch.float32)
        self.team_levels[self.index] = torch.as_tensor(team_level, dtype=torch.float32)
        self.actions[self.index] = torch.as_tensor(action, dtype=torch.float32)
        self.sparse_rewards[self.index] = torch.as_tensor(sparse_reward, dtype=torch.float32)
        self.standard_rewards[self.index] = torch.as_tensor(standard_reward, dtype=torch.float32)
        self.sparse_curiosities[self.index] = torch.as_tensor(sparse_curiosity, dtype=torch.float32)
        self.tile_curiosities[self.index] = torch.as_tensor(tile_curiosity, dtype=torch.float32)

        self.index = (self.index + 1) % self.capacity
        self.full = self.full or (self.index == 0)

    def sample(self, batchSize, sequenceSize):
        N = self.capacity if self.full else self.index
        if N < sequenceSize:
            return None

        num_recent = int(round(batchSize * 0.25))
        num_all = batchSize - num_recent
        sample_indices = []

        # 25% from a recent window, 75% uniformly across the whole buffer.
        if num_recent > 0:
            effective_window = min(20000 * self.num_envs, N)
            max_offset = effective_window - sequenceSize
            if max_offset >= 0:
                offsets = torch.randint(0, max_offset + 1, (num_recent, 1))
                recent_starts = (self.index - sequenceSize - offsets) % self.capacity
                seq_offsets = torch.arange(sequenceSize).reshape(1, -1)
                sample_indices.append((recent_starts + seq_offsets) % self.capacity)
            else:
                num_all += num_recent

        if num_all > 0:
            limit = self.capacity if self.full else (self.index - sequenceSize + 1)
            if limit <= 0:
                return None
            all_starts = torch.randint(0, limit, (num_all, 1))
            seq_offsets = torch.arange(sequenceSize).reshape(1, -1)
            sample_indices.append((all_starts + seq_offsets) % self.capacity)

        sampleIndex = torch.cat(sample_indices, dim=0).long()
        obs_float = self.observations[sampleIndex].to(self.device, non_blocking=True).float() / 255.0
        return {
            "observations": obs_float,
            "ltm_rewards":  self.ltm_rewards[sampleIndex].to(self.device, non_blocking=True).float(),
            "ltm_maps":     self.ltm_maps[sampleIndex].to(self.device, non_blocking=True).float(),
            "item_counts":  self.item_counts[sampleIndex].to(self.device, non_blocking=True),
            "team_levels":  self.team_levels[sampleIndex].to(self.device, non_blocking=True),
            "actions":          self.actions[sampleIndex].to(self.device, non_blocking=True),
            "sparse_rewards":     self.sparse_rewards[sampleIndex].to(self.device, non_blocking=True),
            "standard_rewards":   self.standard_rewards[sampleIndex].to(self.device, non_blocking=True),
            "sparse_curiosities": self.sparse_curiosities[sampleIndex].to(self.device, non_blocking=True),
            "tile_curiosities":   self.tile_curiosities[sampleIndex].to(self.device, non_blocking=True),
            "index":        sampleIndex.to(self.device, non_blocking=True),
        }

    def save(self, path):
        limit = self.capacity if self.full else self.index
        torch.save({
            'observations': self.observations[:limit],
            'ltm_rewards': self.ltm_rewards[:limit],
            'ltm_maps': self.ltm_maps[:limit],
            'item_counts': self.item_counts[:limit],
            'team_levels': self.team_levels[:limit],
            'actions': self.actions[:limit],
            'sparse_rewards': self.sparse_rewards[:limit],
            'standard_rewards': self.standard_rewards[:limit],
            'sparse_curiosities': self.sparse_curiosities[:limit],
            'tile_curiosities': self.tile_curiosities[:limit],
            'index': self.index,
            'full': self.full,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        n = min(ckpt['observations'].shape[0], self.capacity)
        for name in ('observations', 'ltm_rewards', 'ltm_maps', 'item_counts',
                     'team_levels', 'actions', 'sparse_rewards', 'standard_rewards',
                     'sparse_curiosities', 'tile_curiosities'):
            if name in ckpt:
                getattr(self, name)[:n] = ckpt[name][:n]
            elif name == 'sparse_curiosities' and 'curiosities' in ckpt:
                # Legacy migration: pre-split buffers stored a single 'curiosities'
                # field that WAS exactly the sparse map-transition bonus (tile curiosity
                # did not exist), so map it onto sparse_curiosities. tile_curiosities
                # stays zero for these old transitions and fills in as the buffer cycles.
                self.sparse_curiosities[:n] = ckpt['curiosities'][:n]
                print("[*] Buffer field 'sparse_curiosities' migrated from legacy 'curiosities'.")
            else:
                print(f"[*] Buffer field '{name}' absent in checkpoint; leaving zeros.")
        self.index = n % self.capacity
        self.full = (n == self.capacity)
        print(f"[*] Loaded buffer with {n} transitions (full={self.full}, write_head={self.index})")

    def print_diagnostics(self):
        valid = self.capacity if self.full else self.index
        print("\n" + "=" * 50)
        print("          REPLAY BUFFER DIAGNOSTICS")
        print("=" * 50)
        print(f"  Buffer Fill : {valid:,} / {self.capacity:,} ({100.0 * valid / self.capacity:.2f}%)")
        print(f"  Active Environments : {self.num_envs}")
        print(f"  Recent Sampling Window Size : {20000 * self.num_envs:,} steps")
        print("=" * 50 + "\n")


# ==========================================================
# DREAMER
# ==========================================================
class Dreamer:
    def __init__(self, device, obs_dim, envs,
                 action_dim=6, recurrent_dim=512, rows=64, cols=64, latent_dim=256,
                 number_of_sequences=32, steps_per_sequence=64, seed=42, buffer_size=10000,
                 team_dim=6, item_dim=2, curiosity_scale=0.05, mlp_dim=1024):

        # --- Dimensions / config ---
        self.device = device
        self.action_dim = action_dim
        self.recurrent_dim = recurrent_dim
        self.mlp_dim = mlp_dim
        self.rows = rows
        self.cols = cols
        self.latent_dim = rows * cols
        self.concatenated_dim = recurrent_dim + self.latent_dim
        self.total_num_episodes = 0
        self.total_num_steps = 0
        self.total_num_updates = 0

        # Auxiliary feature sizes feeding the posterior.
        self.image_out = 1024
        self.teamitem_out = 256
        self.ltm_reward_out = 512
        self.ltm_map_out = 512
        self.team_dim = team_dim
        self.item_dim = item_dim
        self.enconder_output_size = (self.image_out + self.teamitem_out
                                     + self.ltm_reward_out + self.ltm_map_out)

        self.entropy_scale = 0.001
        self.number_of_sequences = number_of_sequences
        self.steps_per_sequence = steps_per_sequence
        self.buffer_capacity = buffer_size
        self.curiosity_scale = curiosity_scale
        self.envs = envs

        self.dynamicDataNormalizer = DynamicDataNormalizer(self.device)
        self.curiosityDynamicDataNormalizer = DynamicDataNormalizer(self.device)

        # Whole-game LTM loss shaping (sparse multi-label targets).
        self.ltm_reward_pos_weight = torch.tensor(200.0, device=self.device)
        self.ltm_map_pos_weight = torch.tensor(5.0, device=self.device)
        self.ltm_sparsity_weight = 0.5

        # Exploration: fraction of dream starts resampled from high-novelty states.
        self.dream_priority_fraction = 0.10

        # --- Encodings ---
        self.two_hot = TwoHotEncoding(device=self.device)

        # --- RSSM ---
        self.recurrentModel = RecurrentModel(self.recurrent_dim, self.latent_dim, self.action_dim).to(self.device)
        self.posteriorNet = PosteriorNet(self.enconder_output_size + self.recurrent_dim, self.rows, self.cols, mlp_dim=self.mlp_dim).to(self.device)
        self.priorNet = PriorNet(self.recurrent_dim, self.rows, self.cols, mlp_dim=self.mlp_dim).to(self.device)

        # --- Encoders ---
        self.image_encoder = EncoderImage(output_size=self.image_out).to(self.device)
        self.teamitem_encoder = TeamItemEncoder(self.team_dim + self.item_dim, self.teamitem_out, hidden=self.mlp_dim).to(self.device)
        self.ltm_reward_encoder = LongTermMemoryEncoder(LTM_REWARD_DIM, self.ltm_reward_out, hidden=self.mlp_dim).to(self.device)
        self.ltm_map_encoder = LongTermMapEncoder(LTM_MAP_DIM, self.ltm_map_out, hidden=self.mlp_dim).to(self.device)

        # --- Predictors (latent -> observation components) ---
        # Reward prediction is split into two predictors for one conceptual reward:
        #   sparseRewardPredictor   -> memory-tied events (LTM-gated in the dream)
        #   standardRewardPredictor -> non-memory events (healing, level-ups)
        # Their decoded outputs are summed; the sum feeds the single (shared) reward
        # critic and dynamic normalizer exactly as the original single predictor did.
        self.sparseRewardPredictor = EnsembleRewardPredictor(self.concatenated_dim, num_heads=3, mlp_dim=self.mlp_dim).to(self.device)
        self.standardRewardPredictor = EnsembleRewardPredictor(self.concatenated_dim, num_heads=3, mlp_dim=self.mlp_dim).to(self.device)
        # Curiosity is one conceptual signal predicted as two streams (summed in the dream):
        #   curiosityPredictor     -> sparse map-transition curiosity (existing, map-gated)
        #   tileCuriosityPredictor -> per-tile discovery curiosity (+0.01), added ungated
        self.curiosityPredictor = CuriosityPredictor(self.concatenated_dim, mlp_dim=self.mlp_dim).to(self.device)
        self.tileCuriosityPredictor = CuriosityPredictor(self.concatenated_dim, mlp_dim=self.mlp_dim).to(self.device)
        self.teamitemPredictor = TeamItemPredictor(self.concatenated_dim, self.team_dim + self.item_dim, hidden=self.mlp_dim).to(self.device)
        self.ltm_reward_predictor = LongTermMemoryPredictor(self.concatenated_dim, LTM_REWARD_DIM, hidden=self.mlp_dim).to(self.device)
        self.ltm_map_predictor = LongTermMapPredictor(self.concatenated_dim, LTM_MAP_DIM, hidden=self.mlp_dim).to(self.device)

        # --- Representation alignment (Barlow Twins) ---
        self.projector = nn.Linear(self.concatenated_dim, self.image_out).to(self.device)

        # --- Visualization decoder (trained on detached states) ---
        self.decoder = Decoder(input_size=self.concatenated_dim).to(self.device)
        self.decoderOptimizer = torch.optim.Adam(self.decoder.parameters(), lr=2e-4)

        # --- Actor / Critics ---
        self.actor = Actor(self.action_dim, self.device, self.concatenated_dim, mlp_dim=self.mlp_dim).to(self.device)

        self.critic = Critic(self.concatenated_dim, mlp_dim=self.mlp_dim).to(self.device)
        self.ema_critic = copy.deepcopy(self.critic)
        for p in self.ema_critic.parameters():
            p.requires_grad = False

        self.curiosity_critic = Critic(self.concatenated_dim, mlp_dim=self.mlp_dim).to(self.device)
        self.ema_curiosity_critic = copy.deepcopy(self.curiosity_critic)
        for p in self.ema_curiosity_critic.parameters():
            p.requires_grad = False

        self.critic_ema_decay = 0.98
        self.curiosity_critic_ema_decay = 0.90

        # --- Buffer ---
        self.buffer = Buffer(
            device=self.device, capacity=self.buffer_capacity, actionSize=self.action_dim,
            ltm_reward_dim=LTM_REWARD_DIM, ltm_map_dim=LTM_MAP_DIM,
            item_dim=self.item_dim, team_level_dim=self.team_dim,
            num_envs=len(envs),
        )

        # --- World-model parameter group ---
        self.worldModelParameters = (
            list(self.recurrentModel.parameters())
            + list(self.posteriorNet.parameters())
            + list(self.priorNet.parameters())
            + list(self.sparseRewardPredictor.parameters())
            + list(self.standardRewardPredictor.parameters())
            + list(self.image_encoder.parameters())
            + list(self.teamitem_encoder.parameters())
            + list(self.ltm_reward_encoder.parameters())
            + list(self.ltm_map_encoder.parameters())
            + list(self.projector.parameters())
            + list(self.teamitemPredictor.parameters())
            + list(self.ltm_reward_predictor.parameters())
            + list(self.ltm_map_predictor.parameters())
        )

        # --- Optimizers ---
        self.worldModelOptimizer = torch.optim.Adam(self.worldModelParameters, lr=2e-4)
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=4e-5)
        self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.curiosityCriticOptimizer = torch.optim.Adam(self.curiosity_critic.parameters(), lr=1e-4)
        self.curiosityHeadOptimizer = torch.optim.Adam(
            list(self.curiosityPredictor.parameters())
            + list(self.tileCuriosityPredictor.parameters()), lr=1e-4)

        # --- Statistics ---
        self.num_episodes = 0
        self.num_steps = 0
        self.num_updates = 0

    # ------------------------------------------------------
    def sample_batch(self, batchSize, sequenceSize):
        return self.buffer.sample(batchSize, sequenceSize)

    def computeLambdaValues(self, rewards, values, continues, lambda_=0.95):
        returns = torch.zeros_like(rewards)
        bootstrap = values[:, -1]
        for i in reversed(range(rewards.shape[-1])):
            returns[:, i] = rewards[:, i] + continues[:, i] * ((1 - lambda_) * values[:, i + 1] + lambda_ * bootstrap)
            bootstrap = returns[:, i]
        return returns

    def _ltm_loss(self, pred_logits, target, pos_weight):
        """Weighted BCE + sparsity penalty for sparse whole-game multi-label targets."""
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pos_weight)
        sparsity = (torch.sigmoid(pred_logits) * (1.0 - target)).mean()
        return bce + self.ltm_sparsity_weight * sparsity

    # ------------------------------------------------------
    def _encode_components(self, obs, ltm_reward, ltm_map, team, item):
        """Encode all observation components (each input already on device)."""
        teamitem = torch.cat((team / 100.0, item / 10.0), dim=-1)
        return (
            self.image_encoder(obs),
            self.teamitem_encoder(teamitem),
            self.ltm_reward_encoder(ltm_reward),
            self.ltm_map_encoder(ltm_map),
        )

    # ------------------------------------------------------
    def TrainWorldModel(self, batch_data):
        self.worldModelOptimizer.zero_grad(set_to_none=True)

        B, T = self.number_of_sequences, self.steps_per_sequence
        obs_flat = batch_data["observations"].flatten(0, 1)
        ltm_reward_flat = batch_data["ltm_rewards"].flatten(0, 1)
        ltm_map_flat = batch_data["ltm_maps"].flatten(0, 1)
        team_flat = batch_data["team_levels"].flatten(0, 1)
        item_flat = batch_data["item_counts"].flatten(0, 1)

        enc_img, enc_teamitem, enc_ltm_reward, enc_ltm_map = self._encode_components(
            obs_flat, ltm_reward_flat, ltm_map_flat, team_flat, item_flat)
        encoded_images = enc_img.view(B, T, -1)
        encoded_teamitem = enc_teamitem.view(B, T, -1)
        encoded_ltm_reward = enc_ltm_reward.view(B, T, -1)
        encoded_ltm_map = enc_ltm_map.view(B, T, -1)

        previous_recurrent_state = torch.zeros(B, self.recurrent_dim, device=self.device)
        previous_latent_state = torch.zeros(B, self.latent_dim, device=self.device)

        recurrent_states, priors, priors_logits, posteriors_logits, posteriors = [], [], [], [], []
        for t in range(1, T):
            recurrent_state = self.recurrentModel(previous_recurrent_state, previous_latent_state, batch_data["actions"][:, t - 1])
            prior_sample, prior_logits = self.priorNet(recurrent_state)
            posterior_input = torch.cat(
                (recurrent_state, encoded_images[:, t], encoded_teamitem[:, t],
                 encoded_ltm_reward[:, t], encoded_ltm_map[:, t]), dim=-1)
            posterior, posterior_logits = self.posteriorNet(posterior_input)

            recurrent_states.append(recurrent_state)
            priors.append(prior_sample)
            priors_logits.append(prior_logits)
            posteriors_logits.append(posterior_logits)
            posteriors.append(posterior)
            previous_recurrent_state = recurrent_state
            previous_latent_state = posterior

        recurrent_states = torch.stack(recurrent_states, dim=1)
        priors = torch.stack(priors, dim=1)
        priors_logits = torch.stack(priors_logits, dim=1)
        posteriors_logits = torch.stack(posteriors_logits, dim=1)
        posteriors = torch.stack(posteriors, dim=1)
        full_states = torch.cat((recurrent_states, posteriors), -1)
        full_states_prior = torch.cat((recurrent_states, priors), -1)

        # ----------------------------------------------------
        # BARLOW TWINS REPRESENTATION ALIGNMENT (latent <-> image encoder)
        # ----------------------------------------------------
        flat_states = full_states.view(-1, self.concatenated_dim)
        k = self.projector(flat_states)
        e = encoded_images[:, 1:].reshape(-1, self.image_out).detach()

        k_norm = (k - k.mean(dim=0)) / (k.std(dim=0) + 1e-5)
        e_norm = (e - e.mean(dim=0)) / (e.std(dim=0) + 1e-5)
        C = (k_norm.T @ e_norm) / k.size(0)
        invariance_loss = ((torch.diagonal(C) - 1) ** 2).sum()
        off_diag = C.clone()
        off_diag.fill_diagonal_(0)
        redundancy_loss = (off_diag ** 2).sum()
        alpha = 5e-4
        beta_BT = 0.05
        scaled_bt_loss = beta_BT * (invariance_loss + alpha * redundancy_loss)

        # ----------------------------------------------------
        # REWARD (two ensembles) + AUXILIARY PREDICTIONS
        # One conceptual reward, predicted as sparse + standard streams. Each ensemble
        # is trained on its own target; the losses are summed and added to the world
        # model loss just like the original single reward loss.
        # ----------------------------------------------------
        def _ensemble_reward_loss(predictor, logits_ensemble, target_scalar):
            with torch.no_grad():
                target_two_hot = self.two_hot.encode(target_scalar.squeeze(-1))
            loss = 0.0
            for h in range(predictor.num_heads):
                loss += -torch.mean(torch.sum(
                    target_two_hot * torch.log_softmax(logits_ensemble[h], dim=-1), dim=-1)) * 100
            return loss / predictor.num_heads

        sparse_logits_ensemble = self.sparseRewardPredictor(full_states)      # [num_heads, B, T-1, num_bins]
        standard_logits_ensemble = self.standardRewardPredictor(full_states)
        sparse_reward_loss = _ensemble_reward_loss(
            self.sparseRewardPredictor, sparse_logits_ensemble, batch_data["sparse_rewards"][:, :-1])
        standard_reward_loss = _ensemble_reward_loss(
            self.standardRewardPredictor, standard_logits_ensemble, batch_data["standard_rewards"][:, :-1])
        reward_loss = sparse_reward_loss + standard_reward_loss

        # Combined team+item prediction (single decoder).
        pred_teamitem = self.teamitemPredictor(full_states)
        target_teamitem = torch.cat((batch_data["team_levels"][:, 1:] / 100.0,
                                     batch_data["item_counts"][:, 1:] / 10.0), dim=-1)
        teamitem_loss = F.mse_loss(pred_teamitem, target_teamitem) * 100.0

        # Whole-game long-term reward memory (sparse multi-label).
        pred_ltm_reward = self.ltm_reward_predictor(full_states)
        ltm_reward_loss = self._ltm_loss(pred_ltm_reward, batch_data["ltm_rewards"][:, 1:], self.ltm_reward_pos_weight) * 50.0

        # Whole-game long-term map memory (sparse multi-label).
        pred_ltm_map = self.ltm_map_predictor(full_states)
        ltm_map_loss = self._ltm_loss(pred_ltm_map, batch_data["ltm_maps"][:, 1:], self.ltm_map_pos_weight) * 50.0

        # KL loss.
        prior_distribution = Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1)
        prior_distribution_SG = Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1)
        posterior_distribution = Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1)
        posterior_distribution_SG = Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1)
        prior_loss = kl_divergence(posterior_distribution_SG, prior_distribution)
        posterior_loss = kl_divergence(posterior_distribution, prior_distribution_SG)
        freeNats = torch.full_like(prior_loss, 1)
        kl_loss = (1 * torch.maximum(prior_loss, freeNats) + 0.1 * torch.maximum(posterior_loss, freeNats)).mean()

        world_model_loss = (scaled_bt_loss + kl_loss + reward_loss + teamitem_loss
                            + ltm_reward_loss + ltm_map_loss)
        world_model_loss.backward()
        nn.utils.clip_grad_norm_(self.worldModelParameters, 10, norm_type=2)
        self.worldModelOptimizer.step()

        # --- Separate visualization decoder training (detached) ---
        self.decoderOptimizer.zero_grad(set_to_none=True)
        detached_states = full_states.detach().view(-1, self.concatenated_dim)
        recon_imgs = self.decoder(detached_states)
        target_imgs = batch_data["observations"][:, 1:].flatten(0, 1)
        decoder_loss = F.mse_loss(recon_imgs, target_imgs)
        decoder_loss.backward()
        self.decoderOptimizer.step()

        # --- Curiosity head training on real (prior) states ---
        # Two heads, one conceptual curiosity: sparse (map-transition) + tile (per-tile).
        prior_states_flat = full_states_prior.detach().view(-1, self.concatenated_dim)
        with torch.no_grad():
            target_sparse_curiosity = self.two_hot.encode(batch_data["sparse_curiosities"][:, :-1].squeeze(-1).reshape(-1))
            target_tile_curiosity = self.two_hot.encode(batch_data["tile_curiosities"][:, :-1].squeeze(-1).reshape(-1))
        self.curiosityHeadOptimizer.zero_grad(set_to_none=True)
        pred_sparse_curiosity_logits = self.curiosityPredictor(prior_states_flat)
        pred_tile_curiosity_logits = self.tileCuriosityPredictor(prior_states_flat)
        sparse_curiosity_loss = -torch.mean(torch.sum(
            target_sparse_curiosity * torch.log_softmax(pred_sparse_curiosity_logits, dim=-1), dim=-1))
        tile_curiosity_loss = -torch.mean(torch.sum(
            target_tile_curiosity * torch.log_softmax(pred_tile_curiosity_logits, dim=-1), dim=-1))
        curiosity_loss = sparse_curiosity_loss + tile_curiosity_loss
        curiosity_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.curiosityPredictor.parameters())
            + list(self.tileCuriosityPredictor.parameters()), 10, norm_type=2)
        self.curiosityHeadOptimizer.step()

        # Dream-start priorities: total (sparse + tile) curiosity values.
        dream_priorities = (batch_data["sparse_curiosities"][:, :-1]
                            + batch_data["tile_curiosities"][:, :-1]).reshape(-1).detach()

        metrics = {
            "world_model_loss": world_model_loss.item(),
            "reconstruction_loss": scaled_bt_loss.item(),
            "reward_loss": reward_loss.item(),
            "sparse_reward_loss": sparse_reward_loss.item(),
            "standard_reward_loss": standard_reward_loss.item(),
            "kl_loss": kl_loss.item(),
            "teamitem_loss": teamitem_loss.item(),
            "ltm_reward_loss": ltm_reward_loss.item(),
            "ltm_map_loss": ltm_map_loss.item(),
            "curiosity_loss": curiosity_loss.item(),
            "sparse_curiosity_loss": sparse_curiosity_loss.item(),
            "tile_curiosity_loss": tile_curiosity_loss.item(),
            "decoder_loss": decoder_loss.item(),
        }
        return full_states.view(-1, self.concatenated_dim).detach(), dream_priorities, metrics

    # ------------------------------------------------------
    @staticmethod
    def _first_occurrence_mask(types):
        """Given per-step type ids [N, T], return a float mask [N, T] that is 1.0
        the first time each type id appears in a row and 0.0 for later repeats."""
        eq = types.unsqueeze(2) == types.unsqueeze(1)            # [N, T, T]
        T = types.shape[1]
        prev = torch.tril(torch.ones(T, T, dtype=torch.bool, device=types.device), diagonal=-1)
        seen_before = (eq & prev.unsqueeze(0)).any(dim=2)        # [N, T]
        return (~seen_before).float()

    def Dream(self, full_state, batch_data=None, horizon=25, dream_priorities=None):
        self.actorOptimizer.zero_grad(set_to_none=True)
        self.criticOptimizer.zero_grad(set_to_none=True)
        self.curiosityCriticOptimizer.zero_grad(set_to_none=True)

        start_states = full_state.detach()

        # Resample a fraction of dream starts from high-novelty states.
        if dream_priorities is not None:
            pri = dream_priorities.detach().flatten().to(start_states.device)
            N = start_states.shape[0]
            num_pri = int(N * self.dream_priority_fraction)
            if num_pri > 0 and torch.count_nonzero(pri) > 0:
                probs = pri + 1e-8
                pri_idx = torch.multinomial(probs, num_pri, replacement=True)
                keep_idx = torch.randint(0, N, (N - num_pri,), device=start_states.device)
                start_states = torch.cat([start_states[keep_idx], start_states[pri_idx]], dim=0)

        # --- Imagination rollout ---
        full_states = [start_states]
        log_probabilities, entropies, actions_stack = [], [], []
        curr_state = start_states
        recurrent_state, latent_state = torch.split(curr_state, [self.recurrent_dim, self.latent_dim], -1)
        for _ in range(horizon):
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

        # --- Predicted rewards (ensemble means) and curiosity ---
        with torch.no_grad():
            imagined_steps = full_states[:, 1:]

            def _ensemble_mean(predictor, logits_ensemble):
                return torch.stack(
                    [self.two_hot.decode(logits_ensemble[h]).squeeze(-1)
                     for h in range(predictor.num_heads)], dim=0).mean(dim=0)

            predicted_sparse = _ensemble_mean(self.sparseRewardPredictor,
                                              self.sparseRewardPredictor(imagined_steps))
            predicted_standard = _ensemble_mean(self.standardRewardPredictor,
                                                self.standardRewardPredictor(imagined_steps))

            sparse_curiosity = self.two_hot.decode(self.curiosityPredictor(imagined_steps)).squeeze(-1)
            tile_curiosity = self.two_hot.decode(self.tileCuriosityPredictor(imagined_steps)).squeeze(-1)

            # --- Anti-duplication gate ---
            reward_type = self.ltm_reward_predictor(imagined_steps).argmax(dim=-1)  # [N, T]
            map_type = self.ltm_map_predictor(imagined_steps).argmax(dim=-1)        # [N, T]
            predicted_sparse = predicted_sparse * self._first_occurrence_mask(reward_type)
            predicted_rewards = predicted_sparse + predicted_standard
            sparse_curiosity = sparse_curiosity * self._first_occurrence_mask(map_type)
            predicted_curiosity = sparse_curiosity + tile_curiosity

        imagined_states = full_states.detach()
        online_values = self.two_hot.decode(self.critic(imagined_states)).squeeze(-1)
        online_curiosity_values = self.two_hot.decode(self.curiosity_critic(imagined_states)).squeeze(-1)

        # --- Lambda returns ---
        with torch.no_grad():
            continues = torch.full_like(predicted_rewards, 0.999)
            lambda_values = self.computeLambdaValues(predicted_rewards, online_values, continues)
            curiosity_lambda_values = self.computeLambdaValues(predicted_curiosity, online_curiosity_values, continues)

        # --- Advantages ---
        denominator = self.dynamicDataNormalizer(lambda_values)
        reward_advantages = (lambda_values - online_values[:, :-1].detach()) / denominator
        curiosity_denominator = self.curiosityDynamicDataNormalizer(curiosity_lambda_values)
        curiosity_advantages = (curiosity_lambda_values - online_curiosity_values[:, :-1].detach()) / curiosity_denominator
        combined_advantages = ((1.0 - self.curiosity_scale) * reward_advantages
                               + self.curiosity_scale * curiosity_advantages)

        # --- Actor loss ---
        actor_loss = -torch.mean(torch.mean(
            combined_advantages.detach() * log_probabilities + self.entropy_scale * entropies, dim=1))

        # --- Reward critic loss (CE to lambda returns + EMA KL anchor) ---
        critic_logits_to_train = self.critic(imagined_states)[:, :-1]
        target_values_two_hot = self.two_hot.encode(lambda_values.detach())
        critic_loss_main = -torch.mean(torch.sum(target_values_two_hot * torch.log_softmax(critic_logits_to_train, dim=-1), dim=-1))
        with torch.no_grad():
            ema_critic_logits = self.ema_critic(imagined_states[:, :-1])
            ema_probs = torch.softmax(ema_critic_logits, dim=-1)
        critic_ema_reg = torch.mean(torch.sum(
            ema_probs * (torch.log_softmax(ema_critic_logits, dim=-1) - torch.log_softmax(critic_logits_to_train, dim=-1)), dim=-1))
        critic_loss = critic_loss_main + critic_ema_reg

        # --- Curiosity critic loss (CE to lambda returns + EMA KL anchor) ---
        curiosity_critic_logits_to_train = self.curiosity_critic(imagined_states)[:, :-1]
        target_curiosity_values_two_hot = self.two_hot.encode(curiosity_lambda_values.detach())
        curiosity_critic_loss_main = -torch.mean(torch.sum(
            target_curiosity_values_two_hot * torch.log_softmax(curiosity_critic_logits_to_train, dim=-1), dim=-1))
        with torch.no_grad():
            ema_curiosity_critic_logits = self.ema_curiosity_critic(imagined_states[:, :-1])
            ema_curiosity_probs = torch.softmax(ema_curiosity_critic_logits, dim=-1)
        curiosity_critic_ema_reg = torch.mean(torch.sum(
            ema_curiosity_probs * (torch.log_softmax(ema_curiosity_critic_logits, dim=-1) - torch.log_softmax(curiosity_critic_logits_to_train, dim=-1)), dim=-1))
        curiosity_critic_loss = curiosity_critic_loss_main + curiosity_critic_ema_reg

        # --- Optimization ---
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1, norm_type=2)
        self.criticOptimizer.step()

        curiosity_critic_loss.backward()
        nn.utils.clip_grad_norm_(self.curiosity_critic.parameters(), 1, norm_type=2)
        self.curiosityCriticOptimizer.step()

        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1, norm_type=2)
        self.actorOptimizer.step()

        # --- EMA critic updates ---
        with torch.no_grad():
            for param, ema_param in zip(self.critic.parameters(), self.ema_critic.parameters()):
                ema_param.data.copy_(self.critic_ema_decay * ema_param.data + (1.0 - self.critic_ema_decay) * param.data)
            for param, ema_param in zip(self.curiosity_critic.parameters(), self.ema_curiosity_critic.parameters()):
                ema_param.data.copy_(self.curiosity_critic_ema_decay * ema_param.data + (1.0 - self.curiosity_critic_ema_decay) * param.data)

        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "curiosity_critic_loss": curiosity_critic_loss.item(),
            "entropies": entropies.mean().item(),
            "log_probabilities": log_probabilities.mean().item(),
            "reward_advantages": reward_advantages.mean().item(),
            "advantages": reward_advantages.mean().item(),  # alias for policy.py logging
            "curiosity_advantages": curiosity_advantages.mean().item(),
            "critic_values": online_values.mean().item(),
            "curiosity_critic_values": online_curiosity_values.mean().item(),
            "dream_mean_curiosity": predicted_curiosity.mean().item(),
        }

        # --- Package representative trajectories for visualization ---
        def pack(idx, metric, label):
            return (metric,
                    full_states[idx].detach().cpu(),
                    predicted_rewards[idx].detach().cpu(),
                    lambda_values[idx].detach().cpu(),
                    actions_stack[idx].detach().cpu(),
                    reward_advantages[idx].detach().cpu(),
                    curiosity_advantages[idx].detach().cpu(),
                    label)

        trajectory_advantages = combined_advantages.sum(dim=1)
        best_idx = torch.argmax(trajectory_advantages).item()
        rand_idx = torch.randint(0, trajectory_advantages.shape[0], (1,)).item()
        # MaxCuriosity is meant to surface dreams that imagine entering a NEW map
        # (the sparse +5 stream). Selecting on the total curiosity would let the small,
        # ungated, ever-present tile bonus (+0.01/step) accumulate over the horizon and
        # outrank a single gated +5 spike, hiding the map-transition dreams. So rank the
        # MaxCuriosity pick by the gated sparse stream only.
        trajectory_curiosity = sparse_curiosity.sum(dim=1)
        cur_idx = torch.argmax(trajectory_curiosity).item()

        best_dream_data = pack(best_idx, trajectory_advantages[best_idx].item(), "MaxAdv")
        rand_dream_data = pack(rand_idx, trajectory_advantages[rand_idx].item(), "Random")
        cur_dream_data = pack(cur_idx, trajectory_curiosity[cur_idx].item(), "MaxCuriosity")
        return metrics, best_dream_data, rand_dream_data, cur_dream_data

    # ------------------------------------------------------
    @torch.no_grad()
    def Play_the_game(self, number_of_episodes_per_env=1, epsilon=0.05):
        num_envs = len(self.envs)
        episodes_completed = [0] * num_envs
        scores = []
        current_rewards = [0.0] * num_envs
        local_buffers = [[] for _ in range(num_envs)]
        maps_visited = [set() for _ in range(num_envs)]

        recurrent_state = torch.zeros((num_envs, self.recurrent_dim), device=self.device)
        latent_state = torch.zeros((num_envs, self.latent_dim), device=self.device)
        action = torch.zeros((num_envs, self.action_dim), device=self.device)

        observations, ltm_rewards, ltm_maps, team_levels, item_counts = [], [], [], [], []
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            ltm_rewards.append(np.array(info["ltm_reward"], dtype=np.float32))
            ltm_maps.append(np.array(info["ltm_map"], dtype=np.float32))
            team_levels.append(np.array(info["team_levels"], dtype=np.float32))
            item_counts.append(np.array(info["item_counts"], dtype=np.float32))

        while min(episodes_completed) < number_of_episodes_per_env:
            obs_tensor = (torch.from_numpy(np.array(observations)).float() / 255.0).to(self.device)
            ltm_reward_tensor = torch.from_numpy(np.array(ltm_rewards)).float().to(self.device)
            ltm_map_tensor = torch.from_numpy(np.array(ltm_maps)).float().to(self.device)
            team_tensor = torch.from_numpy(np.array(team_levels)).float().to(self.device)
            item_tensor = torch.from_numpy(np.array(item_counts)).float().to(self.device)

            enc_img, enc_teamitem, enc_ltm_reward, enc_ltm_map = self._encode_components(
                obs_tensor, ltm_reward_tensor, ltm_map_tensor, team_tensor, item_tensor)

            recurrent_state = self.recurrentModel(recurrent_state, latent_state, action)
            posterior_input = torch.cat((recurrent_state, enc_img, enc_teamitem,
                                         enc_ltm_reward, enc_ltm_map), -1)
            latent_state, _ = self.posteriorNet(posterior_input)
            action_onehot, _, _ = self.actor(torch.cat((recurrent_state, latent_state), -1))

            if epsilon > 0.0:
                override = torch.rand(num_envs, device=self.device) < epsilon
                if override.any():
                    rand_actions = torch.randint(0, self.action_dim, (int(override.sum().item()),), device=self.device)
                    action_onehot[override] = F.one_hot(rand_actions, self.action_dim).float()

            action = action_onehot
            action_idxs = torch.argmax(action, dim=-1).cpu().numpy()
            actions_for_buffer = action.cpu().numpy().astype(np.float32)

            for i, env in enumerate(self.envs):
                if episodes_completed[i] < number_of_episodes_per_env:
                    next_observation, reward, terminated, truncated, next_info = env.step(action_idxs[i])
                    done = terminated or truncated
                    self.total_num_steps += 1
                    current_rewards[i] += reward
                    maps_visited[i].add(next_info["coord"][0])
                    sparse_reward = next_info.get("sparse_reward", 0.0)
                    standard_reward = next_info.get("standard_reward", 0.0)
                    sparse_curiosity = next_info.get("sparse_curiosity", 0.0)
                    tile_curiosity = next_info.get("tile_curiosity", 0.0)

                    local_buffers[i].append((
                        observations[i].copy(), ltm_rewards[i].copy(), ltm_maps[i].copy(),
                        item_counts[i].copy(), team_levels[i].copy(),
                        actions_for_buffer[i].copy(), sparse_reward, standard_reward,
                        sparse_curiosity, tile_curiosity))

                    observations[i] = next_observation
                    ltm_rewards[i] = np.array(next_info["ltm_reward"], dtype=np.float32)
                    ltm_maps[i] = np.array(next_info["ltm_map"], dtype=np.float32)
                    team_levels[i] = np.array(next_info["team_levels"], dtype=np.float32)
                    item_counts[i] = np.array(next_info["item_counts"], dtype=np.float32)

                    if done:
                        print(f"    [Env {i+1}] Episode done | Reward: {current_rewards[i]:.2f} | "
                              f"Unique maps visited: {len(maps_visited[i])} {sorted(maps_visited[i])}")
                        maps_visited[i] = set()
                        for transition in local_buffers[i]:
                            self.buffer.add(*transition)
                        local_buffers[i].clear()
                        scores.append(current_rewards[i])
                        self.total_num_episodes += 1
                        episodes_completed[i] += 1

                        if episodes_completed[i] < number_of_episodes_per_env:
                            next_obs, next_info = env.reset()
                            observations[i] = next_obs
                            ltm_rewards[i] = np.array(next_info["ltm_reward"], dtype=np.float32)
                            ltm_maps[i] = np.array(next_info["ltm_map"], dtype=np.float32)
                            team_levels[i] = np.array(next_info["team_levels"], dtype=np.float32)
                            item_counts[i] = np.array(next_info["item_counts"], dtype=np.float32)
                            current_rewards[i] = 0.0
                            recurrent_state[i] = torch.zeros(self.recurrent_dim, device=self.device)
                            latent_state[i] = torch.zeros(self.latent_dim, device=self.device)
                            action[i] = torch.zeros(self.action_dim, device=self.device)

        for i in range(num_envs):
            for transition in local_buffers[i]:
                self.buffer.add(*transition)
            local_buffers[i].clear()

        return round(sum(scores) / len(scores), 2) if scores else 0.0

    # ------------------------------------------------------
    # CHECKPOINTING (clean dict-based save/load with graceful fallback)
    # ------------------------------------------------------
    _CHECKPOINT_MODULES = [
        'recurrentModel', 'posteriorNet', 'priorNet', 'sparseRewardPredictor', 'standardRewardPredictor',  # 'curiosityPredictor',  # fresh init: switched to symlog two-hot
        'image_encoder', 'teamitem_encoder', 'ltm_reward_encoder', 'ltm_map_encoder',
        'teamitemPredictor', 'ltm_reward_predictor', 'ltm_map_predictor', 'projector',
        'actor', 'critic', 'ema_critic', 'curiosity_critic', 'ema_curiosity_critic',
        'dynamicDataNormalizer', 'curiosityDynamicDataNormalizer', 'decoder',
    ]
    _CHECKPOINT_OPTIMIZERS = [
        'worldModelOptimizer', 'actorOptimizer', 'criticOptimizer',
        'curiosityCriticOptimizer', 'curiosityHeadOptimizer', 'decoderOptimizer',
    ]

    def saveCheckpoints(self, path):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        names = self._CHECKPOINT_MODULES + self._CHECKPOINT_OPTIMIZERS
        checkpoint = {name: getattr(self, name).state_dict() for name in names}
        checkpoint.update({
            'total_num_episodes': self.total_num_episodes,
            'total_num_steps': self.total_num_steps,
            'total_num_updates': self.total_num_updates,
        })
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

        buffer_path = os.path.join(directory, "replay_buffer.buffer") if directory else "replay_buffer.buffer"
        print("Saving replay buffer...")
        self.buffer.save(buffer_path)
        return 0

    def loadCheckpoints(self, path=None):
        if path is None:
            checkpoint_files = glob.glob("checkpoints/pokemon_model_R*_G*.pt")
            if not checkpoint_files:
                path = 'model.pt'
            else:
                try:
                    path = max(checkpoint_files, key=lambda x: int(x.split('_G')[-1].split('.pt')[0]))
                except Exception:
                    path = max(checkpoint_files, key=os.path.getctime)

        if not os.path.exists(path):
            print(f"No checkpoint found at {path}, starting from scratch.")
            return 0

        checkpoint = torch.load(path, map_location=self.device)
        # worldModelOptimizer is intentionally not reloaded (kept fresh, matching prior behavior).
        skip_optimizers = {'worldModelOptimizer'}
        for name in self._CHECKPOINT_MODULES + self._CHECKPOINT_OPTIMIZERS:
            if name in skip_optimizers or name not in checkpoint:
                if name not in checkpoint:
                    print(f"[load] '{name}' not in checkpoint -> fresh init.")
                continue
            try:
                getattr(self, name).load_state_dict(checkpoint[name])
            except Exception as e:
                print(f"[load] '{name}' shape/key mismatch -> fresh init ({e}).")

        self.total_num_episodes = checkpoint.get('total_num_episodes', 0)
        self.total_num_steps = checkpoint.get('total_num_steps', 0)
        self.total_num_updates = checkpoint.get('total_num_updates', 0)

        directory = os.path.dirname(path)
        buffer_path = os.path.join(directory, "replay_buffer.buffer") if directory else "replay_buffer.buffer"
        if os.path.exists(buffer_path):
            print("Loading replay buffer...")
            self.buffer.load(buffer_path)
        return 0

    # ------------------------------------------------------
    # DREAM VISUALIZATION (decoded frames + per-step reward/curiosity advantages)
    # ------------------------------------------------------
    @torch.no_grad()
    def visualize_single_dream(self, best_states, best_rewards, best_values, best_actions,
                               reward_advantages=None, curiosity_advantages=None, title_prefix="Dream",
                               pdf=None):
        best_states_device = best_states.to(self.device)

        # Predicted total curiosity per imagined state (sparse + tile heads).
        sparse_cur = self.two_hot.decode(self.curiosityPredictor(best_states_device)).squeeze(-1)
        tile_cur = self.two_hot.decode(self.tileCuriosityPredictor(best_states_device)).squeeze(-1)
        curiosities = (sparse_cur + tile_cur).cpu()

        decoded_imgs = self.decoder(best_states_device).clamp(0.0, 1.0).cpu()  # [horizon, 3, 64, 64]

        horizon = best_states.shape[0]
        action_names = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]
        action_icons = {"UP": "▲ UP", "DOWN": "▼ DN", "LEFT": "◀ LT", "RIGHT": "▶ RT", "A": "A", "B": "B"}

        fig, axes = plt.subplots(2, horizon, figsize=(horizon * 2.2, 5.5), facecolor='#0d1117', dpi=120,
                                 gridspec_kw={'height_ratios': [1.5, 1]})
        total_reward = best_rewards.sum().item() if best_rewards is not None else 0
        fig.suptitle(f'{title_prefix} (total reward = {total_reward:.2f})',
                     color='#58a6ff', fontsize=14, fontweight='bold', y=0.98)
        if horizon == 1:
            axes = np.expand_dims(axes, axis=1)

        def _col(v):
            return '#3fb950' if v > 0 else ('#ff7b72' if v < 0 else '#8b949e')

        for i in range(horizon):
            ax_img, ax_info = axes[0, i], axes[1, i]
            ax_img.imshow(decoded_imgs[i].permute(1, 2, 0).numpy())
            ax_img.axis('off')
            for spine in ax_img.spines.values():
                spine.set_visible(True); spine.set_edgecolor('#30363d'); spine.set_linewidth(1.0)

            ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1); ax_info.axis('off'); ax_info.set_facecolor('#161b22')
            for spine in ax_info.spines.values():
                spine.set_visible(True); spine.set_edgecolor('#30363d'); spine.set_linewidth(0.5)

            if i < horizon - 1:
                step_reward = best_rewards[i].item()
                step_cur = curiosities[i].item()
                step_value = best_values[i].item()
                action_idx = torch.argmax(best_actions[i]).item()
                action_str = action_names[action_idx] if action_idx < len(action_names) else str(action_idx)
                icon = action_icons.get(action_str, action_str)

                # action, reward, curiosity, value, then the two advantages.
                lines = [
                    (icon, '#58a6ff', True),
                    (f'r :{step_reward:+.2f}', _col(step_reward), False),
                    (f'c :{step_cur:+.3f}', '#d1f1a5', False),
                    (f'v :{step_value:.2f}', '#c9d1d9', False),
                ]
                if reward_advantages is not None:
                    a_r = reward_advantages[i].item()
                    lines.append((f'Ar:{a_r:+.2f}', _col(a_r), True))
                if curiosity_advantages is not None:
                    a_c = curiosity_advantages[i].item()
                    lines.append((f'Ac:{a_c:+.2f}', _col(a_c), True))

                y = 0.93
                for txt, col, bold in lines:
                    ax_info.text(0.5, y, txt, ha='center', va='center', fontsize=9,
                                 fontweight='bold' if bold else 'normal', color=col,
                                 fontfamily='monospace', transform=ax_info.transAxes)
                    y -= 0.135
            else:
                step_value_str = f"v:{best_values[i].item():.2f}" if i < len(best_values) else "v: N/A"
                ax_info.text(0.5, 0.78, 'END', ha='center', va='center', fontsize=11,
                             fontweight='bold', color='#8b949e', transform=ax_info.transAxes)
                ax_info.text(0.5, 0.52, step_value_str, ha='center', va='center', fontsize=9,
                             color='#c9d1d9', fontfamily='monospace', transform=ax_info.transAxes)
                ax_info.text(0.5, 0.28, f'c :{curiosities[i].item():+.3f}', ha='center', va='center',
                             fontsize=9, color='#d1f1a5', fontfamily='monospace', transform=ax_info.transAxes)

        plt.subplots_adjust(top=0.85, bottom=0.05, hspace=0.15)
        # If a PdfPages handle is provided, save this dream as a page instead of
        # printing it to the console; otherwise fall back to the old behaviour.
        if pdf is not None:
            pdf.savefig(fig, facecolor=fig.get_facecolor())
        else:
            plt.show()
        plt.close(fig)
