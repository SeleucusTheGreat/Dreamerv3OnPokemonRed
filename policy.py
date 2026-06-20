import os
import csv
import glob
import random
import numpy as np
import torch
import torch.nn as nn

IMAGE_SIZE = 64
from dreamer import Dreamer


class Policy(nn.Module):
    def __init__(self, envs, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), visualize_dreams=False):
        super(Policy, self).__init__()
        self.device = device
        self.envs = envs
        self.action_dim = envs[0].action_space.n 
        self.buffer_size = 1000000
        self.mlp_dim = 756       # MLP width for all dense models (configurable)
        self.recurrent_dim = 2048
        self.rows = 32
        self.cols = 32
        self.latent_dim = self.rows * self.cols
        self.total_num_episodes = 10000
        self.training_per_episodes = 400
        self.seed = 1234
        self.number_of_sequences = 64 # Batch size
        self.steps_per_sequence = 64
        self.curiosity_scale = 0.5
        self.checkpoint_interval = 2 # Save every N episodes
        
        self.visualize_dreams = visualize_dreams
        self.seedMeDaddy(self.seed)

        self.dreamer = Dreamer(
            device=self.device,
            obs_dim=(3, IMAGE_SIZE, IMAGE_SIZE),
            envs=self.envs,
            action_dim=self.action_dim,
            recurrent_dim=self.recurrent_dim,
            rows=self.rows,
            cols=self.cols,
            latent_dim=self.latent_dim,
            number_of_sequences=self.number_of_sequences,
            steps_per_sequence=self.steps_per_sequence,
            seed=self.seed,
            buffer_size=self.buffer_size,
            team_dim=6,
            item_dim=2,
            curiosity_scale=self.curiosity_scale,
            mlp_dim=self.mlp_dim
        )

    def train(self):
        self.dreamer.loadCheckpoints()

        csv_filename = "pokemon_training_metrics.csv"
        
        headers =[
            "envSteps", "gradientSteps", "totalReward",
            "worldModelLoss", "reconstructionLoss", "rewardPredictorLoss", "klLoss", "teamItemLoss", "ltmRewardLoss", "ltmMapLoss", "curiosityLoss",
            "actorLoss", "entropies", "criticLoss", "curiosityCriticLoss", "advantages", "curiosityAdvantages", "criticValues", "curiosityCriticValues"
        ]
        
        # --- Initialize or Load CSV ---
        if not os.path.exists(csv_filename):
            with open(csv_filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        else:
            try:
                with open(csv_filename, 'r') as f:
                    lines =[line for line in f.read().splitlines() if line.strip()]
                    if len(lines) > 1: # Ignore if just the header
                        last_line = lines[-1].split(',')
                        self.dreamer.total_num_steps = int(float(last_line[0]))
                        self.dreamer.total_num_updates = int(float(last_line[1]))
                        print(f"[*] Recovered CSV Progress: {self.dreamer.total_num_steps} Env Steps | {self.dreamer.total_num_updates} Gradient Steps")
            except Exception as e:
                print(f"[!] Warning: Could not recover steps from CSV: {e}")

        # --- Initial Buffer Fill ---
        buffer_has_transitions = self.dreamer.buffer.full or (self.dreamer.buffer.index > 0)
        if not buffer_has_transitions:
            print("\n" + "="*50)
            print("[*] Replay buffer is empty. Gathering initial data from environment...")
            initial_score = self.dreamer.Play_the_game(number_of_episodes_per_env=2) 
            print(f"[*] Initial Collection Score: {initial_score}")
            print("="*50 + "\n")
        else:
            print("\n" + "="*50)
            #initial_score = self.dreamer.Play_the_game(number_of_episodes_per_env=2)
            print("[*] Replay buffer already contains data. Skipping initial environment collection.")
            print("="*50 + "\n")

        # --- Main Training Loop ---
        print("[*] Starting Training Loop...")
        for episode in range(self.total_num_episodes):
            print(f"\n" + "-"*50)
            print(f"--- Training Episode {episode + 1} / {self.total_num_episodes} ---")
            
            best_dreams_of_episode = []
            random_dreams_of_episode = []
            curiosity_dreams_of_episode = [] # Track curiosity dreams
            wm_metrics = {}

            for step in range(self.training_per_episodes):
                if step % 50 == 0:
                    print(f"    [Training] Step {step} / {self.training_per_episodes}")
                sample = self.dreamer.sample_batch(
                    batchSize=self.number_of_sequences, 
                    sequenceSize=self.steps_per_sequence
                )
                
                # Update Networks
                full_states, dream_priorities, wm_metrics = self.dreamer.TrainWorldModel(sample)
                # Catch all three dreams (dream starts are partially prioritized by actual curiosities)
                dream_metrics, best_dream, rand_dream, cur_dream = self.dreamer.Dream(
                    full_states, batch_data=sample, dream_priorities=dream_priorities
                )
                
                best_dreams_of_episode.append(best_dream)
                random_dreams_of_episode.append(rand_dream)
                curiosity_dreams_of_episode.append(cur_dream)
                self.dreamer.total_num_updates += 1

            if self.visualize_dreams:
                # --- Visualize Half Random, Half Best ---
                total_to_visualize = 10
                half_k = total_to_visualize // 2
                
                # Sample from the pools
                vis_best = random.sample(best_dreams_of_episode, min(half_k, len(best_dreams_of_episode)))
                vis_rand = random.sample(random_dreams_of_episode, min(half_k, len(random_dreams_of_episode)))
                
                # Sort curiosity dreams by total curiosity (descending) and take top 5
                curiosity_dreams_of_episode.sort(key=lambda x: x[0], reverse=True)
                vis_curiosity = curiosity_dreams_of_episode[:5]
                
                dreams_to_visualize = vis_best + vis_rand + vis_curiosity
                
                # Print a quick summary
                best_adv_str = ", ".join([f"{d[0]:+.2f}" for d in vis_best])
                rand_adv_str = ", ".join([f"{d[0]:+.2f}" for d in vis_rand])
                cur_str = ", ".join([f"{d[0]:+.2f}" for d in vis_curiosity])
                print(f"[*] Saving {len(vis_best)} MAX ADVANTAGE Dreams (Cumulative Advs: [{best_adv_str}])")
                print(f"[*] Saving {len(vis_rand)} RANDOM Dreams (Cumulative Advs: [{rand_adv_str}])")
                print(f"[*] Saving {len(vis_curiosity)} MAX CURIOSITY Dreams (Cumulative Curiosity: [{cur_str}])")

                # --- Save this training phase's dreams to a single file in dreams/ ---
                # Keep at most 10 dream files; the oldest is replaced first.
                dream_path = self._save_dreams_to_file(dreams_to_visualize)
                print(f"[*] Saved {len(dreams_to_visualize)} dreams to {dream_path}")
            # --- Play Game with Updated Policy ---
            print(f"[*] Stepping environment with updated policy...")
            avg_score = self.dreamer.Play_the_game(number_of_episodes_per_env=1)

            self.dreamer.buffer.print_diagnostics()
            
            # Print cleanly formatted metrics
            print(f"    > Total Env Steps : {self.dreamer.total_num_steps}")
            print(f"    > Gradient Steps  : {self.dreamer.total_num_updates}")
            print(f"    > Total Reward    : {avg_score:.2f}")
            print(f"    > Actor Entropy   : {dream_metrics.get('entropies', 0):.4f}")
            print(f"    > Mean dream curiosity: {dream_metrics.get('dream_mean_curiosity', 0):.4f}")

            # --- CSV Logging ---
            row_data =[
                self.dreamer.total_num_steps,                     # envSteps
                self.dreamer.total_num_updates,                   # gradientSteps
                avg_score,                                        # totalReward
                wm_metrics.get('world_model_loss', 0),            # worldModelLoss
                wm_metrics.get('reconstruction_loss', 0),         # reconstructionLoss
                wm_metrics.get('reward_loss', 0),                 # rewardPredictorLoss
                wm_metrics.get('kl_loss', 0),                     # klLoss
                wm_metrics.get('teamitem_loss', 0),               # teamItemLoss
                wm_metrics.get('ltm_reward_loss', 0),             # ltmRewardLoss (whole-game LTM)
                wm_metrics.get('ltm_map_loss', 0),                # ltmMapLoss (whole-game map vector)
                wm_metrics.get('curiosity_loss', 0),              # curiosityLoss
                dream_metrics.get('actor_loss', 0),               # actorLoss
                dream_metrics.get('entropies', 0),                # entropies
                dream_metrics.get('critic_loss', 0),              # criticLoss
                dream_metrics.get('curiosity_critic_loss', 0),    # curiosityCriticLoss
                dream_metrics.get('advantages', 0),               # advantages
                dream_metrics.get('curiosity_advantages', 0),     # curiosityAdvantages
                dream_metrics.get('critic_values', 0),            # criticValues
                dream_metrics.get('curiosity_critic_values', 0),  # curiosityCriticValues
            ]
            
            # Round floats for a cleaner CSV
            row_data =[round(x, 4) if isinstance(x, float) else x for x in row_data]
            
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)

            # --- Checkpoint Saving ---
            if episode > 0 and episode % self.checkpoint_interval == 0:
                score_int = int(avg_score) if avg_score is not None else 0
                filename = f"pokemon_model_R{score_int}_G{self.dreamer.total_num_updates}.pt"
                path = os.path.join("checkpoints", filename)
                self.dreamer.saveCheckpoints(path)

                # Rotation: keep only the 15 most recent checkpoints (oldest out first).
                ckpts = sorted(
                    glob.glob(os.path.join("checkpoints", "pokemon_model_R*_G*.pt")),
                    key=os.path.getmtime,
                )
                while len(ckpts) > 15:
                    oldest = ckpts.pop(0)
                    try:
                        os.remove(oldest)
                    except OSError as e:
                        print(f"[!] Warning: could not remove old checkpoint {oldest}: {e}")

    def _save_dreams_to_file(self, dreams_to_visualize):
        """Save all dreams from one training phase into a single PDF inside the
        `dreams/` folder. Keeps at most 10 files, replacing the oldest first."""
        from matplotlib.backends.backend_pdf import PdfPages

        dreams_dir = "dreams"
        os.makedirs(dreams_dir, exist_ok=True)

        # Named by gradient-step count, e.g. dream_g5000.pdf.
        filename = f"dream_g{self.dreamer.total_num_updates}.pdf"
        file_path = os.path.join(dreams_dir, filename)

        with PdfPages(file_path) as pdf:
            for dream_data in dreams_to_visualize:
                metric_val, b_states, b_rewards, b_values, b_actions, b_adv_r, b_adv_c, label = dream_data

                # Change the title dynamically based on the sorting metric.
                if label == "MaxCuriosity":
                    title_pref = f"{label} Dream (Total Curiosity: {metric_val:+.2f})"
                else:
                    title_pref = f"{label} Dream (Total Adv: {metric_val:+.2f})"

                self.dreamer.visualize_single_dream(
                    b_states.to(self.device),
                    b_rewards,
                    b_values,
                    b_actions,
                    b_adv_r,
                    b_adv_c,
                    title_prefix=title_pref,
                    pdf=pdf,
                )

        # --- Rotation: keep only the 10 most recent dream files ---
        existing = sorted(
            glob.glob(os.path.join(dreams_dir, "*.pdf")),
            key=os.path.getmtime,
        )
        while len(existing) > 10:
            oldest = existing.pop(0)
            try:
                os.remove(oldest)
            except OSError as e:
                print(f"[!] Warning: could not remove old dream file {oldest}: {e}")

        return file_path

    def evaluate(self, num_episodes=1):
        """ Evaluates BOTH agents simultaneously """
        self.dreamer.loadCheckpoints()
        print(f"Starting Parallel Evaluation for {num_episodes} episodes per agent...")
        
        num_envs = len(self.envs)
        episodes_completed = [0] * num_envs
        current_rewards =[0.0] * num_envs
        steps = [0] * num_envs
        
        recurrent_state = torch.zeros((num_envs, self.recurrent_dim), device=self.device)
        latent_state = torch.zeros((num_envs, self.latent_dim), device=self.device)
        action_onehot = torch.zeros((num_envs, self.action_dim), device=self.device)
        
        observations = []
        ltm_rewards = []
        ltm_maps = []
        team_levels = []
        item_counts = []
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            ltm_rewards.append(np.array(info["ltm_reward"], dtype=np.float32))
            ltm_maps.append(np.array(info["ltm_map"], dtype=np.float32))
            team_levels.append(np.array(info["team_levels"], dtype=np.float32))
            item_counts.append(np.array(info["item_counts"], dtype=np.float32))

        while min(episodes_completed) < num_episodes:
            obs_tensor = (torch.from_numpy(np.array(observations)).float() / 255.0).to(self.device)
            ltm_reward_tensor = torch.from_numpy(np.array(ltm_rewards)).float().to(self.device)
            ltm_map_tensor = torch.from_numpy(np.array(ltm_maps)).float().to(self.device)
            team_tensor = torch.from_numpy(np.array(team_levels)).float().to(self.device)
            item_tensor = torch.from_numpy(np.array(item_counts)).float().to(self.device)

            with torch.no_grad():
                enc_img, enc_teamitem, enc_ltm_reward, enc_ltm_map = self.dreamer._encode_components(
                    obs_tensor, ltm_reward_tensor, ltm_map_tensor, team_tensor, item_tensor)
                encoded_obs = torch.cat([enc_img, enc_teamitem, enc_ltm_reward, enc_ltm_map], dim=-1)

                recurrent_state = self.dreamer.recurrentModel(recurrent_state, latent_state, action_onehot)
                latent_state, _ = self.dreamer.posteriorNet(torch.cat((recurrent_state, encoded_obs), -1))
                
                full_state = torch.cat((recurrent_state, latent_state), -1)
                action_onehot, _, _ = self.dreamer.actor(full_state)
                action_idxs = torch.argmax(action_onehot, dim=-1).cpu().numpy()
                
            for i, env in enumerate(self.envs):
                if episodes_completed[i] < num_episodes:
                    obs, reward, terminated, truncated, info = env.step(action_idxs[i])
                    done = terminated or truncated
                    
                    observations[i] = obs
                    ltm_rewards[i] = np.array(info["ltm_reward"], dtype=np.float32)
                    ltm_maps[i] = np.array(info["ltm_map"], dtype=np.float32)
                    team_levels[i] = np.array(info["team_levels"], dtype=np.float32)
                    item_counts[i] = np.array(info["item_counts"], dtype=np.float32)
                    current_rewards[i] += reward
                    steps[i] += 1
                    
                    if steps[i] % 500 == 0:
                        print(f"[Agent {i+1}] Eval Step: {steps[i]}/{info['limit']} | Reward: {current_rewards[i]:.3f}")
                        
                    if done:
                        print(f"--- [Agent {i+1}] Episode Finished! Reward: {current_rewards[i]:.3f} | Steps: {steps[i]} ---")
                        episodes_completed[i] += 1
                        
                        if episodes_completed[i] < num_episodes:
                            obs, info = env.reset()
                            observations[i] = obs
                            ltm_rewards[i] = np.array(info["ltm_reward"], dtype=np.float32)
                            ltm_maps[i] = np.array(info["ltm_map"], dtype=np.float32)
                            team_levels[i] = np.array(info["team_levels"], dtype=np.float32)
                            item_counts[i] = np.array(info["item_counts"], dtype=np.float32)
                            current_rewards[i] = 0.0
                            steps[i] = 0
                            recurrent_state[i] = torch.zeros(self.recurrent_dim, device=self.device)
                            latent_state[i] = torch.zeros(self.latent_dim, device=self.device)
                            action_onehot[i] = torch.zeros(self.action_dim, device=self.device)

    def seedMeDaddy(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True   # your input shapes are fixed (64x64), so autotuning pays off