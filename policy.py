import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn

IMAGE_SIZE = 64
from dreamer import Dreamer


class Policy(nn.Module):
    def __init__(self, envs, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Policy, self).__init__()
        self.device = device
        self.envs = envs
        self.action_dim = envs[0].action_space.n 
        self.buffer_size = 1000000
        self.recurrent_dim = 1024 
        self.rows = 32           
        self.cols = 32
        self.latent_dim = self.rows * self.cols
        self.total_num_episodes = 10000
        self.training_per_episodes = 200
        self.seed = 1234
        self.number_of_sequences = 64 # Batch size
        self.steps_per_sequence = 64
        self.curiosity_scale = 0.1  
        
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
            ram_dim=14,
            team_dim=6,
            item_dim=2,
            curiosity_scale=self.curiosity_scale
        )

    def train(self):
        self.dreamer.loadCheckpoints()
        csv_filename = "pokemon_training_metrics.csv"
        
        headers =[
            "envSteps", "gradientSteps", "totalReward", 
            "worldModelLoss", "reconstructionLoss", "rewardPredictorLoss", "klLoss", "goalLoss", "teamLoss", "itemLoss", "curiosityLoss",
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
        print("\n" + "="*50)
        print("[*] Gathering initial data from environment...")
        initial_score = self.dreamer.Play_the_game(number_of_episodes_per_env=0) 
        print(f"[*] Initial Collection Score: {initial_score}")
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
                full_states, wm_metrics = self.dreamer.TrainWorldModel(sample)
                # Catch all three dreams
                dream_metrics, best_dream, rand_dream, cur_dream = self.dreamer.Dream(full_states, batch_data=sample)
                
                best_dreams_of_episode.append(best_dream)
                random_dreams_of_episode.append(rand_dream)
                curiosity_dreams_of_episode.append(cur_dream)
                self.dreamer.total_num_updates += 1

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
            print(f"[*] Visualizing {len(vis_best)} MAX ADVANTAGE Dreams (Cumulative Advs: [{best_adv_str}])")
            print(f"[*] Visualizing {len(vis_rand)} RANDOM Dreams (Cumulative Advs: [{rand_adv_str}])")
            print(f"[*] Visualizing {len(vis_curiosity)} MAX CURIOSITY Dreams (Cumulative Curiosity: [{cur_str}])")
            
            for rank, dream_data in enumerate(dreams_to_visualize):
                metric_val, b_states, b_rewards, b_values, b_actions, b_advantages, label = dream_data
                
                # Change the text dynamically based on what sorting metric was passed
                if label == "MaxCuriosity":
                    title_pref = f"{label} Dream (Total Curiosity: {metric_val:+.2f})"
                else:
                    title_pref = f"{label} Dream (Total Adv: {metric_val:+.2f})"
                
                self.dreamer.visualize_single_dream(
                    b_states.to(self.device), 
                    b_rewards, 
                    b_values, 
                    b_actions, 
                    b_advantages,
                    title_prefix=title_pref
                )
            # --- Play Game with Updated Policy ---
            print(f"[*] Stepping environment with updated policy...")
            avg_score = self.dreamer.Play_the_game(number_of_episodes_per_env=1)

            self.dreamer.buffer.print_diagnostics()
            
            # Print cleanly formatted metrics
            print(f"    > Total Env Steps : {self.dreamer.total_num_steps}")
            print(f"    > Gradient Steps  : {self.dreamer.total_num_updates}")
            print(f"    > Total Reward    : {avg_score:.2f}")
            print(f"    > Actor Entropy   : {dream_metrics.get('entropies', 0):.4f}")

            # --- CSV Logging ---
            row_data =[
                self.dreamer.total_num_steps,                     # envSteps
                self.dreamer.total_num_updates,                   # gradientSteps
                avg_score,                                        # totalReward
                wm_metrics.get('world_model_loss', 0),            # worldModelLoss
                wm_metrics.get('reconstruction_loss', 0),         # reconstructionLoss
                wm_metrics.get('reward_loss', 0),                 # rewardPredictorLoss
                wm_metrics.get('kl_loss', 0),                     # klLoss
                wm_metrics.get('goal_loss', 0),                   # goalLoss
                wm_metrics.get('team_loss', 0),                   # teamLoss
                wm_metrics.get('item_loss', 0),                   # itemLoss
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
            if episode > 0 and episode % 3 == 0:
                score_int = int(avg_score) if avg_score is not None else 0
                filename = f"pokemon_model_R{score_int}_G{self.dreamer.total_num_updates}.pt"
                path = os.path.join("checkpoints", filename)
                self.dreamer.saveCheckpoints(path)

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
        rams = []
        team_levels = []
        item_counts = []
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            rams.append(np.array(info["milestones"], dtype=np.float32))
            team_levels.append(np.array(info["team_levels"], dtype=np.float32))
            item_counts.append(np.array(info["item_counts"], dtype=np.float32))
            
        while min(episodes_completed) < num_episodes:
            obs_tensor = (torch.from_numpy(np.array(observations)).float() / 255.0).to(self.device)
            ram_tensor = torch.from_numpy(np.array(rams)).float().to(self.device)
            team_tensor = torch.from_numpy(np.array(team_levels)).float().to(self.device)
            item_tensor = torch.from_numpy(np.array(item_counts)).float().to(self.device)
            
            with torch.no_grad():
                encoded_img = self.dreamer.image_encoder(obs_tensor)
                encoded_goal = self.dreamer.goal_encoder(ram_tensor)
                encoded_team = self.dreamer.team_encoder(team_tensor / 100.0)
                encoded_item = self.dreamer.item_encoder(item_tensor / 10.0)
                encoded_obs = torch.cat([encoded_img, encoded_goal, encoded_team, encoded_item], dim=-1)
                
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
                    rams[i] = np.array(info["milestones"], dtype=np.float32)
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
                            rams[i] = np.array(info["milestones"], dtype=np.float32)
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
        torch.backends.cudnn.deterministic = True