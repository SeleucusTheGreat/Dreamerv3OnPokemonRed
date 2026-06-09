import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os
from pyboy import PyBoy
from pyboy.utils import WindowEvent


class PokemonRedEnv(gym.Env):

    # --- RAM Address Constants ---
    RAM_MAP_ID = 0xD35E
    RAM_PLAYER_Y = 0xD361
    RAM_PLAYER_X = 0xD362
    
    RAM_BATTLE_STATE = 0xD057 
    
    RAM_PARTY_COUNT = 0xD163
    RAM_PARTY_BASE = 0xD16B
    PKMN_DATA_LENGTH = 44
    
    RAM_EVENTS_START = 0xD73E
    RAM_EVENTS_END = 0xD85F

    RAM_NUM_BAG_ITEMS = 0xD31D
    RAM_BAG_ITEMS_BASE = 0xD31E
    ITEM_POKEBALL = 0x04

    # Retained original 14-element milestone names for compatibility
    MILESTONE_NAMES = [
        "Left House", "Route 1", "Viridian City", "Rival Route 22",
        "Route 2", "Viridian Forest", "Viridian Trainer", "Pewter City",
        "Defeated Brock", "Route 3", "Mt. Moon", "Cerulean City",
        "Defeated Misty", "Obtained Pokeball"
    ]

    def __init__(self, rom_path, state_path, image_size=64, verbose=False, window="SDL2", speed=0, advanced=False):
        super().__init__()
        
        self.state_path = state_path
        self.image_size = image_size
        self.verbose = verbose
        self.advanced = advanced 
        self.has_obtained_pokeball = False
        
        self.visited_maps = set()
        
        # Start counting events later in memory if using the advanced start
        self.ram_events_start = 0xD74B if self.advanced else self.RAM_EVENTS_START

        # Initialize Emulator
        self.pyboy = PyBoy(
            rom_path, 
            window=window,
            sound_volume=0, 
            cgb=True
        )
        self.pyboy.set_emulation_speed(speed) # 0 = Unlimited speed for training
        
        # Budget Constants
        self.init_steps = 20000


        # --- REWARDS CONSTANTS ---
        self.step_increase = 2000
        self.reward_event_val = 50   
        self.reward_heal_mult = 2.5
        self.reward_lvl_mult = 5
        
        # Action Space Definition (7 buttons)
        self.buttons =[
            (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
            (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
            (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
            (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
            (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
            (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B)
        ]
        self.action_space = spaces.Discrete(len(self.buttons))
        
        # Observation Space
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(3, self.image_size, self.image_size), 
            dtype=np.uint8
        )
        
        # Persistent Episode Trackers
        self.current_step = 0
        self.max_step_limit = self.init_steps
        
        self.max_events = 0
        self.last_active_events = set()
        self.max_level_reward = 0
        self.last_hp_fraction_sum = 0
        self.last_party_count = 0
        self.level_reward_decay_rate = 0.85
        self.episode_level_ups = 0 

    # ==========================================================
    # EMULATOR & RAM HELPERS
    # ==========================================================

    def _has_pokeball(self):
        """Checks the player's Bag RAM to see if a Pokéball (ID: 0x04) is present."""
        num_items = self.pyboy.memory[self.RAM_NUM_BAG_ITEMS]
        num_items = min(max(num_items, 0), 20)
        
        for i in range(num_items):
            item_addr = self.RAM_BAG_ITEMS_BASE + (i * 2)
            item_id = self.pyboy.memory[item_addr]
            if item_id == self.ITEM_POKEBALL:
                return True
        return False

    def _get_item_counts(self):
        """Counts total number of Pokéballs (0x04) and standard Potions (0x14) in the bag."""
        num_items = self.pyboy.memory[self.RAM_NUM_BAG_ITEMS]
        num_items = min(max(num_items, 0), 20)
        pokeballs = 0
        potions = 0
        for i in range(num_items):
            item_addr = self.RAM_BAG_ITEMS_BASE + (i * 2)
            item_id = self.pyboy.memory[item_addr]
            quantity = self.pyboy.memory[item_addr + 1]
            if item_id == self.ITEM_POKEBALL:
                pokeballs += quantity
            elif item_id == 0x14:  # Potion
                potions += quantity
        return [float(pokeballs), float(potions)]

    def _get_team_levels(self):
        """Returns the raw level of each of the 6 party slot positions (0.0 if empty)."""
        party_count = min(max(self.pyboy.memory[self.RAM_PARTY_COUNT], 0), 6)
        levels = [0.0] * 6
        for i in range(party_count):
            base_addr = self.RAM_PARTY_BASE + (i * self.PKMN_DATA_LENGTH)
            levels[i] = float(self.pyboy.memory[base_addr + 0x21])
        return levels

    def _get_current_position(self):
        """Returns the player's current (map_id, x, y) coordinates."""
        map_id = self.pyboy.memory[self.RAM_MAP_ID]
        x = self.pyboy.memory[self.RAM_PLAYER_X]
        y = self.pyboy.memory[self.RAM_PLAYER_Y]
        return (map_id, x, y)

    def _get_active_events(self):
        """Reads the event RAM range and returns a set of (address, bit) pairs for all bits set to 1."""
        active_events = set()
        for addr in range(self.ram_events_start, self.RAM_EVENTS_END):
            val = self.pyboy.memory[addr]
            if val:
                for i in range(8):
                    if val & (1 << i):
                        if addr == 0xD7BF and i == 0:
                            continue
                        active_events.add((addr, i))
        return active_events

    def _get_party_info(self):
        party_count = min(max(self.pyboy.memory[self.RAM_PARTY_COUNT], 0), 6)
        total_level = 0
        hp_fraction_sum = 0.0
        
        for i in range(party_count):
            base_addr = self.RAM_PARTY_BASE + (i * self.PKMN_DATA_LENGTH)
            
            # Level & HP
            total_level += self.pyboy.memory[base_addr + 0x21]
            current_hp = (self.pyboy.memory[base_addr + 0x01] << 8) | self.pyboy.memory[base_addr + 0x02]
            max_hp = (self.pyboy.memory[base_addr + 0x22] << 8) | self.pyboy.memory[base_addr + 0x23]
            
            if max_hp > 0:
                hp_fraction_sum += (current_hp / max_hp)
                
        return total_level, hp_fraction_sum, party_count

    def _calculate_level_reward(self, total_level):
        """Applies Equation (2) from the paper."""
        if total_level <= 22:
            val = total_level
        else:
            val = ((total_level - 22) / 4) + 22
        return self.reward_lvl_mult * val

    def _get_obs(self):
        screen = self.pyboy.screen.ndarray 
        if screen.shape[-1] == 4:
            screen = screen[:, :, :3]
        img = cv2.resize(screen, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        
        # Transpose from (H, W, C) to PyTorch's expected (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        return np.array(img, dtype=np.uint8)

    def _apply_action(self, action):
        press, release = self.buttons[action]
        self.pyboy.send_input(press)
        self.pyboy.tick(16) 
        self.pyboy.send_input(release)
        self.pyboy.tick(8)

    # ==========================================================
    # GYM METHODS
    # ==========================================================

    def step(self, action):
        self._apply_action(action)
        self.current_step += 1
        step_reward = 0.000
        
        coord = self._get_current_position()
        map_id, x, y = coord
        self.visited_maps.add(map_id)

        # EVENT REWARD 
        current_active_events = self._get_active_events()
        current_events = len(current_active_events)
        
        if current_events > self.max_events:
            new_events = current_events - self.max_events
            event_reward_gain = new_events * self.reward_event_val
            
            new_bits = current_active_events - self.last_active_events
            for addr, bit in new_bits:
                print(f"[EVENT REWARD] Triggered by Memory Address: {hex(addr)}, Bit: {bit}")
            
            self.max_step_limit += (new_events * self.step_increase)
            self.max_events = current_events
            step_reward += event_reward_gain
            
            if self.verbose:
                print(f"[EVENT] +{event_reward_gain:.3f} | Budget: {self.max_step_limit}")

        self.last_active_events = current_active_events

        # PARTY HEALING REWARD
        total_level, hp_fraction_sum, party_count = self._get_party_info()

        if hp_fraction_sum > self.last_hp_fraction_sum and party_count == self.last_party_count:
            if self.last_hp_fraction_sum > 0.0:  # Avoid healing reward on respawn
                heal_gain = hp_fraction_sum - self.last_hp_fraction_sum
                heal_reward = self.reward_heal_mult * heal_gain
                step_reward += heal_reward
                if self.verbose:
                    print(f"[HEAL] +{heal_reward:.3f} | HP recovered: {heal_gain:.2f}")

        self.last_hp_fraction_sum = hp_fraction_sum
        self.last_party_count = party_count

        # LEVEL UP REWARD 
        current_lvl_val = self._calculate_level_reward(total_level)
        if current_lvl_val > self.max_level_reward:
            lvl_gain = current_lvl_val - self.max_level_reward
            
            # Decay 
            decayed_lvl_gain = lvl_gain * (self.level_reward_decay_rate ** self.episode_level_ups)
            step_reward += decayed_lvl_gain
            
            self.max_level_reward = current_lvl_val
            self.episode_level_ups += 1  
            
            if self.verbose:
                print(f"[LEVEL] +{decayed_lvl_gain:.3f} | Total Party Level up to: {total_level}")

        # ITEM REWARD
        if not self.has_obtained_pokeball:
            if self._has_pokeball():
                self.has_obtained_pokeball = True 
                step_reward += 50.0
                if self.verbose:
                    print("[ITEM] +50.0 | Obtained first Pokéball!")

        # Termination & Obs
        terminated = self.current_step >= self.max_step_limit
        obs = self._get_obs()
        
        info = {
            "coord": coord, 
            "steps": self.current_step, 
            "limit": self.max_step_limit,
            "events": current_events,
            "total_level": total_level,
            "milestones": self._get_milestones(),
            "team_levels": self._get_team_levels(),
            "item_counts": self._get_item_counts()
        }
        
        return obs, step_reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        
        with open(self.state_path, "rb") as f:
            self.pyboy.load_state(f)
            
        self.current_step = 0
        self.max_step_limit = self.init_steps
        self.episode_level_ups = 0  # Reset the decay counter cleanly
        
        # Reset tracking coordinates cleanly
        self.visited_maps = set()
    
        self.pyboy.tick()
        
        start_coord = self._get_current_position()
        self.visited_maps.add(start_coord[0]) 
        
        # Initial Party and Event Baseline
        self.last_active_events = self._get_active_events()
        self.max_events = len(self.last_active_events)
        total_level, hp_fraction_sum, party_count = self._get_party_info()
        self.last_hp_fraction_sum = hp_fraction_sum
        self.last_party_count = party_count
        self.max_level_reward = self._calculate_level_reward(total_level)
        self.has_obtained_pokeball = self._has_pokeball()
        
        milestones = self._get_milestones()
        
        info = {
            "coord": start_coord,
            "events": self.max_events,
            "total_level": total_level,
            "milestones": milestones,
            "team_levels": self._get_team_levels(),
            "item_counts": self._get_item_counts()
        }
        return self._get_obs(), info

    def close(self):
        self.pyboy.stop()
    
    def _get_milestones(self):
        """Returns a 14-dim vector of macro-level geographic and event milestones."""
        mem = self.pyboy.memory
        
        # MACRO GEOGRAPHY (Map IDs)
        left_house      = 1.0 if 0 in self.visited_maps else 0.0
        route_1         = 1.0 if 1 in self.visited_maps else 0.0
        viridian        = 1.0 if 2 in self.visited_maps else 0.0
        route_2         = 1.0 if 3 in self.visited_maps else 0.0
        viridian_forest = 1.0 if 51 in self.visited_maps else 0.0
        enter_brock_gym = 1.0 if 54 in self.visited_maps else 0.0  
        route_3         = 1.0 if 5 in self.visited_maps else 0.0
        mt_moon         = 1.0 if 59 in self.visited_maps else 0.0
        cerulean        = 1.0 if 6 in self.visited_maps else 0.0
        rival_route_22   = 1.0 if (mem[0xD74A] & 0b00000001) else 0.0
        viridian_trainer = 1.0 if (mem[0xD751] & 0b00000010) else 0.0
        brock    = 1.0 if (mem[0xD356] >> 0) & 1 else 0.0
        exit_forest = 1.0 if 49 in self.visited_maps else 0.0      
        pokeball = 1.0 if self.has_obtained_pokeball else 0.0
        
        return [
            left_house, route_1, viridian, rival_route_22,
            route_2, viridian_forest, viridian_trainer, enter_brock_gym,
            brock, route_3, mt_moon, cerulean,
            exit_forest, pokeball
        ]

class RotatingPokemonRedEnv(PokemonRedEnv):
    def __init__(self, rom_path, state_paths, env_id, **kwargs):
        if isinstance(state_paths, str):
            state_paths = [state_paths]
            
        self.state_paths = state_paths
        self.current_state_idx = 0
        self.env_id = env_id 
        
        super().__init__(rom_path, state_path=self.state_paths[0], **kwargs)

    def reset(self, seed=None, options=None):
        self.state_path = self.state_paths[self.current_state_idx]
        print(f" -> Env {self.env_id} loaded with {len(self.state_paths)} rotating state(s). Current state: {self.state_path}")
        self.current_state_idx = (self.current_state_idx + 1) % len(self.state_paths)
        return super().reset(seed=seed, options=options)


def create_envs(num_envs, rom_path, state_dir="StartingFiles", wind="null"):
    start_state = "PokemonRed.Start.state"
    other_states = ["PokemonRed.PowerStart.state"]
    
    envs = []
    print(f"Initializing {num_envs} PyBoy Environments...")
    
    for i in range(num_envs):
        if i < 3:
            paths = [os.path.join(state_dir, start_state)]
        else:
            start_idx = (i - 3) % len(other_states)
            rotated_states = other_states[start_idx:] + other_states[:start_idx]
            paths = [os.path.join(state_dir, s) for s in rotated_states]
        
        env = RotatingPokemonRedEnv(
            rom_path=rom_path, 
            state_paths=paths, 
            env_id=i+1,        
            verbose=True, 
            window=wind, 
            speed=0, 
            advanced=True 
        )
        envs.append(env)
        
    return envs