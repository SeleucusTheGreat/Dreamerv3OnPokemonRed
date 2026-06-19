import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os
from pyboy import PyBoy
from pyboy.utils import WindowEvent

# ==========================================================
# LONG-TERM MEMORY / MAP DIMENSIONS (whole-game scope)
# ==========================================================
RAM_EVENTS_REGION_START = 0xD73E
RAM_EVENTS_REGION_END = 0xD860           
NUM_EVENT_BITS = (RAM_EVENTS_REGION_END - RAM_EVENTS_REGION_START) * 8  # 2320
LTM_REWARD_DIM = NUM_EVENT_BITS + 1       # +1 pokeball flag = 2321

# --- PROLOGUE REWARD SUPPRESSION ---
WEVENTFLAGS_START = 0xD747
EVENT_GOT_POKEDEX = 0x025
PROLOGUE_LTM_END = (WEVENTFLAGS_START - RAM_EVENTS_REGION_START) * 8 + EVENT_GOT_POKEDEX + 1  # 110

# Noisy event flags to exclude from reward, by LTM index
IGNORED_EVENT_INDICES = (1032,)

# Maps to track for the curiosity bonus
MONITORED_MAPS = [
    # --- Pallet -> Pewter (Brock) ---
    12,    # ROUTE_1
    1,     # VIRIDIAN_CITY
    13,    # ROUTE_2
    51,    # VIRIDIAN_FOREST
    2,     # PEWTER_CITY
    54,    # PEWTER_GYM            (Brock, Boulder Badge)
    # --- Route 3 -> Mt. Moon -> Cerulean (Misty) ---
    14,    # ROUTE_3
    59,    # MT_MOON_1F
    60,    # MT_MOON_B1F
    61,    # MT_MOON_B2F
    15,    # ROUTE_4
    3,     # CERULEAN_CITY
    65,    # CERULEAN_GYM          (Misty, Cascade Badge)
    35,    # ROUTE_24             (Nugget Bridge)
    36,    # ROUTE_25             (Bill -> S.S. Ticket)
    # --- Cerulean -> Vermilion (Surge) ---
    16,    # ROUTE_5
    17,    # ROUTE_6
    5,     # VERMILION_CITY
    95,    # SS_ANNE_1F
    96,    # SS_ANNE_2F
    101,   # SS_ANNE_CAPTAINS_ROOM (HM01 Cut)
    92,    # VERMILION_GYM         (Lt. Surge, Thunder Badge)
    # --- Route 9/10 -> Rock Tunnel -> Lavender ---
    20,    # ROUTE_9
    21,    # ROUTE_10
    82,    # ROCK_TUNNEL_1F
    232,   # ROCK_TUNNEL_B1F
    4,     # LAVENDER_TOWN
    # --- Lavender -> Celadon (Erika) -> Rocket Hideout ---
    19,    # ROUTE_8
    18,    # ROUTE_7
    6,     # CELADON_CITY
    134,   # CELADON_GYM           (Erika, Rainbow Badge)
    135,   # GAME_CORNER           (Rocket Hideout entrance)
    199,   # ROCKET_HIDEOUT_B1F
    200,   # ROCKET_HIDEOUT_B2F
    201,   # ROCKET_HIDEOUT_B3F
    202,   # ROCKET_HIDEOUT_B4F    (Silph Scope)
    # --- Pokemon Tower (Poke Flute) ---
    142,   # POKEMON_TOWER_1F
    143,   # POKEMON_TOWER_2F
    144,   # POKEMON_TOWER_3F
    145,   # POKEMON_TOWER_4F
    146,   # POKEMON_TOWER_5F
    147,   # POKEMON_TOWER_6F
    148,   # POKEMON_TOWER_7F      (Mr. Fuji -> Poke Flute)
    # --- Saffron -> Silph Co. (Giovanni) -> Sabrina ---
    10,    # SAFFRON_CITY
    181,   # SILPH_CO_1F
    207,   # SILPH_CO_2F
    208,   # SILPH_CO_3F
    209,   # SILPH_CO_4F
    210,   # SILPH_CO_5F
    211,   # SILPH_CO_6F
    212,   # SILPH_CO_7F
    213,   # SILPH_CO_8F
    233,   # SILPH_CO_9F
    234,   # SILPH_CO_10F
    235,   # SILPH_CO_11F          (Giovanni -> Master Ball)
    178,   # SAFFRON_GYM           (Sabrina, Marsh Badge)
    # --- Route 12-15 -> Fuchsia (Koga) -> Safari Zone (Surf/Strength) ---
    23,    # ROUTE_12
    24,    # ROUTE_13
    25,    # ROUTE_14
    26,    # ROUTE_15
    7,     # FUCHSIA_CITY
    157,   # FUCHSIA_GYM           (Koga, Soul Badge)
    220,   # SAFARI_ZONE_CENTER
    217,   # SAFARI_ZONE_EAST
    218,   # SAFARI_ZONE_NORTH
    219,   # SAFARI_ZONE_WEST
    222,   # SAFARI_ZONE_SECRET_HOUSE (HM03 Surf; Gold Teeth -> HM04 Strength)
    # --- Surf to Cinnabar -> Pokemon Mansion (Secret Key) -> Blaine ---
    30,    # ROUTE_19
    31,    # ROUTE_20
    8,     # CINNABAR_ISLAND
    165,   # POKEMON_MANSION_1F
    214,   # POKEMON_MANSION_2F
    215,   # POKEMON_MANSION_3F
    216,   # POKEMON_MANSION_B1F   (Secret Key)
    166,   # CINNABAR_GYM          (Blaine, Volcano Badge)
    # --- Viridian Gym (Giovanni) -> Victory Road -> Elite Four ---
    45,    # VIRIDIAN_GYM          (Giovanni, Earth Badge)
    33,    # ROUTE_22
    34,    # ROUTE_23
    108,   # VICTORY_ROAD_1F
    194,   # VICTORY_ROAD_2F
    198,   # VICTORY_ROAD_3F
    9,     # INDIGO_PLATEAU
    174,   # INDIGO_PLATEAU_LOBBY
    245,   # LORELEIS_ROOM
    246,   # BRUNOS_ROOM
    247,   # AGATHAS_ROOM
    113,   # LANCES_ROOM
    120,   # CHAMPIONS_ROOM        (rival, final battle)
    118,   # HALL_OF_FAME          (game complete)
]
LTM_MAP_DIM = len(MONITORED_MAPS)
MONITORED_MAPS_SET = set(MONITORED_MAPS)

# Map-transition curiosity: the ONLY exploration/curiosity signal. Granted once per
# episode the first time the agent enters each monitored map. There is no per-tile
# or coverage curiosity anymore.
MAP_CURIOSITY_BONUS = 5.0


class PokemonRedEnv(gym.Env):

    # --- RAM Address Constants ---
    RAM_MAP_ID = 0xD35E
    RAM_PLAYER_Y = 0xD361
    RAM_PLAYER_X = 0xD362

    RAM_PARTY_COUNT = 0xD163
    RAM_PARTY_BASE = 0xD16B
    PKMN_DATA_LENGTH = 44

    RAM_NUM_BAG_ITEMS = 0xD31D
    RAM_BAG_ITEMS_BASE = 0xD31E
    ITEM_POKEBALL = 0x04

    def __init__(self, rom_path, state_path, image_size=64, verbose=False, window="SDL2", speed=0):
        super().__init__()

        self.state_path = state_path
        self.image_size = image_size
        self.verbose = verbose
        self.has_obtained_pokeball = False

        self.visited_maps = set()

        # Whole-game long-term event memory: snapshot of the start-state event bits, used
        # to mask out flags that were already set so the LTM vector reflects *new* progress.
        self.start_event_bits = np.zeros(NUM_EVENT_BITS, dtype=np.float32)
        self.last_event_bits = np.zeros(NUM_EVENT_BITS, dtype=np.float32)

        # Reward-eligibility mask: zero out the prologue region (idx 0..109, up to GOT_POKEDEX)
        # and any explicitly ignored noisy flags, so neither the event reward nor the LTM
        # reward vector ever pays for prologue/noise flags.
        self.reward_keep_mask = np.ones(NUM_EVENT_BITS, dtype=np.float32)
        self.reward_keep_mask[:PROLOGUE_LTM_END] = 0.0
        for idx in IGNORED_EVENT_INDICES:
            self.reward_keep_mask[idx] = 0.0

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
        self.max_level_reward = 0
        self.last_hp_fraction_sum = 0
        self.last_party_count = 0
        self.level_reward_decay_rate = 0.85
        self.episode_level_ups = 0 
        self.brock_defeated = False
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
        brock_before = self.brock_defeated
        self._apply_action(action)
        self.current_step += 1
        # Reward is split into two streams that are summed for the env return:
        #   sparse_reward   -> memory-tied events (event flags, Brock, first pokeball).
        #                      These correspond to changes in the LTM reward memory and
        #                      are the ones gated by the LTM predictor during the dream.
        #   standard_reward -> non-memory events (healing, level-ups).
        sparse_reward = 0.000
        standard_reward = 0.000

        coord = self._get_current_position()
        map_id, x, y = coord
        
        # --- Map-transition curiosity (the ONLY curiosity signal): +MAP_CURIOSITY_BONUS
        # the first time each monitored map is entered this episode, 0 otherwise. ---
        curiosity = 0.0
        if map_id in MONITORED_MAPS_SET and map_id not in self.curiosity_triggered_maps:
            curiosity = MAP_CURIOSITY_BONUS
            self.curiosity_triggered_maps.add(map_id)
            if self.verbose:
                print(f"[CURIOSITY BONUS] +{MAP_CURIOSITY_BONUS} awarded for transitioning to Map ID: {map_id}")

        self.visited_maps.add(map_id)

        # EVENT REWARD: +reward_event_val per newly-set, reward-eligible (post-prologue) flag.
        current_event_bits = self._get_new_event_bits()
        current_events = int(current_event_bits.sum())

        if current_events > self.max_events:
            new_events = current_events - self.max_events
            event_reward_gain = new_events * self.reward_event_val

            newly_on = np.where((current_event_bits > 0.5) & (self.last_event_bits < 0.5))[0]
            for idx in newly_on:
                idx = int(idx)
                addr = RAM_EVENTS_REGION_START + idx // 8
                bit = idx % 8
                print(f"[EVENT REWARD] Triggered by Memory Address: {hex(addr)}, Bit: {bit} (LTM idx {idx})")

            self.max_step_limit += (new_events * self.step_increase)
            self.max_events = current_events
            sparse_reward += event_reward_gain

            if self.verbose:
                print(f"[EVENT] +{event_reward_gain:.3f} | Budget: {self.max_step_limit}")

        self.last_event_bits = current_event_bits

        # Check for Brock defeat milestone reward
        brock_after = (self.pyboy.memory[0xD356] & 1) == 1
        if not brock_before and brock_after:
            self.brock_defeated = True
            # Beating Brock should yield +200 rather than the normal +50.
            # Since the event reward gain already added +50 (or will add +50),
            # we add +50 to make the total +200.
            sparse_reward += 50.0
            if self.verbose:
                print(f"[BROCK DEFEAT] +50.0 added (total 100.0 for beating Brock)")

        # PARTY HEALING REWARD
        total_level, hp_fraction_sum, party_count = self._get_party_info()

        if hp_fraction_sum > self.last_hp_fraction_sum and party_count == self.last_party_count:
            if self.last_hp_fraction_sum > 0.0:  # Avoid healing reward on respawn
                heal_gain = hp_fraction_sum - self.last_hp_fraction_sum
                heal_reward = self.reward_heal_mult * heal_gain
                standard_reward += heal_reward
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
            standard_reward += decayed_lvl_gain
            
            self.max_level_reward = current_lvl_val
            self.episode_level_ups += 1  
            
            if self.verbose:
                print(f"[LEVEL] +{decayed_lvl_gain:.3f} | Total Party Level up to: {total_level}")

        # ITEM REWARD
        if not self.has_obtained_pokeball:
            if self._has_pokeball():
                self.has_obtained_pokeball = True
                sparse_reward += 50.0
                if self.verbose:
                    print("[ITEM] +50.0 | Obtained first Pokéball!")

        # Total env reward is the sum of the two streams (unchanged externally).
        step_reward = sparse_reward + standard_reward

        # Termination & Obs
        terminated = self.current_step >= self.max_step_limit
        obs = self._get_obs()

        info = {
            "coord": coord,
            "curiosity": curiosity,
            "sparse_reward": sparse_reward,
            "standard_reward": standard_reward,
            "steps": self.current_step, 
            "limit": self.max_step_limit,
            "events": current_events,
            "total_level": total_level,
            "ltm_reward": self._get_ltm_reward(),
            "ltm_map": self._get_ltm_map(),
            "team_levels": self._get_team_levels(),
            "item_counts": self._get_item_counts(),
        }

        return obs, step_reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        
        with open(self.state_path, "rb") as f:
            self.pyboy.load_state(f)
            
        self.current_step = 0
        self.max_step_limit = self.init_steps
        self.episode_level_ups = 0  # Reset the decay counter cleanly
        
        # Reset tracking sets cleanly
        self.visited_maps = set()

        # --- Track map curiosity triggers for the current episode ---
        self.curiosity_triggered_maps = set()
    
        self.pyboy.tick()
        
        start_coord = self._get_current_position()
        start_map = start_coord[0]
        self.visited_maps.add(start_map)
        
        # Avoid awarding curiosity for simply spawning on a monitored map
        if start_map in MONITORED_MAPS_SET:
            self.curiosity_triggered_maps.add(start_map)

        # Snapshot the whole-game event bits already set in the start state, so the LTM
        # reward vector only reflects new progress made during the episode.
        self.start_event_bits = self._get_event_bits()
        self.last_event_bits = self._get_new_event_bits()  # all zeros after masking

        # Initial Party and Event Baseline (max_events is now 0: prologue + start flags masked).
        self.max_events = int(self.last_event_bits.sum())
        total_level, hp_fraction_sum, party_count = self._get_party_info()
        self.last_hp_fraction_sum = hp_fraction_sum
        self.last_party_count = party_count
        self.max_level_reward = self._calculate_level_reward(total_level)
        self.has_obtained_pokeball = self._has_pokeball()
        self.brock_defeated = (self.pyboy.memory[0xD356] & 1) == 1

        info = {
            "coord": start_coord,
            "curiosity": 0.0,  # No curiosity on spawn; only awarded on new-map entry
            "events": self.max_events,
            "total_level": total_level,
            "ltm_reward": self._get_ltm_reward(),
            "ltm_map": self._get_ltm_map(),
            "team_levels": self._get_team_levels(),
            "item_counts": self._get_item_counts(),
        }
        return self._get_obs(), info

    def close(self):
        self.pyboy.stop()

    # ==========================================================
    # WHOLE-GAME LONG-TERM MEMORY HELPERS
    # ==========================================================

    def _get_event_bits(self):
        """Raw 2320-length 0/1 vector of the whole-game event-flag RAM region."""
        block = np.frombuffer(
            bytes(self.pyboy.memory[RAM_EVENTS_REGION_START:RAM_EVENTS_REGION_END]),
            dtype=np.uint8,
        )
        return np.unpackbits(block, bitorder="little").astype(np.float32)

    def _get_new_event_bits(self):
        """Reward-eligible event bits: start-state flags and the prologue/ignored region
        masked out, so only genuinely new post-prologue progress is counted."""
        return self._get_event_bits() * (1.0 - self.start_event_bits) * self.reward_keep_mask

    def _get_ltm_reward(self):
        """Whole-game long-term reward memory: new event bits + pokeball flag (LTM_REWARD_DIM)."""
        vec = np.empty(LTM_REWARD_DIM, dtype=np.float32)
        vec[:NUM_EVENT_BITS] = self._get_new_event_bits()
        vec[NUM_EVENT_BITS] = 1.0 if self.has_obtained_pokeball else 0.0
        return vec

    def _get_ltm_map(self):
        """Whole-game long-term map memory: 1.0 once a monitored map is visited this episode."""
        return np.array(
            [1.0 if m in self.visited_maps else 0.0 for m in MONITORED_MAPS],
            dtype=np.float32,
        )

# The one and only start state allowed. Every environment loads from this file.
START_STATE = "PokemonRed.Start.state"


def create_envs(num_envs, rom_path, state_dir="StartingFiles", wind="null"):
    """Create `num_envs` environments, all loading exclusively from PokemonRed.Start.state.

    This project intentionally supports a single start state. No other .state files
    exist or are accepted.
    """
    state_path = os.path.join(state_dir, START_STATE)
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"Required start state not found: {state_path}. "
            f"Only '{START_STATE}' is supported."
        )

    envs = []
    print(f"Initializing {num_envs} PyBoy Environments (all using {START_STATE})...")

    for i in range(num_envs):
        env = PokemonRedEnv(
            rom_path=rom_path,
            state_path=state_path,
            verbose=True,
            window=wind,
            speed=0,
        )
        envs.append(env)

    return envs