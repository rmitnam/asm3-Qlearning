"""
Gridworld Environment
Implements the core game logic for the Q-Learning gridworld.
"""

import numpy as np
import random
from copy import deepcopy
import config


class GridWorld:
    """
    GridWorld environment for reinforcement learning.
    
    State representation:
    - Levels 0-1: (x, y) - agent position only
    - Levels 2-6: (x, y, has_key) - agent position + key status
    - Levels 4-5: (x, y, has_key, monster_positions...) - includes monster positions
    """
    
    def __init__(self, level_id=0):
        """
        Initialize the gridworld environment.
        
        Args:
            level_id: Which level to load (0-6)
        """
        self.level_id = level_id
        self.level_data = config.LEVELS[level_id]
        
        # Parse the grid
        self.grid_size = config.GRID_SIZE
        self.static_grid = self._parse_grid(self.level_data['grid'])
        
        # Initial positions
        self.start_pos = self.level_data['start_pos']
        
        # State variables
        self.agent_pos = None
        self.has_key = False
        self.monsters = []  # List of (x, y) positions
        self.collected_apples = set()  # Track collected apple positions
        self.chest_opened = False
        
        # Episode tracking
        self.steps = 0
        self.total_reward = 0
        self.done = False
        
        # Intrinsic reward tracking (per episode)
        self.state_visit_counts = {}
        
        # Find initial monster positions and collectibles
        self._initialize_entities()
        
    def _parse_grid(self, grid_strings):
        """
        Parse grid from string representation to 2D numpy array.
        
        Args:
            grid_strings: List of strings representing the grid
            
        Returns:
            2D numpy array with entity type codes
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        for y, row in enumerate(grid_strings):
            for x, char in enumerate(row):
                if char in config.ENTITY_SYMBOLS:
                    grid[y, x] = config.ENTITY_SYMBOLS[char]
                    
        return grid
    
    def _initialize_entities(self):
        """Find all monsters and collectibles in the grid."""
        self.initial_monsters = []
        self.apple_positions = set()
        self.key_position = None
        self.chest_position = None
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                entity = self.static_grid[y, x]
                
                if entity == config.MONSTER:
                    self.initial_monsters.append((x, y))
                elif entity == config.APPLE:
                    self.apple_positions.add((x, y))
                elif entity == config.KEY:
                    self.key_position = (x, y)
                elif entity == config.CHEST:
                    self.chest_position = (x, y)
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state
        """
        self.agent_pos = self.start_pos
        self.has_key = False
        self.monsters = deepcopy(self.initial_monsters)
        self.collected_apples = set()
        self.chest_opened = False
        self.steps = 0
        self.total_reward = 0
        self.done = False
        self.state_visit_counts = {}
        self.last_info = {}  # Store last step info for renderer

        return self.get_state()
    
    def get_state(self):
        """
        Get current state representation.

        Returns:
            State tuple - format depends on level:
            - Levels 0-1: (x, y, collected_apples_tuple)
            - Levels 2-3, 6: (x, y, has_key, chest_opened, collected_apples_tuple)
            - Levels 4-5: (x, y, has_key, chest_opened, collected_apples_tuple, nearest_monster_direction)

        Note: Monster positions are NOT included directly since that causes state explosion.
        Instead, we use the direction to the nearest monster as a simplified representation.
        """
        x, y = self.agent_pos

        # Convert collected apples to a sorted tuple for consistent hashing
        collected_tuple = tuple(sorted(self.collected_apples))

        # For levels with monsters, include nearest monster direction instead of positions
        if len(self.monsters) > 0:
            # Calculate direction to nearest monster (simplified state)
            nearest_dir = self._get_nearest_monster_direction()
            return (x, y, self.has_key, self.chest_opened, collected_tuple, nearest_dir)

        # For levels with key/chest, include has_key and chest_opened
        elif self.key_position is not None or self.chest_position is not None:
            return (x, y, self.has_key, self.chest_opened, collected_tuple)

        # Simple levels: position + collected apples
        else:
            return (x, y, collected_tuple)

    def _get_nearest_monster_direction(self):
        """
        Get a simplified representation of monster threat direction.
        Returns a tuple indicating which quadrant the nearest monster is in.
        This keeps state space manageable while giving some monster awareness.
        """
        if not self.monsters:
            return (0, 0)

        ax, ay = self.agent_pos
        min_dist = float('inf')
        nearest = None

        for mx, my in self.monsters:
            dist = abs(mx - ax) + abs(my - ay)  # Manhattan distance
            if dist < min_dist:
                min_dist = dist
                nearest = (mx, my)

        if nearest is None:
            return (0, 0)

        mx, my = nearest
        # Return relative direction: (-1, 0, 1) for each axis
        dx = 0 if mx == ax else (1 if mx > ax else -1)
        dy = 0 if my == ay else (1 if my > ay else -1)

        # Also encode distance category: close (<=2), medium (3-5), far (>5)
        if min_dist <= 2:
            dist_cat = 0  # danger close
        elif min_dist <= 5:
            dist_cat = 1  # medium
        else:
            dist_cat = 2  # far

        return (dx, dy, dist_cat)
    
    def is_valid_position(self, pos):
        """
        Check if a position is valid (within bounds and not a wall).
        
        Args:
            pos: (x, y) tuple
            
        Returns:
            True if position is valid
        """
        x, y = pos
        
        # Check bounds
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        
        # Check for wall
        if self.static_grid[y, x] == config.WALL:
            return False
        
        return True
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: Action to take (UP, DOWN, LEFT, RIGHT)
            
        Returns:
            (next_state, reward, done, info)
        """
        if self.done:
            raise ValueError("Episode is done. Call reset() to start a new episode.")
        
        self.steps += 1
        reward = config.REWARD_STEP  # Default step reward (usually 0)
        
        # Calculate new position
        dx, dy = config.ACTION_DELTA[action]
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy
        new_pos = (new_x, new_y)
        
        # Check if movement is valid (not wall/boundary)
        if self.is_valid_position(new_pos):
            self.agent_pos = new_pos
        # If invalid, agent stays in place (no movement)
        
        # Check for death (fire or monster collision)
        entity_at_pos = self.static_grid[self.agent_pos[1], self.agent_pos[0]]
        
        if entity_at_pos == config.FIRE:
            reward = config.REWARD_DEATH
            self.done = True
            self.last_info = {'death': 'fire'}
            return self.get_state(), reward, self.done, self.last_info

        if self.agent_pos in self.monsters:
            reward = config.REWARD_DEATH
            self.done = True
            self.last_info = {'death': 'monster'}
            return self.get_state(), reward, self.done, self.last_info
        
        # Check for collectibles
        if entity_at_pos == config.APPLE and self.agent_pos not in self.collected_apples:
            reward += config.REWARD_APPLE
            self.collected_apples.add(self.agent_pos)
        
        elif entity_at_pos == config.KEY and not self.has_key:
            reward += config.REWARD_KEY
            self.has_key = True
        
        elif entity_at_pos == config.CHEST and self.has_key and not self.chest_opened:
            reward += config.REWARD_CHEST
            self.chest_opened = True
        
        # Update visit count for intrinsic reward tracking (but don't add to reward here)
        state = self.get_state()
        if state not in self.state_visit_counts:
            self.state_visit_counts[state] = 0
        self.state_visit_counts[state] += 1

        # Move monsters (probabilistic)
        if len(self.monsters) > 0:
            self._move_monsters()
            
            # Check if monster moved into agent
            if self.agent_pos in self.monsters:
                reward = config.REWARD_DEATH
                self.done = True
                self.last_info = {'death': 'monster'}
                return self.get_state(), reward, self.done, self.last_info

        # Check if episode is complete (all rewards collected)
        if self._all_rewards_collected():
            self.done = True
            self.total_reward += reward
            self.last_info = {'result': 'success'}
            return self.get_state(), reward, self.done, self.last_info

        # Check for timeout
        if self.steps >= config.MAX_STEPS_PER_EPISODE:
            self.done = True
            self.total_reward += reward
            self.last_info = {'result': 'timeout'}
            return self.get_state(), reward, self.done, self.last_info

        self.total_reward += reward
        self.last_info = {}

        return self.get_state(), reward, self.done, self.last_info
    
    def get_intrinsic_reward(self, state):
        """
        Compute intrinsic reward for a state based on visit count.
        Formula: beta / sqrt(n(s) + 1) where n(s) is visits to state s this episode.
        Called by agent during Q-value updates, not added to env reward.
        """
        n_s = self.state_visit_counts.get(state, 0)
        return config.INTRINSIC_REWARD_BETA / np.sqrt(n_s + 1)
    
    def _move_monsters(self):
        """
        Move monsters with probability MONSTER_MOVE_PROBABILITY.
        Monsters choose randomly from valid adjacent positions.
        """
        new_monster_positions = []
        
        for monster_pos in self.monsters:
            # Check if this monster moves
            if random.random() < config.MONSTER_MOVE_PROBABILITY:
                # Get valid adjacent positions
                valid_moves = []
                
                for action in range(config.NUM_ACTIONS):
                    dx, dy = config.ACTION_DELTA[action]
                    new_x = monster_pos[0] + dx
                    new_y = monster_pos[1] + dy
                    new_pos = (new_x, new_y)
                    
                    # Monsters can move to empty spaces (not walls, not fire)
                    if self.is_valid_position(new_pos):
                        entity = self.static_grid[new_y, new_x]
                        if entity != config.FIRE:
                            valid_moves.append(new_pos)
                
                # Choose random valid move, or stay if no valid moves
                if valid_moves:
                    new_monster_positions.append(random.choice(valid_moves))
                else:
                    new_monster_positions.append(monster_pos)
            else:
                # Monster doesn't move
                new_monster_positions.append(monster_pos)
        
        self.monsters = new_monster_positions
    
    def _all_rewards_collected(self):
        """
        Check if all collectible rewards have been obtained.
        
        Returns:
            True if all rewards collected
        """
        # All apples collected
        apples_collected = (len(self.collected_apples) == len(self.apple_positions))
        
        # Chest opened (if chest exists)
        chest_done = (self.chest_position is None) or self.chest_opened
        
        return apples_collected and chest_done
    
    def get_available_actions(self):
        """
        Get list of available actions (always all 4 actions in gridworld).
        
        Returns:
            List of valid actions
        """
        return list(range(config.NUM_ACTIONS))
    
    def render_text(self):
        """
        Render the current state as text (for debugging).
        
        Returns:
            String representation of the grid
        """
        # Create display grid
        display = np.copy(self.static_grid)
        
        # Symbol mapping
        symbols = {
            config.EMPTY: '.',
            config.WALL: '#',
            config.FIRE: 'F',
            config.APPLE: 'A',
            config.KEY: 'K',
            config.CHEST: 'C',
            config.MONSTER: 'M',
        }
        
        # Build string
        lines = []
        for y in range(self.grid_size):
            line = ""
            for x in range(self.grid_size):
                pos = (x, y)
                
                # Agent position
                if pos == self.agent_pos:
                    line += '@'
                # Monster position
                elif pos in self.monsters:
                    line += 'M'
                # Collected apple
                elif pos in self.collected_apples:
                    line += '.'
                # Collected key
                elif pos == self.key_position and self.has_key:
                    line += '.'
                # Opened chest
                elif pos == self.chest_position and self.chest_opened:
                    line += 'c'
                # Static entity
                else:
                    entity = display[y, x]
                    line += symbols.get(entity, '?')
            
            lines.append(line)
        
        # Add status info
        status = f"\nSteps: {self.steps} | Reward: {self.total_reward:.2f} | "
        status += f"Has Key: {self.has_key} | Apples: {len(self.collected_apples)}/{len(self.apple_positions)}"
        
        return '\n'.join(lines) + status
    
    def get_state_space_size(self):
        """
        Estimate the state space size for this level.
        
        Returns:
            Approximate number of possible states
        """
        # Base: number of positions
        num_positions = self.grid_size * self.grid_size
        
        # Multiply by key states if applicable
        if self.key_position is not None:
            num_positions *= 2  # has_key or not
        
        # Monster states are much larger but we approximate
        if len(self.monsters) > 0:
            # This is a huge state space, but most won't be visited
            return num_positions * (num_positions ** len(self.monsters))
        
        return num_positions


if __name__ == "__main__":
    # Quick test of the environment
    print("Testing GridWorld Environment\n")
    
    # Test Level 0
    print("=" * 50)
    print("Level 0: Basic Pathfinding")
    print("=" * 50)
    env = GridWorld(level_id=0)
    state = env.reset()
    print(f"Initial state: {state}")
    print(env.render_text())
    
    # Take a few actions
    print("\nTaking action RIGHT")
    state, reward, done, info = env.step(config.RIGHT)
    print(f"State: {state}, Reward: {reward}, Done: {done}")
    print(env.render_text())
    
    print("\nTaking action DOWN")
    state, reward, done, info = env.step(config.DOWN)
    print(f"State: {state}, Reward: {reward}, Done: {done}")
    print(env.render_text())
    
    # Test Level 2 (with key and chest)
    print("\n" + "=" * 50)
    print("Level 2: Key and Chest")
    print("=" * 50)
    env2 = GridWorld(level_id=2)
    state = env2.reset()
    print(f"Initial state: {state}")
    print(env2.render_text())
    
    print(f"\nState space size estimate: {env.get_state_space_size()}")
    print(f"Level 2 state space size: {env2.get_state_space_size()}")
    
    print("\n✓ GridWorld environment test complete!")
