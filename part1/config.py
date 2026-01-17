"""
Configuration file for Q-Learning Gridworld Project
Contains training parameters and level definitions
"""

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Q-Learning and SARSA parameters
ALPHA = 0.1                 # Learning rate
GAMMA = 0.95                # Discount factor
EPSILON_START = 1.0         # Initial exploration rate
EPSILON_END = 0.01          # Minimum exploration rate
EPSILON_DECAY_EPISODES = 4000  # Episodes over which epsilon decays linearly

# Training settings
MAX_EPISODES = 5000         # Maximum training episodes
MAX_STEPS_PER_EPISODE = 200 # Maximum steps per episode (prevent infinite loops)

# Rendering settings
RENDER_DURING_TRAINING = False  # Set to True to visualize during training
RENDER_FREQUENCY = 100          # Render every N episodes if enabled
RENDER_EVALUATION = True        # Always render during final evaluation

# Intrinsic reward settings
USE_INTRINSIC_REWARD = False    # Enable intrinsic rewards (for Level 6)
INTRINSIC_REWARD_BETA = 0.1     # Scaling factor for intrinsic rewards (small to not overwhelm env rewards)

# Monster settings
MONSTER_MOVE_PROBABILITY = 0.4  # Probability monster moves after agent action


# ============================================================================
# DISPLAY PARAMETERS
# ============================================================================

GRID_SIZE = 12              # Grid dimensions (GRID_SIZE x GRID_SIZE)
CELL_SIZE = 50              # Pixel size of each grid cell
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 100  # Extra space for info display
FPS = 10                    # Frames per second for rendering

# Colors (R, G, B)
COLOR_BACKGROUND = (240, 240, 240)
COLOR_GRID_LINE = (200, 200, 200)
COLOR_WALL = (70, 70, 70)
COLOR_FIRE = (255, 69, 0)
COLOR_APPLE = (0, 200, 0)
COLOR_KEY = (255, 215, 0)
COLOR_CHEST = (139, 69, 19)
COLOR_AGENT = (0, 100, 255)
COLOR_MONSTER = (150, 0, 150)
COLOR_TEXT = (0, 0, 0)


# ============================================================================
# ENTITY TYPES
# ============================================================================

EMPTY = 0
WALL = 1
FIRE = 2
APPLE = 3
KEY = 4
CHEST = 5
MONSTER = 6

# Entity symbols for level design
ENTITY_SYMBOLS = {
    '.': EMPTY,
    '#': WALL,
    'F': FIRE,
    'A': APPLE,
    'K': KEY,
    'C': CHEST,
    'M': MONSTER,
}


# ============================================================================
# LEVEL DEFINITIONS
# ============================================================================
# Legend:
# '.' = Empty space
# '#' = Wall/Rock (blocks movement)
# 'F' = Fire (causes death)
# 'A' = Apple (+1 reward)
# 'K' = Key (required to open chest)
# 'C' = Chest (+2 reward, requires key)
# 'M' = Monster (moves randomly, causes death on contact)
# 'S' = Start position (converted to empty space)

LEVELS = {
    # ========================================================================
    # LEVEL 0: Basic Q-Learning
    # Goal: Learn shortest path to apples on the right side
    # No hazards, simple pathfinding task
    # ========================================================================
    0: {
        'name': 'Level 0: Basic Pathfinding',
        'description': 'Simple level with apples on the right. Learn shortest path.',
        'grid': [
            "############",
            "#..........#",
            "#..........#",
            "#..........#",
            "#..........#",
            "#..........#",
            "#..........#",
            "#..........#",
            "#.........A#",
            "#.........A#",
            "#.........A#",
            "############",
        ],
        'start_pos': (1, 1),
    },

    # ========================================================================
    # LEVEL 1: SARSA with Hazards
    # Goal: Navigate around fire to reach apples
    # Tests on-policy learning vs off-policy (Q-Learning)
    # ========================================================================
    1: {
        'name': 'Level 1: Fire Hazards',
        'description': 'Navigate around fire. SARSA should be more conservative.',
        'grid': [
            "############",
            "#..........#",
            "#..........#",
            "#..........#",
            "#....FFF...#",
            "#....FFF...#",
            "#....FFF...#",
            "#..........#",
            "#.........A#",
            "#.........A#",
            "#.........A#",
            "############",
        ],
        'start_pos': (1, 1),
    },

    # ========================================================================
    # LEVEL 2: Key and Chest Introduction
    # Goal: Learn to collect key before opening chest
    # Multiple apples + key-chest mechanics
    # ========================================================================
    2: {
        'name': 'Level 2: Key and Chest',
        'description': 'Collect key to open chest. Multiple rewards.',
        'grid': [
            "############",
            "#..........#",
            "#.A......A.#",
            "#..........#",
            "#....##....#",
            "#....##....#",
            "#..K.##....#",
            "#....##....#",
            "#..........#",
            "#.A......C.#",
            "#..........#",
            "############",
        ],
        'start_pos': (1, 1),
    },

    # ========================================================================
    # LEVEL 3: Complex Multi-Reward
    # Goal: Optimal collection order with obstacles
    # More complex navigation with key-chest and multiple apples
    # ========================================================================
    3: {
        'name': 'Level 3: Complex Navigation',
        'description': 'Complex level with walls, key, chest, and multiple apples.',
        'grid': [
            "############",
            "#A.........#",
            "#.###......#",
            "#.#.#......#",
            "#.#.####...#",
            "#.#....#...#",
            "#.####.#.K.#",
            "#......#...#",
            "#.######...#",
            "#..........#",
            "#A.......CA#",
            "############",
        ],
        'start_pos': (1, 1),
    },

    # ========================================================================
    # LEVEL 4: Monsters Introduction
    # Goal: Learn to avoid moving monsters
    # 40% chance monsters move after each agent action
    # ========================================================================
    4: {
        'name': 'Level 4: Monster Avoidance',
        'description': 'Monsters move randomly. Avoid them while collecting rewards.',
        'grid': [
            "############",
            "#..........#",
            "#..M.......#",
            "#..........#",
            "#....##....#",
            "#....##....#",
            "#....##....#",
            "#....##....#",
            "#......M...#",
            "#.........A#",
            "#.........A#",
            "############",
        ],
        'start_pos': (1, 1),
    },

    # ========================================================================
    # LEVEL 5: Monsters with Key-Chest
    # Goal: Complete objectives while avoiding stochastic monsters
    # Combines key, chest, apples, and monsters (simplified for learning)
    # ========================================================================
    5: {
        'name': 'Level 5: Monsters + Key & Chest',
        'description': 'Navigate monsters while collecting key and opening chest.',
        'grid': [
            "############",
            "#A.........#",
            "#..........#",
            "#..........#",
            "#.....M....#",
            "#..K.......#",
            "#..........#",
            "#..........#",
            "#.......M..#",
            "#.........C#",
            "#.........A#",
            "############",
        ],
        'start_pos': (1, 1),
    },

    # ========================================================================
    # LEVEL 6: Intrinsic Reward Test
    # Goal: Demonstrate exploration bonus with intrinsic rewards
    # Single distant reward at the end of a winding path
    # Intrinsic rewards encourage systematic exploration of the maze
    # ========================================================================
    6: {
        'name': 'Level 6: Exploration Challenge',
        'description': 'Find the hidden reward. Intrinsic rewards aid exploration.',
        'grid': [
            "############",
            "#..........#",
            "#.######.#.#",
            "#.#....#.#.#",
            "#.#.##.#.#.#",
            "#.#.##.#.#.#",
            "#.#....#.#.#",
            "#.######.#.#",
            "#........#.#",
            "#.########.#",
            "#.........A#",
            "############",
        ],
        'start_pos': (1, 1),  # Start at top-left, reward at bottom-right corner
    },
}


# ============================================================================
# ACTIONS
# ============================================================================

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_NAMES = {
    UP: 'UP',
    DOWN: 'DOWN',
    LEFT: 'LEFT',
    RIGHT: 'RIGHT',
}

# Action to delta position mapping
ACTION_DELTA = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
}

NUM_ACTIONS = 4


# ============================================================================
# REWARDS
# ============================================================================

REWARD_APPLE = 1.0
REWARD_CHEST = 2.0
REWARD_KEY = 0.0
REWARD_DEATH = -10.0
REWARD_STEP = 0.0           # Reward for each step (typically 0 or small negative)
