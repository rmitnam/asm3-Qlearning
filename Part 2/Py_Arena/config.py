"""
Configuration file for the Arena RL Environment
Contains all game constants, hyperparameters, and settings
"""

# =============================================================================
# WINDOW SETTINGS
# =============================================================================
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 680
FPS = 60

# =============================================================================
# COLORS (RGB)
# =============================================================================
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 150, 255)
YELLOW = (255, 255, 50)
ORANGE = (255, 165, 0)
PURPLE = (150, 50, 255)
CYAN = (50, 255, 255)
DARK_GRAY = (40, 40, 40)
LIGHT_GRAY = (100, 100, 100)

# =============================================================================
# PLAYER SETTINGS
# =============================================================================
PLAYER_SIZE = 20
PLAYER_MAX_HEALTH = 100
PLAYER_MAX_SPEED = 5.0
PLAYER_ACCELERATION = 0.3
PLAYER_FRICTION = 0.98
PLAYER_ROTATION_SPEED = 5.0  # degrees per frame
PLAYER_SHOOT_COOLDOWN = 10  # frames between shots
PLAYER_INVINCIBILITY_FRAMES = 30  # frames of invincibility after taking damage

# =============================================================================
# ENEMY SETTINGS
# =============================================================================
ENEMY_SIZE = 15
ENEMY_HEALTH = 30
ENEMY_SPEED = 2.0
ENEMY_DAMAGE = 10
ENEMY_SPAWN_DELAY = 60  # frames between enemy spawns per spawner

# =============================================================================
# SPAWNER SETTINGS
# =============================================================================
SPAWNER_SIZE = 30
SPAWNER_HEALTH = 100
SPAWNER_SPAWN_RATE = 180  # frames between spawns

# =============================================================================
# PROJECTILE SETTINGS
# =============================================================================
PROJECTILE_SIZE = 5
PROJECTILE_SPEED = 10.0
PROJECTILE_DAMAGE = 20
PROJECTILE_LIFETIME = 90  # frames before projectile disappears

# =============================================================================
# PHASE SETTINGS
# =============================================================================
INITIAL_SPAWNERS = 2
SPAWNERS_PER_PHASE = 1  # additional spawners per phase
MAX_PHASE = 5
PHASE_SPAWN_RATE_REDUCTION = 5  # spawn faster each phase

# =============================================================================
# EPISODE SETTINGS
# =============================================================================
MAX_EPISODE_STEPS = 3000  # maximum steps per episode
STEP_PENALTY = -0.001  # small penalty per step to encourage efficiency

# =============================================================================
# REWARD SETTINGS
# =============================================================================
REWARD_KILL_ENEMY = 10.0
REWARD_KILL_SPAWNER = 50.0
REWARD_PHASE_PROGRESS = 100.0
REWARD_DAMAGE_TAKEN = -5.0
REWARD_DEATH = -200.0
REWARD_SURVIVAL_BONUS = 0.1  # per step alive
REWARD_APPROACH_ENEMY = 0.5  # for getting closer to enemies/spawners
REWARD_SHOOT_HIT = 2.0  # bonus for projectile hitting target

# =============================================================================
# OBSERVATION SETTINGS
# =============================================================================
# Observation vector size breakdown:
# - Player position (2): x, y normalized
# - Player velocity (2): vx, vy normalized
# - Player orientation (2): cos(angle), sin(angle) for rotation mode
# - Player health (1): normalized 0-1
# - Current phase (1): normalized
# - Nearest enemy info (4): distance, angle, relative_x, relative_y
# - Nearest spawner info (4): distance, angle, relative_x, relative_y
# - Number of enemies (1): normalized
# - Number of spawners (1): normalized
# - Additional nearby enemies (8): up to 4 more enemies (distance, angle each)
# Total: ~26 features

MAX_ENEMIES_IN_OBS = 5
MAX_SPAWNERS_IN_OBS = 3
OBSERVATION_SIZE_ROTATION = 28
OBSERVATION_SIZE_DIRECTIONAL = 26  # no orientation needed

# =============================================================================
# TRAINING HYPERPARAMETERS - PPO (Optimized for Arena Survival)
# =============================================================================
# Learning rate: Start higher for faster initial learning, use schedule for stability
PPO_LEARNING_RATE = 2.5e-4  # Slightly lower for stability in combat scenarios

# Rollout settings: More steps for better advantage estimation in long episodes
PPO_N_STEPS = 4096  # Increased for better trajectory sampling (captures more combat sequences)
PPO_BATCH_SIZE = 256  # Larger batches for more stable gradient estimates
PPO_N_EPOCHS = 15  # More epochs to extract more learning from each rollout

# Discount factors: Tuned for survival game dynamics
PPO_GAMMA = 0.995  # Higher discount for long-term survival planning
PPO_GAE_LAMBDA = 0.98  # Higher lambda for less bias in advantage estimation

# PPO clipping: Conservative clipping for stable learning
PPO_CLIP_RANGE = 0.15  # Slightly tighter clipping for more stable updates
PPO_CLIP_RANGE_VF = 0.2  # Clip value function updates (reduces value function overestimation)

# Entropy and value coefficients: Encourage exploration early
PPO_ENT_COEF = 0.02  # Higher entropy for more exploration (important for combat variety)
PPO_VF_COEF = 0.5  # Standard value function coefficient
PPO_MAX_GRAD_NORM = 0.5  # Gradient clipping for stability

# Neural network architecture: Deeper network for complex spatial reasoning
POLICY_NET_ARCH = [512, 256, 128]  # 3-layer network for better feature extraction

# =============================================================================
# TRAINING HYPERPARAMETERS - DQN (alternative)
# =============================================================================
DQN_LEARNING_RATE = 1e-4
DQN_BUFFER_SIZE = 100000
DQN_LEARNING_STARTS = 1000
DQN_BATCH_SIZE = 64
DQN_TAU = 1.0
DQN_GAMMA = 0.99
DQN_TRAIN_FREQ = 4
DQN_TARGET_UPDATE_INTERVAL = 1000
DQN_EXPLORATION_FRACTION = 0.1
DQN_EXPLORATION_FINAL_EPS = 0.05

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
TOTAL_TIMESTEPS = 1000000  # Increased for better convergence
EVAL_FREQ = 20000  # Evaluate less frequently for faster training
N_EVAL_EPISODES = 10  # More episodes for reliable evaluation
LOG_DIR = "logs"
MODEL_DIR = "models"
N_ENVS = 8  # More parallel environments for better sample diversity

# =============================================================================
# ACTION SPACES
# =============================================================================
# Rotation control scheme (expanded with combined actions)
ROTATION_ACTIONS = {
    0: "NO_ACTION",
    1: "THRUST_FORWARD",
    2: "ROTATE_LEFT",
    3: "ROTATE_RIGHT",
    4: "SHOOT",
    5: "THRUST_SHOOT",
    6: "ROTATE_LEFT_SHOOT",
    7: "ROTATE_RIGHT_SHOOT"
}

# Directional control scheme
DIRECTIONAL_ACTIONS = {
    0: "NO_ACTION",
    1: "MOVE_UP",
    2: "MOVE_DOWN",
    3: "MOVE_LEFT",
    4: "MOVE_RIGHT",
    5: "SHOOT"
}
