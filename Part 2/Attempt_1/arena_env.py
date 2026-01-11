import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Game Constants
PLAYER_SPEED = 5
ROTATION_SPEED = 5
BULLET_SPEED = 10
ENEMY_SPEED = 2
SPAWNER_HP = 50
ENEMY_HP = 10
PLAYER_HP = 30

class SpaceArenaEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    control_style: 1 = Thrust/Rotate, 2 = Direct Directional
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, control_style=1, render_mode=None):
        super(SpaceArenaEnv, self).__init__()
        
        self.control_style = control_style
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Define Action Space
        if self.control_style == 1:
            # 0: No-op, 1: Thrust, 2: Rotate Left, 3: Rotate Right, 4: Shoot
            self.action_space = spaces.Discrete(5)
        else:
            # 0: No-op, 1: Up, 2: Down, 3: Left, 4: Right, 5: Shoot
            self.action_space = spaces.Discrete(6)

        # Define Observation Space
        # Size 14: 
        # [Player X, Player Y, Player VX, Player VY, Player Angle, Player HP,
        #  Nearest Enemy Dist, Nearest Enemy Angle, 
        #  Nearest Spawner Dist, Nearest Spawner Angle,
        #  Can Shoot (bool), Phase, Relative X to Spawner, Relative Y to Spawner]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game State
        self.player_pos = np.array([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.player_angle = 0.0  # Degrees
        self.player_hp = PLAYER_HP
        self.phase = 1
        self.step_count = 0
        self.max_steps = 2000
        self.shoot_cooldown = 0
        
        # Entities
        self.bullets = []
        self.enemies = []
        self._spawn_spawners()
        
        if self.render_mode == "human" and self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

        return self._get_obs(), {}

    def _spawn_spawners(self):
        """Create spawners based on phase difficulty"""
        self.spawners = []
        count = 1 + (self.phase // 2) # Difficulty progression
        for _ in range(count):
            x = np.random.randint(50, SCREEN_WIDTH - 50)
            y = np.random.randint(50, SCREEN_HEIGHT - 50)
            # Ensure not spawning on top of player
            while np.linalg.norm([x - self.player_pos[0], y - self.player_pos[1]]) < 100:
                x = np.random.randint(50, SCREEN_WIDTH - 50)
                y = np.random.randint(50, SCREEN_HEIGHT - 50)
            
            self.spawners.append({
                'pos': np.array([x, y], dtype=float),
                'hp': SPAWNER_HP,
                'timer': 0
            })

    def step(self, action):
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False
        
        # 1. Apply Actions (Physics)
        self._handle_player_movement(action)
        self._handle_shooting(action)
        
        # 2. Game Logic
        # Update Bullets
        bullets_to_remove = set()
        for b in self.bullets:
            b['pos'] += b['vel']
            # Remove if out of bounds
            if not (0 <= b['pos'][0] <= SCREEN_WIDTH and 0 <= b['pos'][1] <= SCREEN_HEIGHT):
                bullets_to_remove.add(id(b))

        # Update Spawners (Spawn Logic)
        active_spawners = [s for s in self.spawners if s['hp'] > 0]
        if not active_spawners:
            # Phase Complete!
            reward += 100  # Big reward for progression
            self.phase += 1
            self.player_hp = min(self.player_hp + 20, PLAYER_HP) # Heal slightly
            self._spawn_spawners()
        else:
            for s in active_spawners:
                s['timer'] += 1
                if s['timer'] > 120: # Spawn enemy every 2 seconds
                    s['timer'] = 0
                    self.enemies.append({
                        'pos': s['pos'].copy(),
                        'hp': ENEMY_HP
                    })

        # Update Enemies (Tracking Player)
        for e in self.enemies:
            direction = self.player_pos - e['pos']
            dist = np.linalg.norm(direction)
            if dist > 0:
                e['pos'] += (direction / dist) * ENEMY_SPEED

        # 3. Collision Detection
        enemies_to_remove = set()
        # Bullets hitting Enemies or Spawners
        for b in list(self.bullets):
            hit = False
            # Check Enemies
            for e in self.enemies[:]:
                if np.linalg.norm(b['pos'] - e['pos']) < 15:
                    e['hp'] -= 10
                    if e['hp'] <= 0:
                        enemies_to_remove.add(id(e))
                        reward += 5 # Reward for kill
                    hit = True
                    break
            
            # Check Spawners if bullet didn't hit enemy
            if not hit:
                for s in self.spawners:
                    if s['hp'] > 0 and np.linalg.norm(b['pos'] - s['pos']) < 25:
                        s['hp'] -= 10
                        hit = True
                        if s['hp'] <= 0:
                            reward += 20 # Reward for destroying spawner
                        break
            
            if hit:
                bullets_to_remove.add(id(b))

        # Enemies hitting Player
        for e in self.enemies[:]:
            if np.linalg.norm(e['pos'] - self.player_pos) < 20:
                self.player_hp -= 10
                enemies_to_remove.add(id(e)) # Enemy explodes on impact
                reward -= 10 # Penalty for taking damage

        # Actually remove bullets marked for deletion (use identity to avoid
        # triggering numpy array equality comparisons on dict remove)
        if bullets_to_remove:
            self.bullets = [bb for bb in self.bullets if id(bb) not in bullets_to_remove]
        # Remove enemies marked for deletion by identity as well
        if enemies_to_remove:
            self.enemies = [ee for ee in self.enemies if id(ee) not in enemies_to_remove]
        
        # 4. Termination Conditions
        if self.player_hp <= 0:
            terminated = True
            reward -= 50 # Strong penalty for death
        
        if self.step_count >= self.max_steps:
            truncated = True

        # Shaping Reward: Slight penalty for existing (encourage speed)
        reward -= 0.05

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _handle_player_movement(self, action):
        accel = 0.5
        friction = 0.95
        
        if self.control_style == 1: # Rotate and Thrust
            # 1: Thrust, 2: Left, 3: Right
            if action == 2: self.player_angle -= ROTATION_SPEED
            if action == 3: self.player_angle += ROTATION_SPEED
            
            if action == 1:
                rad = math.radians(self.player_angle)
                self.player_vel[0] += math.cos(rad) * accel
                self.player_vel[1] += math.sin(rad) * accel
            
            self.player_pos += self.player_vel
            self.player_vel *= friction # Space friction
            
        elif self.control_style == 2: # Direct Directional
            # 1: Up, 2: Down, 3: Left, 4: Right
            desired_vel = np.array([0.0, 0.0])
            if action == 1: desired_vel[1] = -PLAYER_SPEED
            elif action == 2: desired_vel[1] = PLAYER_SPEED
            elif action == 3: desired_vel[0] = -PLAYER_SPEED
            elif action == 4: desired_vel[0] = PLAYER_SPEED
            
            # Simple easing
            self.player_pos += desired_vel

        # Boundaries
        self.player_pos = np.clip(self.player_pos, 0, [SCREEN_WIDTH, SCREEN_HEIGHT])

    def _handle_shooting(self, action):
        shoot_action = 4 if self.control_style == 1 else 5
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        
        if action == shoot_action and self.shoot_cooldown == 0:
            self.shoot_cooldown = 15
            rad = math.radians(self.player_angle)
            if self.control_style == 2:
                # In style 2, shoot towards nearest enemy or just 'up' if none?
                # Let's keep it simple: shoot up, or add auto-aim.
                # Requirement: "Shoot". Let's default to shooting UP or Last Movement Dir.
                # For RL stability, let's fix it to shooting UP (angle -90) for Style 2
                rad = math.radians(-90) 

            vel = np.array([math.cos(rad), math.sin(rad)]) * BULLET_SPEED
            self.bullets.append({'pos': self.player_pos.copy(), 'vel': vel})

    def _get_obs(self):
        # Normalize positions to [0, 1] for better NN performance
        norm_pos_x = self.player_pos[0] / SCREEN_WIDTH
        norm_pos_y = self.player_pos[1] / SCREEN_HEIGHT
        
        # Find nearest enemy
        nearest_enemy_dist = 1.0
        nearest_enemy_angle = 0.0
        if self.enemies:
            dists = [np.linalg.norm(e['pos'] - self.player_pos) for e in self.enemies]
            min_idx = np.argmin(dists)
            nearest_enemy_dist = dists[min_idx] / 1000.0 # simple norm
            
            # Relative angle
            diff = self.enemies[min_idx]['pos'] - self.player_pos
            nearest_enemy_angle = math.atan2(diff[1], diff[0])

        # Find nearest spawner
        nearest_spawn_dist = 1.0
        nearest_spawn_angle = 0.0
        rel_spawn_x = 0.0
        rel_spawn_y = 0.0
        
        active_spawners = [s for s in self.spawners if s['hp'] > 0]
        if active_spawners:
            dists = [np.linalg.norm(s['pos'] - self.player_pos) for s in active_spawners]
            min_idx = np.argmin(dists)
            nearest_spawn_dist = dists[min_idx] / 1000.0
            diff = active_spawners[min_idx]['pos'] - self.player_pos
            nearest_spawn_angle = math.atan2(diff[1], diff[0])
            rel_spawn_x = diff[0] / SCREEN_WIDTH
            rel_spawn_y = diff[1] / SCREEN_HEIGHT

        obs = np.array([
            norm_pos_x, norm_pos_y,
            self.player_vel[0] / 10.0, self.player_vel[1] / 10.0,
            math.radians(self.player_angle),
            self.player_hp / 100.0,
            nearest_enemy_dist,
            nearest_enemy_angle,
            nearest_spawn_dist,
            nearest_spawn_angle,
            1.0 if self.shoot_cooldown == 0 else 0.0,
            self.phase / 10.0,
            rel_spawn_x,
            rel_spawn_y
        ], dtype=np.float32)
        return obs

    def render(self):
        if self.window is None: return
        
        self.window.fill((0, 0, 0))
        
        # Draw Spawners (Blue Squares)
        for s in self.spawners:
            if s['hp'] > 0:
                color = (0, 0, 255)
                rect = pygame.Rect(s['pos'][0]-15, s['pos'][1]-15, 30, 30)
                pygame.draw.rect(self.window, color, rect)
                # Health bar
                pygame.draw.rect(self.window, (0,255,0), (s['pos'][0]-15, s['pos'][1]-25, 30 * (s['hp']/SPAWNER_HP), 5))

        # Draw Enemies (Red Circles)
        for e in self.enemies:
            pygame.draw.circle(self.window, (255, 0, 0), e['pos'].astype(int), 10)

        # Draw Player (White Triangle)
        # Calculate triangle points based on rotation
        rad = math.radians(self.player_angle)
        front = self.player_pos + np.array([math.cos(rad), math.sin(rad)]) * 20
        left = self.player_pos + np.array([math.cos(rad + 2.5), math.sin(rad + 2.5)]) * 15
        right = self.player_pos + np.array([math.cos(rad - 2.5), math.sin(rad - 2.5)]) * 15
        pygame.draw.polygon(self.window, (255, 255, 255), [front, left, right])

        # Draw Bullets (Yellow)
        for b in self.bullets:
            pygame.draw.circle(self.window, (255, 255, 0), b['pos'].astype(int), 3)

        # HUD
        font = pygame.font.Font(None, 36)
        text = font.render(f"HP: {self.player_hp} | Phase: {self.phase}", True, (255, 255, 255))
        self.window.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.quit()