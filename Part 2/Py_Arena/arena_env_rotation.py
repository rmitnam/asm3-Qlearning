"""
Arena Environment with Rotation-Based Controls
Gym-style environment for RL training with thrust and rotation movement.

Actions (expanded for better control):
0 - No action
1 - Thrust forward
2 - Rotate left
3 - Rotate right
4 - Shoot
5 - Thrust + Shoot (combined)
6 - Rotate left + Shoot (combined)
7 - Rotate right + Shoot (combined)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from typing import Optional, Tuple, Dict, Any, List

from config import *
from entities import Player, Enemy, Spawner, Projectile, ParticleEffect


class ArenaEnvRotation(gym.Env):
    """
    Arena environment with rotation-based controls.
    
    The player controls a ship that can thrust forward, rotate left/right, and shoot.
    Combined actions allow shooting while moving/rotating for better gameplay.
    The goal is to destroy enemy spawners while surviving waves of enemies.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None
        
        # Action space: 8 discrete actions (including combined actions)
        self.action_space = spaces.Discrete(8)
        
        # Observation space: continuous vector
        # Structure:
        # [0-1]: Player position (x, y) normalized
        # [2-3]: Player velocity (vx, vy) normalized
        # [4-5]: Player orientation (cos, sin)
        # [6]: Player health normalized
        # [7]: Current phase normalized
        # [8-11]: Nearest enemy (distance, angle, rel_x, rel_y)
        # [12-15]: Second nearest enemy
        # [16-19]: Nearest spawner (distance, angle, rel_x, rel_y)
        # [20-23]: Second nearest spawner
        # [24]: Number of enemies normalized
        # [25]: Number of spawners normalized
        # [26]: Can shoot (1 if cooldown = 0)
        # [27]: Time remaining normalized
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(OBSERVATION_SIZE_ROTATION,),
            dtype=np.float32
        )
        
        # Game state
        self.player: Optional[Player] = None
        self.enemies: List[Enemy] = []
        self.spawners: List[Spawner] = []
        self.projectiles: List[Projectile] = []
        self.effects: List[ParticleEffect] = []
        
        self.current_phase = 1
        self.steps = 0
        self.total_reward = 0
        
        # Statistics tracking
        self.enemies_killed = 0
        self.spawners_killed = 0
        self.damage_taken = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize player at center
        self.player = Player(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2)
        
        # Clear all entities
        self.enemies = []
        self.projectiles = []
        self.effects = []
        
        # Initialize spawners for phase 1
        self.current_phase = 1
        self._spawn_phase_spawners()
        
        # Reset counters
        self.steps = 0
        self.total_reward = 0
        self.enemies_killed = 0
        self.spawners_killed = 0
        self.damage_taken = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        reward = 0.0
        terminated = False
        truncated = False
        
        self.steps += 1
        
        # Update player and get projectile if shooting
        projectile = self.player.update_rotation(action)
        if projectile:
            self.projectiles.append(projectile)
            
        # Update enemies
        for enemy in self.enemies:
            if enemy.alive:
                enemy.update(self.player.x, self.player.y)
                
                # Check collision with player
                if enemy.check_collision_with_player(self.player):
                    if self.player.take_damage(enemy.damage):
                        reward += REWARD_DAMAGE_TAKEN
                        self.damage_taken += enemy.damage
                        self.effects.append(ParticleEffect(self.player.x, self.player.y, RED, 15))
                    # Kill enemy on collision
                    enemy.alive = False
                    self.effects.append(ParticleEffect(enemy.x, enemy.y, ORANGE, 8))
                    
        # Update spawners and spawn enemies
        for spawner in self.spawners:
            if spawner.alive and spawner.update():
                new_enemy = spawner.spawn_enemy()
                self.enemies.append(new_enemy)
                
        # Update projectiles and check collisions
        for proj in self.projectiles:
            if proj.alive:
                proj.update()
                
                if proj.is_player_projectile:
                    # Check enemy collisions
                    for enemy in self.enemies:
                        if enemy.alive and proj.check_collision(enemy):
                            if enemy.take_damage(proj.damage):
                                reward += REWARD_KILL_ENEMY
                                self.enemies_killed += 1
                                self.effects.append(ParticleEffect(enemy.x, enemy.y, ORANGE, 12))
                            else:
                                reward += REWARD_SHOOT_HIT
                                self.effects.append(ParticleEffect(proj.x, proj.y, YELLOW, 5))
                            proj.alive = False
                            break
                            
                    # Check spawner collisions
                    if proj.alive:
                        for spawner in self.spawners:
                            if spawner.alive and proj.check_collision(spawner):
                                if spawner.take_damage(proj.damage):
                                    reward += REWARD_KILL_SPAWNER
                                    self.spawners_killed += 1
                                    self.effects.append(ParticleEffect(spawner.x, spawner.y, PURPLE, 20))
                                else:
                                    reward += REWARD_SHOOT_HIT
                                    self.effects.append(ParticleEffect(proj.x, proj.y, YELLOW, 5))
                                proj.alive = False
                                break
                                
        # Clean up dead entities
        self.enemies = [e for e in self.enemies if e.alive]
        self.projectiles = [p for p in self.projectiles if p.alive]
        
        # Update effects
        self.effects = [e for e in self.effects if e.update()]
        
        # Check phase progression
        alive_spawners = [s for s in self.spawners if s.alive]
        if len(alive_spawners) == 0:
            if self.current_phase < MAX_PHASE:
                self.current_phase += 1
                reward += REWARD_PHASE_PROGRESS
                self._spawn_phase_spawners()
            else:
                # Won the game!
                reward += REWARD_PHASE_PROGRESS * 2
                terminated = True
                
        # Survival bonus
        reward += REWARD_SURVIVAL_BONUS
        
        # Small step penalty to encourage efficiency
        reward += STEP_PENALTY
        
        # Shaping reward: encourage approaching spawners
        reward += self._get_shaping_reward()
        
        # Check death
        if not self.player.alive:
            reward += REWARD_DEATH
            terminated = True
            
        # Check max steps
        if self.steps >= MAX_EPISODE_STEPS:
            truncated = True
            
        self.total_reward += reward
        
        observation = self._get_observation()
        info = self._get_info()
        
        # Render if in human mode
        if self.render_mode == "human":
            self.render()
            
        return observation, reward, terminated, truncated, info
    
    def _spawn_phase_spawners(self):
        """Spawn spawners for the current phase."""
        self.spawners = []
        num_spawners = INITIAL_SPAWNERS + (self.current_phase - 1) * SPAWNERS_PER_PHASE
        spawn_rate = max(30, SPAWNER_SPAWN_RATE - (self.current_phase - 1) * PHASE_SPAWN_RATE_REDUCTION)
        
        # Calculate spawn positions (away from player)
        margin = 100
        for i in range(num_spawners):
            # Try to find a good spawn position
            for _ in range(20):
                x = self.np_random.uniform(margin, WINDOW_WIDTH - margin)
                y = self.np_random.uniform(margin, WINDOW_HEIGHT - margin)
                
                # Check distance from player
                dist_to_player = math.sqrt((x - self.player.x)**2 + (y - self.player.y)**2)
                if dist_to_player > 200:
                    # Check distance from other spawners
                    too_close = False
                    for s in self.spawners:
                        if math.sqrt((x - s.x)**2 + (y - s.y)**2) < 150:
                            too_close = True
                            break
                    if not too_close:
                        break
                        
            self.spawners.append(Spawner(x, y, spawn_rate))
            
    def _get_observation(self) -> np.ndarray:
        """Build the observation vector."""
        obs = np.zeros(OBSERVATION_SIZE_ROTATION, dtype=np.float32)
        
        # Player position normalized to [-1, 1]
        obs[0] = (self.player.x / WINDOW_WIDTH) * 2 - 1
        obs[1] = (self.player.y / WINDOW_HEIGHT) * 2 - 1
        
        # Player velocity normalized
        max_vel = PLAYER_MAX_SPEED
        obs[2] = np.clip(self.player.vx / max_vel, -1, 1)
        obs[3] = np.clip(self.player.vy / max_vel, -1, 1)
        
        # Player orientation as unit vector
        rad = math.radians(self.player.angle)
        obs[4] = math.cos(rad)
        obs[5] = math.sin(rad)
        
        # Player health normalized
        obs[6] = self.player.health / PLAYER_MAX_HEALTH
        
        # Current phase normalized
        obs[7] = self.current_phase / MAX_PHASE
        
        # Get nearest enemies
        enemy_data = self._get_nearest_entities(self.enemies, 2)
        for i, (dist, angle, rel_x, rel_y) in enumerate(enemy_data):
            idx = 8 + i * 4
            obs[idx] = dist
            obs[idx + 1] = angle
            obs[idx + 2] = rel_x
            obs[idx + 3] = rel_y
            
        # Get nearest spawners
        alive_spawners = [s for s in self.spawners if s.alive]
        spawner_data = self._get_nearest_entities(alive_spawners, 2)
        for i, (dist, angle, rel_x, rel_y) in enumerate(spawner_data):
            idx = 16 + i * 4
            obs[idx] = dist
            obs[idx + 1] = angle
            obs[idx + 2] = rel_x
            obs[idx + 3] = rel_y
            
        # Entity counts normalized
        obs[24] = min(len(self.enemies) / 20, 1.0)
        obs[25] = min(len(alive_spawners) / 10, 1.0)
        
        # Can shoot
        obs[26] = 1.0 if self.player.shoot_cooldown <= 0 else 0.0
        
        # Time remaining normalized
        obs[27] = 1.0 - (self.steps / MAX_EPISODE_STEPS)
        
        return obs
    
    def _get_nearest_entities(self, entities: List, count: int) -> List[Tuple[float, float, float, float]]:
        """Get normalized distance and direction to nearest entities."""
        result = []
        
        # Calculate distances
        distances = []
        for entity in entities:
            if hasattr(entity, 'alive') and not entity.alive:
                continue
            dx = entity.x - self.player.x
            dy = entity.y - self.player.y
            dist = math.sqrt(dx**2 + dy**2)
            distances.append((dist, entity))
            
        # Sort by distance
        distances.sort(key=lambda x: x[0])
        
        # Get data for nearest entities
        max_dist = math.sqrt(WINDOW_WIDTH**2 + WINDOW_HEIGHT**2)
        
        for i in range(count):
            if i < len(distances):
                dist, entity = distances[i]
                dx = entity.x - self.player.x
                dy = entity.y - self.player.y
                
                # Normalize distance to [0, 1]
                norm_dist = dist / max_dist
                
                # Angle relative to player facing direction
                angle_to_entity = math.atan2(dy, dx)
                player_angle = math.radians(self.player.angle)
                relative_angle = angle_to_entity - player_angle
                
                # Normalize to [-1, 1]
                norm_angle = math.sin(relative_angle)
                
                # Relative position normalized
                rel_x = dx / max_dist
                rel_y = dy / max_dist
                
                result.append((norm_dist, norm_angle, rel_x, rel_y))
            else:
                # No entity, use placeholder
                result.append((1.0, 0.0, 0.0, 0.0))
                
        return result
    
    def _get_shaping_reward(self) -> float:
        """
        Calculate shaping reward to guide learning for rotation controls.
        
        Justification for rotation-specific shaping:
        - Rotation controls are harder because agent must coordinate facing direction with movement
        - Additional shaping rewards help agent learn:
          1. Face toward targets (aiming reward)
          2. Approach spawners (distance reward)
          3. Avoid getting surrounded (enemy avoidance)
        - All shaping rewards are small enough not to overshadow main task rewards
        """
        reward = 0.0
        max_dist = math.sqrt(WINDOW_WIDTH**2 + WINDOW_HEIGHT**2)
        
        # Get player facing direction
        player_rad = math.radians(self.player.angle)
        facing_x = math.cos(player_rad)
        facing_y = math.sin(player_rad)
        
        # 1. Reward for facing toward nearest spawner (helps learn aiming)
        alive_spawners = [s for s in self.spawners if s.alive]
        if alive_spawners:
            # Find nearest spawner
            min_dist = float('inf')
            nearest_spawner = None
            for spawner in alive_spawners:
                dx = spawner.x - self.player.x
                dy = spawner.y - self.player.y
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_spawner = spawner
                    
            if nearest_spawner:
                dx = nearest_spawner.x - self.player.x
                dy = nearest_spawner.y - self.player.y
                dist = math.sqrt(dx**2 + dy**2)
                
                if dist > 0:
                    # Direction to spawner (normalized)
                    dir_x = dx / dist
                    dir_y = dy / dist
                    
                    # Dot product: 1 = facing directly at target, -1 = facing away
                    facing_alignment = facing_x * dir_x + facing_y * dir_y
                    
                    # Reward for facing spawner (max ~0.03 when perfectly aligned)
                    reward += max(0, facing_alignment) * 0.03
                    
                # Distance reward (approach spawner) - max ~0.02
                reward += (1.0 - min_dist / max_dist) * 0.02
                
        # 2. Reward for facing toward nearest enemy when close (combat awareness)
        if self.enemies:
            nearest_enemy = None
            min_enemy_dist = float('inf')
            for enemy in self.enemies:
                if enemy.alive:
                    dx = enemy.x - self.player.x
                    dy = enemy.y - self.player.y
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist < min_enemy_dist:
                        min_enemy_dist = dist
                        nearest_enemy = enemy
                        
            if nearest_enemy and min_enemy_dist < 200:  # Only when enemy is close
                dx = nearest_enemy.x - self.player.x
                dy = nearest_enemy.y - self.player.y
                dist = math.sqrt(dx**2 + dy**2)
                
                if dist > 0:
                    dir_x = dx / dist
                    dir_y = dy / dist
                    facing_alignment = facing_x * dir_x + facing_y * dir_y
                    
                    # Small reward for facing nearby enemies (defensive awareness)
                    reward += max(0, facing_alignment) * 0.01
                    
        # 3. Small penalty for being surrounded by many enemies (encourages clearing)
        nearby_enemies = sum(1 for e in self.enemies if e.alive and 
                           math.sqrt((e.x - self.player.x)**2 + (e.y - self.player.y)**2) < 150)
        if nearby_enemies > 3:
            reward -= 0.01 * (nearby_enemies - 3)
            
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the environment state."""
        return {
            'phase': self.current_phase,
            'enemies_alive': len(self.enemies),
            'spawners_alive': len([s for s in self.spawners if s.alive]),
            'player_health': self.player.health,
            'enemies_killed': self.enemies_killed,
            'spawners_killed': self.spawners_killed,
            'damage_taken': self.damage_taken,
            'total_reward': self.total_reward,
            'steps': self.steps
        }
        
    def render(self):
        """Render the game state."""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Arena RL - Rotation Controls")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
                
        # Clear screen with dark background
        self.window.fill(BLACK)
        
        # Draw grid for visual reference
        self._draw_grid()
        
        # Draw all entities
        for effect in self.effects:
            effect.draw(self.window)
            
        for proj in self.projectiles:
            proj.draw(self.window)
            
        for enemy in self.enemies:
            enemy.draw(self.window)
            
        for spawner in self.spawners:
            spawner.draw(self.window)
            
        if self.player.alive:
            self.player.draw(self.window)
            
        # Draw HUD
        self._draw_hud()
        
        pygame.display.flip()
        self.clock.tick(FPS)
        
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.window)
            
    def _draw_grid(self):
        """Draw background grid."""
        grid_size = 50
        for x in range(0, WINDOW_WIDTH, grid_size):
            pygame.draw.line(self.window, DARK_GRAY, (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, grid_size):
            pygame.draw.line(self.window, DARK_GRAY, (0, y), (WINDOW_WIDTH, y))
            
    def _draw_hud(self):
        """Draw heads-up display with game info."""
        # Phase
        phase_text = self.font.render(f"Phase: {self.current_phase}/{MAX_PHASE}", True, WHITE)
        self.window.blit(phase_text, (10, 10))
        
        # Health
        health_text = self.font.render(f"Health: {self.player.health}/{PLAYER_MAX_HEALTH}", True, GREEN)
        self.window.blit(health_text, (10, 35))
        
        # Enemies
        enemies_text = self.font.render(f"Enemies: {len(self.enemies)}", True, RED)
        self.window.blit(enemies_text, (10, 60))
        
        # Spawners
        alive_spawners = len([s for s in self.spawners if s.alive])
        spawners_text = self.font.render(f"Spawners: {alive_spawners}", True, PURPLE)
        self.window.blit(spawners_text, (10, 85))
        
        # Score/Kills
        kills_text = self.font.render(f"Kills: {self.enemies_killed} | Spawners: {self.spawners_killed}", True, YELLOW)
        self.window.blit(kills_text, (WINDOW_WIDTH - 250, 10))
        
        # Steps/Time
        steps_text = self.font.render(f"Steps: {self.steps}/{MAX_EPISODE_STEPS}", True, WHITE)
        self.window.blit(steps_text, (WINDOW_WIDTH - 200, 35))
        
        # Reward
        reward_text = self.font.render(f"Reward: {self.total_reward:.1f}", True, CYAN)
        self.window.blit(reward_text, (WINDOW_WIDTH - 150, 60))
        
        # Controls hint
        controls_text = self.font.render("Rotation Mode: Thrust/Rotate L/R/Shoot", True, LIGHT_GRAY)
        self.window.blit(controls_text, (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT - 25))
        
    def close(self):
        """Clean up pygame resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


# Register the environment
if __name__ == "__main__":
    # Test the environment
    env = ArenaEnvRotation(render_mode="human")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Action space: {env.action_space}")
    
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
    print(f"Episode finished. Total reward: {total_reward:.2f}")
    print(f"Info: {info}")
    env.close()
