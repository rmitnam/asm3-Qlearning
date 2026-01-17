"""
Entity classes for the Arena RL Environment
Contains Player, Enemy, Spawner, and Projectile classes with Pygame rendering
"""

import pygame
import math
import numpy as np
from config import *


class Player:
    """
    Controllable player ship with movement, shooting, and health management.
    Supports both rotation-based and directional movement control schemes.
    """
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = -90  # Facing up initially (in degrees)
        self.health = PLAYER_MAX_HEALTH
        self.max_health = PLAYER_MAX_HEALTH
        self.shoot_cooldown = 0
        self.invincibility = 0
        self.alive = True
        self.size = PLAYER_SIZE
        
    def reset(self, x: float, y: float):
        """Reset player to initial state."""
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = -90
        self.health = PLAYER_MAX_HEALTH
        self.shoot_cooldown = 0
        self.invincibility = 0
        self.alive = True
        
    def update_rotation(self, action: int) -> 'Projectile':
        """
        Update player with rotation-based controls (expanded action space).
        Actions: 
            0=None, 1=Thrust, 2=RotateLeft, 3=RotateRight, 4=Shoot,
            5=Thrust+Shoot, 6=RotateLeft+Shoot, 7=RotateRight+Shoot
        Returns a projectile if shooting, else None.
        """
        projectile = None
        
        # Update cooldowns
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.invincibility > 0:
            self.invincibility -= 1
        
        # Decode combined actions
        do_thrust = action in [1, 5]
        do_rotate_left = action in [2, 6]
        do_rotate_right = action in [3, 7]
        do_shoot = action in [4, 5, 6, 7]
            
        # Process movement
        if do_thrust:
            rad = math.radians(self.angle)
            self.vx += math.cos(rad) * PLAYER_ACCELERATION
            self.vy += math.sin(rad) * PLAYER_ACCELERATION
            
        if do_rotate_left:
            self.angle -= PLAYER_ROTATION_SPEED
        elif do_rotate_right:
            self.angle += PLAYER_ROTATION_SPEED
            
        # Process shooting
        if do_shoot and self.shoot_cooldown <= 0:
            projectile = self._create_projectile()
            self.shoot_cooldown = PLAYER_SHOOT_COOLDOWN
                
        # Apply friction
        self.vx *= PLAYER_FRICTION
        self.vy *= PLAYER_FRICTION
        
        # Clamp velocity
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > PLAYER_MAX_SPEED:
            self.vx = (self.vx / speed) * PLAYER_MAX_SPEED
            self.vy = (self.vy / speed) * PLAYER_MAX_SPEED
            
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Keep in bounds
        self._clamp_to_bounds()
        
        return projectile
    
    def update_directional(self, action: int) -> 'Projectile':
        """
        Update player with directional controls.
        Actions: 0=None, 1=Up, 2=Down, 3=Left, 4=Right, 5=Shoot
        Returns a projectile if shooting, else None.
        """
        projectile = None
        
        # Update cooldowns
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.invincibility > 0:
            self.invincibility -= 1
            
        # Process action
        if action == 1:  # Move up
            self.vy -= PLAYER_ACCELERATION * 3
            self.angle = -90
        elif action == 2:  # Move down
            self.vy += PLAYER_ACCELERATION * 3
            self.angle = 90
        elif action == 3:  # Move left
            self.vx -= PLAYER_ACCELERATION * 3
            self.angle = 180
        elif action == 4:  # Move right
            self.vx += PLAYER_ACCELERATION * 3
            self.angle = 0
        elif action == 5:  # Shoot
            if self.shoot_cooldown <= 0:
                projectile = self._create_projectile()
                self.shoot_cooldown = PLAYER_SHOOT_COOLDOWN
                
        # Apply friction
        self.vx *= PLAYER_FRICTION
        self.vy *= PLAYER_FRICTION
        
        # Clamp velocity
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > PLAYER_MAX_SPEED:
            self.vx = (self.vx / speed) * PLAYER_MAX_SPEED
            self.vy = (self.vy / speed) * PLAYER_MAX_SPEED
            
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Keep in bounds
        self._clamp_to_bounds()
        
        return projectile
    
    def _create_projectile(self) -> 'Projectile':
        """Create a projectile from player's current position and direction."""
        rad = math.radians(self.angle)
        # Spawn projectile slightly in front of player
        spawn_x = self.x + math.cos(rad) * (self.size + 5)
        spawn_y = self.y + math.sin(rad) * (self.size + 5)
        return Projectile(spawn_x, spawn_y, self.angle, is_player_projectile=True)
    
    def _clamp_to_bounds(self):
        """Keep player within screen bounds."""
        margin = self.size
        if self.x < margin:
            self.x = margin
            self.vx = 0
        elif self.x > WINDOW_WIDTH - margin:
            self.x = WINDOW_WIDTH - margin
            self.vx = 0
        if self.y < margin:
            self.y = margin
            self.vy = 0
        elif self.y > WINDOW_HEIGHT - margin:
            self.y = WINDOW_HEIGHT - margin
            self.vy = 0
            
    def take_damage(self, damage: int) -> bool:
        """
        Apply damage to player if not invincible.
        Returns True if damage was applied.
        """
        if self.invincibility > 0:
            return False
        self.health -= damage
        self.invincibility = PLAYER_INVINCIBILITY_FRAMES
        if self.health <= 0:
            self.health = 0
            self.alive = False
        return True
    
    def draw(self, screen: pygame.Surface):
        """Draw player as a triangle pointing in movement direction."""
        # Calculate triangle points
        rad = math.radians(self.angle)
        
        # Front point
        front_x = self.x + math.cos(rad) * self.size
        front_y = self.y + math.sin(rad) * self.size
        
        # Back left point
        back_left_angle = rad + math.radians(140)
        back_left_x = self.x + math.cos(back_left_angle) * self.size * 0.7
        back_left_y = self.y + math.sin(back_left_angle) * self.size * 0.7
        
        # Back right point
        back_right_angle = rad - math.radians(140)
        back_right_x = self.x + math.cos(back_right_angle) * self.size * 0.7
        back_right_y = self.y + math.sin(back_right_angle) * self.size * 0.7
        
        points = [(front_x, front_y), (back_left_x, back_left_y), (back_right_x, back_right_y)]
        
        # Draw with flashing effect when invincible
        color = CYAN if self.invincibility % 6 < 3 or self.invincibility == 0 else WHITE
        pygame.draw.polygon(screen, color, points)
        pygame.draw.polygon(screen, WHITE, points, 2)
        
        # Draw health bar above player
        bar_width = 40
        bar_height = 5
        bar_x = self.x - bar_width // 2
        bar_y = self.y - self.size - 15
        health_ratio = self.health / self.max_health
        
        pygame.draw.rect(screen, RED, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(screen, GREEN, (bar_x, bar_y, int(bar_width * health_ratio), bar_height))
        pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 1)


class Enemy:
    """
    Enemy entity that navigates toward the player.
    Simple AI that moves directly toward player position.
    """
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.health = ENEMY_HEALTH
        self.max_health = ENEMY_HEALTH
        self.size = ENEMY_SIZE
        self.speed = ENEMY_SPEED
        self.alive = True
        self.damage = ENEMY_DAMAGE
        
    def update(self, player_x: float, player_y: float):
        """Move toward player position."""
        if not self.alive:
            return
            
        # Calculate direction to player
        dx = player_x - self.x
        dy = player_y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            # Normalize and apply speed
            self.vx = (dx / distance) * self.speed
            self.vy = (dy / distance) * self.speed
            
        # Update position
        self.x += self.vx
        self.y += self.vy
        
    def take_damage(self, damage: int) -> bool:
        """Apply damage to enemy. Returns True if enemy dies."""
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            self.alive = False
            return True
        return False
    
    def check_collision_with_player(self, player: Player) -> bool:
        """Check if enemy collides with player."""
        dx = player.x - self.x
        dy = player.y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        return distance < (self.size + player.size)
    
    def draw(self, screen: pygame.Surface):
        """Draw enemy as a red diamond shape."""
        if not self.alive:
            return
            
        # Diamond points
        points = [
            (self.x, self.y - self.size),  # Top
            (self.x + self.size, self.y),   # Right
            (self.x, self.y + self.size),   # Bottom
            (self.x - self.size, self.y)    # Left
        ]
        
        pygame.draw.polygon(screen, RED, points)
        pygame.draw.polygon(screen, ORANGE, points, 2)
        
        # Small health indicator
        health_ratio = self.health / self.max_health
        bar_width = 20
        bar_x = self.x - bar_width // 2
        bar_y = self.y - self.size - 8
        pygame.draw.rect(screen, DARK_GRAY, (bar_x, bar_y, bar_width, 3))
        pygame.draw.rect(screen, RED, (bar_x, bar_y, int(bar_width * health_ratio), 3))


class Spawner:
    """
    Enemy spawner that periodically creates enemies.
    Destroying all spawners progresses to the next phase.
    """
    
    def __init__(self, x: float, y: float, spawn_rate: int = SPAWNER_SPAWN_RATE):
        self.x = x
        self.y = y
        self.health = SPAWNER_HEALTH
        self.max_health = SPAWNER_HEALTH
        self.size = SPAWNER_SIZE
        self.spawn_rate = spawn_rate
        self.spawn_timer = spawn_rate // 2  # Start with half timer
        self.alive = True
        self.pulse_timer = 0
        
    def update(self) -> bool:
        """
        Update spawner timer.
        Returns True if ready to spawn an enemy.
        """
        if not self.alive:
            return False
            
        self.pulse_timer += 1
        self.spawn_timer -= 1
        
        if self.spawn_timer <= 0:
            self.spawn_timer = self.spawn_rate
            return True
        return False
    
    def spawn_enemy(self) -> Enemy:
        """Create a new enemy at spawner's position with slight offset."""
        offset_angle = np.random.uniform(0, 2 * math.pi)
        offset_dist = self.size + ENEMY_SIZE + 5
        spawn_x = self.x + math.cos(offset_angle) * offset_dist
        spawn_y = self.y + math.sin(offset_angle) * offset_dist
        
        # Clamp to screen bounds
        spawn_x = max(ENEMY_SIZE, min(WINDOW_WIDTH - ENEMY_SIZE, spawn_x))
        spawn_y = max(ENEMY_SIZE, min(WINDOW_HEIGHT - ENEMY_SIZE, spawn_y))
        
        return Enemy(spawn_x, spawn_y)
    
    def take_damage(self, damage: int) -> bool:
        """Apply damage to spawner. Returns True if spawner is destroyed."""
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            self.alive = False
            return True
        return False
    
    def draw(self, screen: pygame.Surface):
        """Draw spawner as a pulsing hexagon."""
        if not self.alive:
            return
            
        # Pulsing effect
        pulse = math.sin(self.pulse_timer * 0.1) * 3
        size = self.size + pulse
        
        # Hexagon points
        points = []
        for i in range(6):
            angle = math.radians(60 * i - 30)
            px = self.x + math.cos(angle) * size
            py = self.y + math.sin(angle) * size
            points.append((px, py))
            
        pygame.draw.polygon(screen, PURPLE, points)
        pygame.draw.polygon(screen, YELLOW, points, 3)
        
        # Inner decoration
        inner_points = []
        for i in range(6):
            angle = math.radians(60 * i - 30)
            px = self.x + math.cos(angle) * (size * 0.5)
            py = self.y + math.sin(angle) * (size * 0.5)
            inner_points.append((px, py))
        pygame.draw.polygon(screen, YELLOW, inner_points, 1)
        
        # Health bar
        bar_width = 50
        bar_height = 6
        bar_x = self.x - bar_width // 2
        bar_y = self.y - self.size - 15
        health_ratio = self.health / self.max_health
        
        pygame.draw.rect(screen, DARK_GRAY, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(screen, PURPLE, (bar_x, bar_y, int(bar_width * health_ratio), bar_height))
        pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 1)
        
        # Spawn timer indicator
        timer_ratio = 1.0 - (self.spawn_timer / self.spawn_rate)
        pygame.draw.arc(screen, YELLOW, 
                       (self.x - size - 5, self.y - size - 5, (size + 5) * 2, (size + 5) * 2),
                       -math.pi/2, -math.pi/2 + timer_ratio * 2 * math.pi, 2)


class Projectile:
    """
    Projectile fired by player.
    Travels in a straight line and damages enemies/spawners on collision.
    """
    
    def __init__(self, x: float, y: float, angle: float, is_player_projectile: bool = True):
        self.x = x
        self.y = y
        self.angle = angle
        self.is_player_projectile = is_player_projectile
        self.size = PROJECTILE_SIZE
        self.speed = PROJECTILE_SPEED
        self.damage = PROJECTILE_DAMAGE
        self.lifetime = PROJECTILE_LIFETIME
        self.alive = True
        
        # Calculate velocity
        rad = math.radians(angle)
        self.vx = math.cos(rad) * self.speed
        self.vy = math.sin(rad) * self.speed
        
    def update(self):
        """Update projectile position and lifetime."""
        if not self.alive:
            return
            
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        
        # Check bounds and lifetime
        if (self.x < 0 or self.x > WINDOW_WIDTH or 
            self.y < 0 or self.y > WINDOW_HEIGHT or
            self.lifetime <= 0):
            self.alive = False
            
    def check_collision(self, entity) -> bool:
        """Check collision with any entity (Enemy or Spawner)."""
        dx = entity.x - self.x
        dy = entity.y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        return distance < (self.size + entity.size)
    
    def draw(self, screen: pygame.Surface):
        """Draw projectile as a glowing circle."""
        if not self.alive:
            return
            
        # Outer glow
        pygame.draw.circle(screen, YELLOW, (int(self.x), int(self.y)), self.size + 2)
        # Core
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.size)


class ParticleEffect:
    """Simple particle effect for explosions and hits."""
    
    def __init__(self, x: float, y: float, color: tuple, count: int = 10):
        self.particles = []
        for _ in range(count):
            angle = np.random.uniform(0, 2 * math.pi)
            speed = np.random.uniform(2, 6)
            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': np.random.randint(10, 25),
                'color': color,
                'size': np.random.randint(2, 5)
            })
            
    def update(self) -> bool:
        """Update particles. Returns True if effect is still active."""
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vx'] *= 0.95
            p['vy'] *= 0.95
            p['life'] -= 1
            
        self.particles = [p for p in self.particles if p['life'] > 0]
        return len(self.particles) > 0
    
    def draw(self, screen: pygame.Surface):
        """Draw all particles."""
        for p in self.particles:
            alpha = p['life'] / 25
            size = max(1, int(p['size'] * alpha))
            pygame.draw.circle(screen, p['color'], (int(p['x']), int(p['y'])), size)
