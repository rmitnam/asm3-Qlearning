"""
Pygame Renderer for GridWorld
Handles visual rendering and optional manual control.
"""

import pygame
import sys
import config
from gridworld import GridWorld


class GridWorldRenderer:
    """
    Pygame-based renderer for the GridWorld environment.
    Can render the environment state and optionally support manual control.
    """
    
    def __init__(self, env, manual_control=False):
        """
        Initialize the renderer.
        
        Args:
            env: GridWorld environment instance
            manual_control: If True, allow keyboard control
        """
        self.env = env
        self.manual_control = manual_control
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        pygame.display.set_caption(f"GridWorld - {env.level_data['name']}")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Key mapping for manual control
        self.key_to_action = {
            pygame.K_UP: config.UP,
            pygame.K_w: config.UP,
            pygame.K_DOWN: config.DOWN,
            pygame.K_s: config.DOWN,
            pygame.K_LEFT: config.LEFT,
            pygame.K_a: config.LEFT,
            pygame.K_RIGHT: config.RIGHT,
            pygame.K_d: config.RIGHT,
        }
        
    def render(self, delay_ms=None):
        """
        Render the current state of the environment.
        
        Args:
            delay_ms: Optional delay in milliseconds after rendering
            
        Returns:
            True if should continue, False if quit requested
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            # Manual control
            if self.manual_control and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    self.env.reset()
                elif event.key in self.key_to_action and not self.env.done:
                    action = self.key_to_action[event.key]
                    state, reward, done, info = self.env.step(action)
                    
                    if done:
                        print(f"Episode finished! Total reward: {self.env.total_reward}")
                        if 'death' in info:
                            print(f"Death by: {info['death']}")
        
        # Clear screen
        self.screen.fill(config.COLOR_BACKGROUND)
        
        # Draw grid
        self._draw_grid()
        
        # Draw entities
        self._draw_entities()
        
        # Draw agent
        self._draw_agent()
        
        # Draw info panel
        self._draw_info_panel()
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        self.clock.tick(config.FPS)
        
        # Optional delay
        if delay_ms:
            pygame.time.delay(delay_ms)
        
        return True
    
    def _draw_grid(self):
        """Draw the grid lines."""
        for x in range(config.GRID_SIZE + 1):
            # Vertical lines
            pygame.draw.line(
                self.screen,
                config.COLOR_GRID_LINE,
                (x * config.CELL_SIZE, 0),
                (x * config.CELL_SIZE, config.GRID_SIZE * config.CELL_SIZE),
                1
            )
        
        for y in range(config.GRID_SIZE + 1):
            # Horizontal lines
            pygame.draw.line(
                self.screen,
                config.COLOR_GRID_LINE,
                (0, y * config.CELL_SIZE),
                (config.GRID_SIZE * config.CELL_SIZE, y * config.CELL_SIZE),
                1
            )
    
    def _draw_entities(self):
        """Draw all static entities (walls, fire, apples, etc.)."""
        for y in range(config.GRID_SIZE):
            for x in range(config.GRID_SIZE):
                entity = self.env.static_grid[y, x]
                pos = (x, y)
                
                # Skip empty cells
                if entity == config.EMPTY:
                    continue
                
                # Skip collected apples
                if entity == config.APPLE and pos in self.env.collected_apples:
                    continue
                
                # Skip collected key
                if entity == config.KEY and self.env.has_key:
                    continue
                
                # Draw opened chest differently
                if entity == config.CHEST and self.env.chest_opened:
                    self._draw_cell(x, y, config.COLOR_CHEST, 'O')
                    continue
                
                # Draw entity
                if entity == config.WALL:
                    self._draw_cell(x, y, config.COLOR_WALL, filled=True)
                elif entity == config.FIRE:
                    self._draw_cell(x, y, config.COLOR_FIRE, 'F')
                elif entity == config.APPLE:
                    self._draw_cell(x, y, config.COLOR_APPLE, 'A')
                elif entity == config.KEY:
                    self._draw_cell(x, y, config.COLOR_KEY, 'K')
                elif entity == config.CHEST:
                    self._draw_cell(x, y, config.COLOR_CHEST, 'C')
        
        # Draw monsters (dynamic entities)
        for monster_pos in self.env.monsters:
            mx, my = monster_pos
            self._draw_cell(mx, my, config.COLOR_MONSTER, 'M')
    
    def _draw_agent(self):
        """Draw the agent."""
        ax, ay = self.env.agent_pos
        self._draw_cell(ax, ay, config.COLOR_AGENT, '@', filled=True)
    
    def _draw_cell(self, x, y, color, symbol=None, filled=False):
        """
        Draw a cell with optional symbol.
        
        Args:
            x, y: Grid coordinates
            color: RGB color tuple
            symbol: Optional character to draw
            filled: If True, fill the cell; otherwise draw circle or symbol
        """
        pixel_x = x * config.CELL_SIZE
        pixel_y = y * config.CELL_SIZE
        
        if filled:
            # Draw filled rectangle
            pygame.draw.rect(
                self.screen,
                color,
                (pixel_x, pixel_y, config.CELL_SIZE, config.CELL_SIZE)
            )
        elif symbol:
            # Draw symbol in center
            text = self.font.render(symbol, True, color)
            text_rect = text.get_rect(
                center=(pixel_x + config.CELL_SIZE // 2, 
                       pixel_y + config.CELL_SIZE // 2)
            )
            self.screen.blit(text, text_rect)
        else:
            # Draw circle
            center = (pixel_x + config.CELL_SIZE // 2, 
                     pixel_y + config.CELL_SIZE // 2)
            radius = config.CELL_SIZE // 3
            pygame.draw.circle(self.screen, color, center, radius)
    
    def _draw_info_panel(self):
        """Draw information panel at the bottom."""
        panel_y = config.GRID_SIZE * config.CELL_SIZE
        
        # Background for info panel
        pygame.draw.rect(
            self.screen,
            (250, 250, 250),
            (0, panel_y, config.WINDOW_WIDTH, 100)
        )
        
        # Level info
        level_text = self.font.render(
            f"Level {self.env.level_id}: {self.env.level_data['name']}", 
            True, 
            config.COLOR_TEXT
        )
        self.screen.blit(level_text, (10, panel_y + 5))
        
        # Stats
        stats_y = panel_y + 30
        stats = [
            f"Steps: {self.env.steps}",
            f"Reward: {self.env.total_reward:.1f}",
            f"Apples: {len(self.env.collected_apples)}/{len(self.env.apple_positions)}",
        ]
        
        if self.env.key_position is not None:
            key_status = "Yes" if self.env.has_key else "No"
            stats.append(f"Has Key: {key_status}")
        
        if self.env.chest_position is not None:
            chest_status = "Yes" if self.env.chest_opened else "No"
            stats.append(f"Chest Opened: {chest_status}")
        
        stats_text = " | ".join(stats)
        stats_render = self.small_font.render(stats_text, True, config.COLOR_TEXT)
        self.screen.blit(stats_render, (10, stats_y))
        
        # Episode status
        status_y = panel_y + 50
        if self.env.done:
            if self.env.total_reward < 0:
                status = "FAILED - Agent Died"
                color = (255, 0, 0)
            else:
                status = "SUCCESS - All Rewards Collected!"
                color = (0, 150, 0)
            status_text = self.font.render(status, True, color)
            self.screen.blit(status_text, (10, status_y))
        
        # Manual control instructions
        if self.manual_control:
            help_y = panel_y + 70
            help_text = "Controls: Arrow Keys/WASD = Move | R = Reset | ESC = Quit"
            help_render = self.small_font.render(help_text, True, (100, 100, 100))
            self.screen.blit(help_render, (10, help_y))
    
    def close(self):
        """Clean up Pygame resources."""
        pygame.quit()


def manual_play(level_id=0):
    """
    Launch manual play mode for testing.
    
    Args:
        level_id: Which level to play
    """
    print(f"Starting manual play mode for Level {level_id}")
    print("Controls:")
    print("  Arrow Keys or WASD - Move")
    print("  R - Reset level")
    print("  ESC - Quit")
    print()
    
    env = GridWorld(level_id=level_id)
    env.reset()
    
    renderer = GridWorldRenderer(env, manual_control=True)
    
    running = True
    while running:
        running = renderer.render()
    
    renderer.close()
    print("Manual play ended.")


if __name__ == "__main__":
    # Test renderer with manual control
    import sys
    
    level = 0
    if len(sys.argv) > 1:
        level = int(sys.argv[1])
    
    manual_play(level)
