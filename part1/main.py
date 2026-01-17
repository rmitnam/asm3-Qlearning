"""
Main entry point with menu UI for the Gridworld RL project.
Provides a unified interface for manual play, training, and agent demos.
"""

import pygame
import sys
from pathlib import Path

import config
from gridworld import GridWorld
from q_learning import QLearningAgent
from sarsa import SARSAAgent
from renderer import GridWorldRenderer
# train_agent import removed - training now done inline with progress bar


# UI Constants
MENU_WIDTH = 600
MENU_HEIGHT = 700
BUTTON_WIDTH = 250
BUTTON_HEIGHT = 50
BUTTON_SPACING = 15

# Colors
COLOR_BG = (30, 30, 40)
COLOR_TITLE = (100, 200, 255)
COLOR_BUTTON = (60, 60, 80)
COLOR_BUTTON_HOVER = (80, 80, 110)
COLOR_BUTTON_SELECTED = (100, 150, 200)
COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_DIM = (150, 150, 150)
COLOR_SUCCESS = (100, 255, 100)
COLOR_FAIL = (255, 100, 100)


class Button:
    def __init__(self, x, y, width, height, text, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.hovered = False
        self.selected = False

    def draw(self, screen, font):
        # Check hover state based on current mouse position
        mouse_pos = pygame.mouse.get_pos()
        self.hovered = self.rect.collidepoint(mouse_pos)

        if self.selected:
            color = COLOR_BUTTON_SELECTED
        elif self.hovered:
            color = COLOR_BUTTON_HOVER
        else:
            color = COLOR_BUTTON

        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, COLOR_TEXT_DIM, self.rect, 2, border_radius=8)

        text_surf = font.render(self.text, True, COLOR_TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos) and self.action:
                return self.action
        return None


class MenuSystem:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))
        pygame.display.set_caption("Gridworld RL - Main Menu")

        self.title_font = pygame.font.Font(None, 56)
        self.subtitle_font = pygame.font.Font(None, 36)
        self.button_font = pygame.font.Font(None, 32)
        self.info_font = pygame.font.Font(None, 24)

        self.clock = pygame.time.Clock()
        self.running = True

        # State
        self.current_menu = "main"
        self.selected_level = 0
        self.selected_algorithm = "qlearning"
        self.use_intrinsic = False

        # Trained agents cache
        self.trained_agents = {}

    def create_main_menu_buttons(self):
        start_y = 200
        center_x = MENU_WIDTH // 2 - BUTTON_WIDTH // 2

        buttons = [
            Button(center_x, start_y, BUTTON_WIDTH, BUTTON_HEIGHT,
                   "Manual Play", "manual_play"),
            Button(center_x, start_y + BUTTON_HEIGHT + BUTTON_SPACING, BUTTON_WIDTH, BUTTON_HEIGHT,
                   "Watch Trained Agent", "watch_agent"),
            Button(center_x, start_y + 2 * (BUTTON_HEIGHT + BUTTON_SPACING), BUTTON_WIDTH, BUTTON_HEIGHT,
                   "Settings", "settings"),
            Button(center_x, start_y + 3 * (BUTTON_HEIGHT + BUTTON_SPACING), BUTTON_WIDTH, BUTTON_HEIGHT,
                   "Quit", "quit"),
        ]
        return buttons

    def create_level_select_buttons(self):
        buttons = []
        start_y = 180
        center_x = MENU_WIDTH // 2 - BUTTON_WIDTH // 2

        level_names = [
            "Level 0: Basic",
            "Level 1: Fire Hazards",
            "Level 2: Key & Chest",
            "Level 3: Complex Maze",
            "Level 4: Monsters",
            "Level 5: Monsters + Key",
            "Level 6: Exploration",
        ]

        for i, name in enumerate(level_names):
            btn = Button(center_x, start_y + i * (BUTTON_HEIGHT + 10),
                        BUTTON_WIDTH, BUTTON_HEIGHT, name, f"level_{i}")
            btn.selected = (i == self.selected_level)
            buttons.append(btn)

        buttons.append(Button(center_x, start_y + 7 * (BUTTON_HEIGHT + 10) + 20,
                             BUTTON_WIDTH, BUTTON_HEIGHT, "Back", "back"))

        return buttons

    def create_settings_buttons(self):
        buttons = []
        start_y = 200
        center_x = MENU_WIDTH // 2 - BUTTON_WIDTH // 2

        algo_text = f"Algorithm: {self.selected_algorithm.upper()}"
        buttons.append(Button(center_x, start_y, BUTTON_WIDTH, BUTTON_HEIGHT,
                             algo_text, "toggle_algorithm"))

        intrinsic_text = f"Intrinsic Rewards: {'ON' if self.use_intrinsic else 'OFF'}"
        buttons.append(Button(center_x, start_y + BUTTON_HEIGHT + BUTTON_SPACING,
                             BUTTON_WIDTH, BUTTON_HEIGHT, intrinsic_text, "toggle_intrinsic"))

        buttons.append(Button(center_x, start_y + 3 * (BUTTON_HEIGHT + BUTTON_SPACING),
                             BUTTON_WIDTH, BUTTON_HEIGHT, "Back", "back"))

        return buttons

    def draw_main_menu(self, buttons):
        self.screen.fill(COLOR_BG)

        # Title
        title = self.title_font.render("Gridworld RL", True, COLOR_TITLE)
        title_rect = title.get_rect(center=(MENU_WIDTH // 2, 80))
        self.screen.blit(title, title_rect)

        subtitle = self.subtitle_font.render("Q-Learning & SARSA Demo", True, COLOR_TEXT_DIM)
        subtitle_rect = subtitle.get_rect(center=(MENU_WIDTH // 2, 130))
        self.screen.blit(subtitle, subtitle_rect)

        for btn in buttons:
            btn.draw(self.screen, self.button_font)

        # Footer
        info = self.info_font.render("Use mouse to navigate | ESC to go back", True, COLOR_TEXT_DIM)
        self.screen.blit(info, (MENU_WIDTH // 2 - info.get_width() // 2, MENU_HEIGHT - 40))

    def draw_level_select(self, buttons, title_text):
        self.screen.fill(COLOR_BG)

        title = self.subtitle_font.render(title_text, True, COLOR_TITLE)
        title_rect = title.get_rect(center=(MENU_WIDTH // 2, 80))
        self.screen.blit(title, title_rect)

        # Current settings
        settings = self.info_font.render(
            f"Algorithm: {self.selected_algorithm.upper()} | Intrinsic: {'ON' if self.use_intrinsic else 'OFF'}",
            True, COLOR_TEXT_DIM)
        self.screen.blit(settings, (MENU_WIDTH // 2 - settings.get_width() // 2, 120))

        for btn in buttons:
            btn.draw(self.screen, self.button_font)

    def draw_settings(self, buttons):
        self.screen.fill(COLOR_BG)

        title = self.subtitle_font.render("Settings", True, COLOR_TITLE)
        title_rect = title.get_rect(center=(MENU_WIDTH // 2, 80))
        self.screen.blit(title, title_rect)

        for btn in buttons:
            btn.draw(self.screen, self.button_font)

        # Info text
        info_lines = [
            "Q-Learning: Off-policy, uses max Q-value",
            "SARSA: On-policy, uses actual next action",
            "",
            "Intrinsic Rewards: Exploration bonus for Level 6",
        ]
        for i, line in enumerate(info_lines):
            text = self.info_font.render(line, True, COLOR_TEXT_DIM)
            self.screen.blit(text, (50, 450 + i * 25))

    def run_manual_play(self, level_id):
        """Run manual play mode for a level."""
        pygame.display.set_caption(f"Manual Play - Level {level_id}")

        env = GridWorld(level_id=level_id)
        env.reset()  # Initialize the environment
        renderer = GridWorldRenderer(env, manual_control=True)

        running = True
        while running:
            # renderer.render() handles all events including input and ESC to quit
            result = renderer.render(delay_ms=100)
            if result == False:
                running = False

        # Don't call renderer.close() - it quits pygame entirely
        pygame.display.set_caption("Gridworld RL - Main Menu")

    def draw_training_progress(self, episode, max_episodes, algorithm, level_id, epsilon, recent_reward):
        """Draw a training progress screen with progress bar."""
        self.screen.fill(COLOR_BG)

        # Title
        title = self.subtitle_font.render(f"Training {algorithm.upper()}", True, COLOR_TITLE)
        title_rect = title.get_rect(center=(MENU_WIDTH // 2, 100))
        self.screen.blit(title, title_rect)

        # Level info
        level_text = self.info_font.render(f"Level {level_id}", True, COLOR_TEXT_DIM)
        self.screen.blit(level_text, (MENU_WIDTH // 2 - level_text.get_width() // 2, 140))

        # Progress bar
        bar_width = 400
        bar_height = 30
        bar_x = (MENU_WIDTH - bar_width) // 2
        bar_y = 200

        # Background bar
        pygame.draw.rect(self.screen, (60, 60, 80), (bar_x, bar_y, bar_width, bar_height), border_radius=5)

        # Progress fill
        progress = episode / max_episodes
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            pygame.draw.rect(self.screen, COLOR_BUTTON_SELECTED, (bar_x, bar_y, fill_width, bar_height), border_radius=5)

        # Border
        pygame.draw.rect(self.screen, COLOR_TEXT_DIM, (bar_x, bar_y, bar_width, bar_height), 2, border_radius=5)

        # Progress text
        progress_text = self.button_font.render(f"{episode}/{max_episodes} episodes ({progress*100:.0f}%)", True, COLOR_TEXT)
        self.screen.blit(progress_text, (MENU_WIDTH // 2 - progress_text.get_width() // 2, bar_y + 45))

        # Stats
        stats_y = 300
        epsilon_text = self.info_font.render(f"Epsilon: {epsilon:.3f}", True, COLOR_TEXT_DIM)
        self.screen.blit(epsilon_text, (MENU_WIDTH // 2 - epsilon_text.get_width() // 2, stats_y))

        reward_text = self.info_font.render(f"Recent Reward: {recent_reward:.1f}", True, COLOR_TEXT_DIM)
        self.screen.blit(reward_text, (MENU_WIDTH // 2 - reward_text.get_width() // 2, stats_y + 30))

        # Hint
        hint = self.info_font.render("Training is fast - please wait...", True, COLOR_TEXT_DIM)
        self.screen.blit(hint, (MENU_WIDTH // 2 - hint.get_width() // 2, 400))

        pygame.display.flip()

    def train_agent_with_progress(self, level_id):
        """Train agent with progress bar display."""
        config.USE_INTRINSIC_REWARD = self.use_intrinsic

        env = GridWorld(level_id=level_id)
        if self.selected_algorithm == "qlearning":
            agent = QLearningAgent()
        else:
            agent = SARSAAgent()

        max_episodes = 5000
        recent_reward = 0.0

        for episode in range(max_episodes):
            # Process events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None

            state = env.reset()
            done = False
            episode_reward = 0
            action = agent.select_action(state, explore=True)

            while not done:
                next_state, reward, done, info = env.step(action)

                update_reward = reward
                if config.USE_INTRINSIC_REWARD:
                    update_reward += env.get_intrinsic_reward(next_state)

                next_action = agent.select_action(next_state, explore=True) if not done else None

                if hasattr(agent, 'update') and 'next_action' in agent.update.__code__.co_varnames:
                    agent.update(state, action, update_reward, next_state, next_action, done)
                else:
                    agent.update(state, action, update_reward, next_state, done)

                state = next_state
                action = next_action
                episode_reward += reward

            agent.decay_epsilon()
            recent_reward = episode_reward

            # Update progress display every 50 episodes
            if episode % 50 == 0 or episode == max_episodes - 1:
                self.draw_training_progress(episode + 1, max_episodes, self.selected_algorithm,
                                           level_id, agent.epsilon, recent_reward)

        return agent

    def run_trained_agent(self, level_id):
        """Run a trained agent on a level."""
        pygame.display.set_caption(f"Trained Agent - Level {level_id}")

        config.USE_INTRINSIC_REWARD = self.use_intrinsic

        # Check if we have a cached trained agent
        cache_key = (level_id, self.selected_algorithm, self.use_intrinsic)

        if cache_key not in self.trained_agents:
            # Train with progress bar
            agent = self.train_agent_with_progress(level_id)
            if agent is None:
                # User cancelled
                pygame.display.set_caption("Gridworld RL - Main Menu")
                return
            self.trained_agents[cache_key] = agent
        else:
            agent = self.trained_agents[cache_key]

        # Run single demo episode
        env = GridWorld(level_id=level_id)
        env.reset()  # Initialize the environment
        renderer = GridWorldRenderer(env, manual_control=False)

        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        running = True

        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if not running:
                break

            action = agent.select_action(state, explore=False)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

            if renderer.render(delay_ms=120) == False:
                running = False

        if running and done:
            # Show result briefly
            result_color = COLOR_SUCCESS if total_reward > 0 else COLOR_FAIL
            result_text = f"Completed: {steps} steps, Reward: {total_reward:.1f}"
            self.show_overlay_message(renderer, result_text, result_color, 2000)

        # Don't call renderer.close() - it quits pygame entirely
        pygame.display.set_caption("Gridworld RL - Main Menu")

    def show_message(self, title, subtitle=""):
        """Show a message screen."""
        self.screen.fill(COLOR_BG)

        title_surf = self.subtitle_font.render(title, True, COLOR_TITLE)
        title_rect = title_surf.get_rect(center=(MENU_WIDTH // 2, MENU_HEIGHT // 2 - 20))
        self.screen.blit(title_surf, title_rect)

        if subtitle:
            sub_surf = self.info_font.render(subtitle, True, COLOR_TEXT_DIM)
            sub_rect = sub_surf.get_rect(center=(MENU_WIDTH // 2, MENU_HEIGHT // 2 + 20))
            self.screen.blit(sub_surf, sub_rect)

        pygame.display.flip()

    def show_overlay_message(self, renderer, text, color, duration_ms):
        """Show an overlay message on the game screen."""
        font = pygame.font.Font(None, 36)
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=(config.WINDOW_WIDTH // 2, 50))

        # Draw on renderer's screen
        renderer.screen.blit(text_surf, text_rect)
        pygame.display.flip()
        pygame.time.wait(duration_ms)

    def wait_for_key(self):
        """Wait for any key press."""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    waiting = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False

    def run(self):
        """Main menu loop."""
        while self.running:
            if self.current_menu == "main":
                buttons = self.create_main_menu_buttons()
                self.draw_main_menu(buttons)

            elif self.current_menu == "level_select_manual":
                buttons = self.create_level_select_buttons()
                self.draw_level_select(buttons, "Select Level - Manual Play")

            elif self.current_menu == "level_select_watch":
                buttons = self.create_level_select_buttons()
                self.draw_level_select(buttons, "Select Level - Watch Agent")

            elif self.current_menu == "settings":
                buttons = self.create_settings_buttons()
                self.draw_settings(buttons)

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.current_menu != "main":
                            self.current_menu = "main"
                        else:
                            self.running = False

                for btn in buttons:
                    action = btn.handle_event(event)
                    if action:
                        self.handle_action(action)

            self.clock.tick(60)

        pygame.quit()

    def handle_action(self, action):
        """Handle button actions."""
        if action == "quit":
            self.running = False

        elif action == "manual_play":
            self.current_menu = "level_select_manual"

        elif action == "watch_agent":
            self.current_menu = "level_select_watch"

        elif action == "settings":
            self.current_menu = "settings"

        elif action == "back":
            self.current_menu = "main"

        elif action == "toggle_algorithm":
            if self.selected_algorithm == "qlearning":
                self.selected_algorithm = "sarsa"
            else:
                self.selected_algorithm = "qlearning"

        elif action == "toggle_intrinsic":
            self.use_intrinsic = not self.use_intrinsic

        elif action.startswith("level_"):
            level_id = int(action.split("_")[1])
            self.selected_level = level_id

            if self.current_menu == "level_select_manual":
                self.run_manual_play(level_id)
            elif self.current_menu == "level_select_watch":
                self.run_trained_agent(level_id)


def main():
    menu = MenuSystem()
    menu.run()


if __name__ == "__main__":
    main()
