"""Main interactive application window for Overcooked-AI Runner.

Provides:
- Live Overcooked gameplay rendering on the left pane (Chef 0 with red hat, Chef 1 with blue hat).
- Real-time layout preview that updates instantly when a layout is selected from the dropdown.
- Agent 1 and Agent 2 selection dropdowns (Human, PPO, Greedy, Upstream, Random, Stay).
- Pause and Run / Restart controls.
- "Done!" celebration state with [View Graph] button to open the Streamlit dashboard for the run.
- Telemetry logging to CSV compatible with dashboard/run_telemetry.py.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import webbrowser
from enum import Enum, auto
from pathlib import Path
from typing import Any

import pygame

from gui.theme import (
    WINDOW_DEFAULT_WIDTH,
    WINDOW_DEFAULT_HEIGHT,
    SIDEBAR_WIDTH,
    BG_WINDOW,
    BG_GAME_FRAME,
    BG_SIDEBAR,
    BG_CARD,
    BORDER_DEFAULT,
    BORDER_FOCUS,
    BORDER_SUBTLE,
    DIVIDER_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    COLOR_AGENT_RED,
    COLOR_AGENT_RED_BG,
    COLOR_AGENT_BLUE,
    COLOR_AGENT_BLUE_BG,
    COLOR_RUN,
    COLOR_RUN_HOVER,
    COLOR_PAUSE,
    COLOR_PAUSE_HOVER,
    COLOR_GRAPH,
    COLOR_GRAPH_HOVER,
    COLOR_DONE_BADGE,
    COLOR_DONE_BADGE_BG,
    FontManager,
)
from gui.widgets import UIButton, UIDropdown, UICard, UIBadge
from gui.agent_manager import (
    REPO_ROOT,
    AGENT_EVAL_DIR,
    InteractiveHumanAgent,
    create_agent,
    get_available_layouts,
    is_ppo_supported,
)
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


DEFAULT_LAYOUT = "cramped_room"
DEFAULT_HORIZON = 400
DEFAULT_FPS = 10
DEFAULT_TILE_SIZE = 75
LAST_RUN_CSV = AGENT_EVAL_DIR / "results" / "last_render_run.csv"
ALT_LAST_RUN_CSV = REPO_ROOT / "results" / "last_render_run.csv"


class AppState(Enum):
    SETUP = auto()      # Previewing initial layout and selecting options
    RUNNING = auto()    # Game running actively
    PAUSED = auto()     # Simulation paused
    DONE = auto()       # Horizon reached or episode done


class OvercookedApp:
    """The interactive desktop application for Overcooked-AI."""

    def __init__(
        self,
        initial_layout: str = DEFAULT_LAYOUT,
        horizon: int = DEFAULT_HORIZON,
        fps: int = DEFAULT_FPS,
        tile_size: int = DEFAULT_TILE_SIZE,
    ) -> None:
        pygame.init()
        pygame.display.init()
        pygame.font.init()

        self.horizon = horizon
        self.fps = fps
        self.tile_size = tile_size
        self.state = AppState.SETUP

        # Window dimensions
        self.window_width = WINDOW_DEFAULT_WIDTH
        self.window_height = WINDOW_DEFAULT_HEIGHT
        self.game_pane_width = self.window_width - SIDEBAR_WIDTH

        self.window = pygame.display.set_mode(
            (self.window_width, self.window_height), pygame.RESIZABLE
        )
        pygame.display.set_caption("Overcooked-AI Runner")

        self.clock = pygame.time.Clock()
        self.running = True

        # Layouts and Agents
        self.available_layouts = get_available_layouts()
        self.current_layout = initial_layout if initial_layout in self.available_layouts else self.available_layouts[0]

        self.agent_options = [
            ("Human (Keyboard)", "human"),
            ("PPO (Pretrained RL)", "ppo"),
            ("Greedy (BFS Search)", "greedy"),
            ("Upstream (Baseline)", "upstream"),
            ("Random Agent", "random"),
            ("Stay (Idle)", "stay"),
        ]

        self.selected_agent_0_type = "human"
        self.selected_agent_1_type = "ppo"

        # Game simulation state
        self.mdp: OvercookedGridworld | None = None
        self.env: OvercookedEnv | None = None
        self.visualizer: StateVisualizer | None = None
        self.agents: list[Any] = []
        self.step_count = 0
        self.cumulative_score = 0.0
        self.recorded_rows: list[dict[str, Any]] = []
        self.status_message: str = "Ready. Select agents & layout, then click Run!"
        self.warning_message: str | None = None

        # Background Streamlit process tracker
        self.streamlit_proc: subprocess.Popen[Any] | None = None

        # Build visualizer and initial environment
        self._init_visualizer()
        self._load_environment(self.current_layout)

        # Setup GUI widgets
        self._setup_widgets()

    def _init_visualizer(self) -> None:
        """Create StateVisualizer with red hat for Chef 0 and blue hat for Chef 1."""
        self.visualizer = StateVisualizer(
            tile_size=self.tile_size,
            player_colors=["red", "blue"],  # Agent 1 = Red Hat, Agent 2 = Blue Hat
            is_rendering_hud=True,
        )

    def _load_environment(self, layout_name: str) -> None:
        """Load or reload the Overcooked environment for a specific layout."""
        try:
            self.current_layout = layout_name
            self.mdp = OvercookedGridworld.from_layout_name(layout_name)
            self.env = OvercookedEnv.from_mdp(self.mdp, horizon=self.horizon)
            self.step_count = 0
            self.cumulative_score = 0.0
            self.recorded_rows.clear()
            self.status_message = f"Layout '{layout_name}' loaded."
            self.warning_message = None

            # Check PPO availability notification
            if self.selected_agent_1_type == "ppo" or self.selected_agent_0_type == "ppo":
                if not is_ppo_supported(layout_name):
                    self.warning_message = f"Note: PPO is not bundled for '{layout_name}'. Will use Greedy."
        except Exception as e:
            self.status_message = f"Error loading layout: {e}"

    def _setup_widgets(self) -> None:
        """Initialize sidebar dropdowns and buttons."""
        sb_x = self.game_pane_width + 16
        sb_w = SIDEBAR_WIDTH - 32
        item_h = 36

        # Agent 1 dropdown (Red hat)
        self.dropdown_agent_0 = UIDropdown(
            rect=(sb_x, 90, sb_w, item_h),
            options=self.agent_options,
            selected_value=self.selected_agent_0_type,
            on_change=self._on_agent_0_changed,
        )

        # Agent 2 dropdown (Blue hat)
        self.dropdown_agent_1 = UIDropdown(
            rect=(sb_x, 175, sb_w, item_h),
            options=self.agent_options,
            selected_value=self.selected_agent_1_type,
            on_change=self._on_agent_1_changed,
        )

        # Layout dropdown
        layout_opts = [(name, name) for name in self.available_layouts]
        self.dropdown_layout = UIDropdown(
            rect=(sb_x, 260, sb_w, item_h),
            options=layout_opts,
            selected_value=self.current_layout,
            on_change=self._on_layout_changed,
            max_visible_items=8,
        )

        # Action buttons
        btn_y = self.window_height - 90
        half_w = (sb_w - 12) // 2

        self.btn_pause = UIButton(
            rect=(sb_x, btn_y, half_w, 42),
            text="Pause",
            on_click=self.toggle_pause,
            bg_color=COLOR_PAUSE,
            hover_color=COLOR_PAUSE_HOVER,
            font_size=15,
        )

        self.btn_run = UIButton(
            rect=(sb_x + half_w + 12, btn_y, half_w, 42),
            text="Run",
            on_click=self.start_or_restart_game,
            bg_color=COLOR_RUN,
            hover_color=COLOR_RUN_HOVER,
            font_size=15,
        )

        # View Graph button (opens Streamlit dashboard)
        self.btn_view_graph = UIButton(
            rect=(sb_x, btn_y - 52, sb_w, 42),
            text="View Graph (Streamlit)",
            on_click=self.open_streamlit_dashboard,
            bg_color=COLOR_GRAPH,
            hover_color=COLOR_GRAPH_HOVER,
            font_size=14,
        )

    def _on_agent_0_changed(self, new_val: str) -> None:
        self.selected_agent_0_type = new_val
        self.warning_message = None
        if new_val == "ppo" and not is_ppo_supported(self.current_layout):
            self.warning_message = f"Note: PPO is not bundled for '{self.current_layout}'. Will use Greedy."

    def _on_agent_1_changed(self, new_val: str) -> None:
        self.selected_agent_1_type = new_val
        self.warning_message = None
        if new_val == "ppo" and not is_ppo_supported(self.current_layout):
            self.warning_message = f"Note: PPO is not bundled for '{self.current_layout}'. Will use Greedy."

    def _on_layout_changed(self, new_layout: str) -> None:
        """Handle live layout switching from dropdown."""
        print(f"[GUI] Layout changed live to: {new_layout}")
        self._load_environment(new_layout)
        if self.state in (AppState.RUNNING, AppState.PAUSED):
            self.state = AppState.SETUP

    def start_or_restart_game(self) -> None:
        """Start or restart the game with the selected agents and layout."""
        if self.mdp is None or self.env is None:
            return

        print(f"[GUI] Starting episode on {self.current_layout} (Agent 0: {self.selected_agent_0_type}, Agent 1: {self.selected_agent_1_type})")
        self.status_message = "Initializing agents..."
        self._draw_frame()  # Render status feedback immediately

        # Reset environment
        self.env.reset()
        self.step_count = 0
        self.cumulative_score = 0.0
        self.recorded_rows.clear()

        # Instantiate agents
        a0, warn0 = create_agent(self.selected_agent_0_type, 0, self.mdp, self.current_layout)
        a1, warn1 = create_agent(self.selected_agent_1_type, 1, self.mdp, self.current_layout)
        self.agents = [a0, a1]

        if warn0 or warn1:
            self.warning_message = warn0 or warn1

        self.state = AppState.RUNNING
        self.status_message = f"Playing {self.current_layout}!"

    def toggle_pause(self) -> None:
        """Toggle simulation pause state."""
        if self.state == AppState.RUNNING:
            self.state = AppState.PAUSED
            self.status_message = "Simulation paused."
        elif self.state == AppState.PAUSED:
            self.state = AppState.RUNNING
            self.status_message = "Simulation resumed."

    def open_streamlit_dashboard(self) -> None:
        """Launch and display the Streamlit telemetry dashboard."""
        self._save_telemetry()
        self.status_message = "Opening Streamlit dashboard in browser..."

        # Check if streamlit is already started by us
        if self.streamlit_proc is None or self.streamlit_proc.poll() is not None:
            venv_python = AGENT_EVAL_DIR / ".venv" / "bin" / "python"
            python_bin = str(venv_python) if venv_python.is_file() else sys.executable
            dashboard_script = REPO_ROOT / "dashboard" / "run_telemetry.py"

            cmd = [
                python_bin,
                "-m",
                "streamlit",
                "run",
                str(dashboard_script),
                "--server.headless",
                "true",
                "--server.port",
                "8501",
            ]
            print(f"[GUI] Launching Streamlit: {' '.join(cmd)}")
            try:
                self.streamlit_proc = subprocess.Popen(
                    cmd, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except Exception as e:
                print(f"[GUI] Could not launch Streamlit subprocess: {e}")

        # Open web browser
        webbrowser.open("http://localhost:8501")

    def _save_telemetry(self) -> None:
        """Save recorded gameplay rows to both telemetry CSV locations."""
        if not self.recorded_rows:
            return

        for target in (LAST_RUN_CSV, ALT_LAST_RUN_CSV):
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(self.recorded_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(self.recorded_rows)
                print(f"[GUI] Telemetry saved to: {target.resolve()}")
            except Exception as e:
                print(f"[GUI] Error saving telemetry to {target}: {e}")

    def _action_to_name(self, action: Any) -> str:
        name_map = {
            Direction.NORTH: "north",
            Direction.SOUTH: "south",
            Direction.EAST: "east",
            Direction.WEST: "west",
            Action.STAY: "stay",
            Action.INTERACT: "interact",
            (0, -1): "north",
            (0, 1): "south",
            (1, 0): "east",
            (-1, 0): "west",
            (0, 0): "stay",
            "interact": "interact",
        }
        return name_map.get(action, str(action))

    def _step_simulation(self) -> None:
        """Advance the Overcooked simulation by one timestep."""
        if self.state != AppState.RUNNING or not self.agents or self.env is None:
            return

        state = self.env.state
        joint_action = tuple(agent.action(state)[0] for agent in self.agents)

        next_state, sparse_reward, done, info = self.env.step(joint_action)
        self.cumulative_score += sparse_reward
        self.step_count += 1

        shaped = [0.0, 0.0]
        if isinstance(info, dict) and "shaped_r_by_agent" in info:
            shaped = info.get("shaped_r_by_agent", [0.0, 0.0])

        self.recorded_rows.append({
            "episode": 1,
            "timestep": self.step_count,
            "agent_0_action": self._action_to_name(joint_action[0]),
            "agent_1_action": self._action_to_name(joint_action[1]),
            "sparse_reward": float(sparse_reward),
            "agent_0_shaped_reward": float(shaped[0]),
            "agent_1_shaped_reward": float(shaped[1]),
            "cumulative_sparse_reward": float(self.cumulative_score),
            "agent_0_position": repr(next_state.players[0].position),
            "agent_1_position": repr(next_state.players[1].position),
            "done": done,
            "layout_name": self.current_layout,
        })

        if done or self.step_count >= self.horizon:
            self.state = AppState.DONE
            self.status_message = f"Episode Finished! Final Score: {int(self.cumulative_score)}"
            self._save_telemetry()
            print(f"[GUI] Episode Done at step {self.step_count}. Score: {self.cumulative_score}")

    def handle_events(self) -> None:
        """Process keyboard, mouse, and window events."""
        events = pygame.event.get()
        # Active dropdown reference to close others when one opens
        dropdowns = [self.dropdown_agent_0, self.dropdown_agent_1, self.dropdown_layout]

        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                return

            elif event.type == pygame.VIDEORESIZE:
                self.window_width = max(800, event.w)
                self.window_height = max(600, event.h)
                self.game_pane_width = self.window_width - SIDEBAR_WIDTH
                self.window = pygame.display.set_mode(
                    (self.window_width, self.window_height), pygame.RESIZABLE
                )
                self._setup_widgets()

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                    return

                elif event.key == pygame.K_p:
                    self.toggle_pause()

                elif event.key == pygame.K_r:
                    self.start_or_restart_game()

                # Dispatch keyboard input to interactive human agents
                if self.state == AppState.RUNNING and self.agents:
                    two_humans = sum(isinstance(a, InteractiveHumanAgent) for a in self.agents) > 1
                    for a in self.agents:
                        if isinstance(a, InteractiveHumanAgent):
                            a.handle_keydown(event.key, is_two_player=two_humans)

            # Dropdown events (open dropdown gets priority)
            open_dd = next((dd for dd in dropdowns if dd.is_open), None)
            if open_dd is not None:
                consumed = open_dd.handle_event(event)
                if consumed:
                    continue

            # Check other dropdowns
            for dd in dropdowns:
                if dd is not open_dd:
                    dd.handle_event(event)

            # Buttons events
            self.btn_pause.handle_event(event)
            self.btn_run.handle_event(event)
            if self.state == AppState.DONE or self.recorded_rows:
                self.btn_view_graph.handle_event(event)

    def _draw_frame(self) -> None:
        """Render the complete application frame (game canvas on left, sidebar on right)."""
        self.window.fill(BG_WINDOW)

        # ----------------------------------------------------
        # 1. Left Game Canvas Pane
        # ----------------------------------------------------
        game_rect = pygame.Rect(0, 0, self.game_pane_width, self.window_height)
        pygame.draw.rect(self.window, BG_GAME_FRAME, game_rect)
        pygame.draw.line(self.window, DIVIDER_COLOR, (self.game_pane_width, 0), (self.game_pane_width, self.window_height), width=1)

        if self.env is not None and self.visualizer is not None:
            time_left = max(0, self.horizon - self.step_count)
            hud_data = StateVisualizer.default_hud_data(
                self.env.state,
                score=self.cumulative_score,
                time_left=time_left,
            )
            game_surface = self.visualizer.render_state(
                self.env.state, self.env.mdp.terrain_mtx, hud_data=hud_data
            )

            # Center surface in the game pane (and scale if larger than pane)
            gw, gh = game_surface.get_size()
            avail_w = self.game_pane_width - 32
            avail_h = self.window_height - 32

            if gw > avail_w or gh > avail_h:
                scale = min(avail_w / gw, avail_h / gh)
                new_size = (int(gw * scale), int(gh * scale))
                game_surface = pygame.transform.smoothscale(game_surface, new_size)
                gw, gh = new_size

            render_x = (self.game_pane_width - gw) // 2
            render_y = (self.window_height - gh) // 2
            self.window.blit(game_surface, (render_x, render_y))

        # ----------------------------------------------------
        # 2. Right Control Sidebar
        # ----------------------------------------------------
        sb_x = self.game_pane_width + 16
        sb_w = SIDEBAR_WIDTH - 32

        # Header Title
        font_title = FontManager.get_font(18, bold=True)
        title_surf = font_title.render("Overcooked-AI", True, TEXT_PRIMARY)
        self.window.blit(title_surf, (sb_x, 18))

        # Status Badge (Top right of sidebar)
        if self.state == AppState.DONE:
            UIBadge.draw(self.window, sb_x + sb_w - 45, 28, "Done! 🎉", COLOR_DONE_BADGE_BG, COLOR_DONE_BADGE, font_size=13, bold=True)
        elif self.state == AppState.RUNNING:
            UIBadge.draw(self.window, sb_x + sb_w - 45, 28, "Running ⚡", (20, 60, 45), (52, 211, 153), font_size=12, bold=True)
        elif self.state == AppState.PAUSED:
            UIBadge.draw(self.window, sb_x + sb_w - 45, 28, "Paused ⏸", (70, 45, 15), (251, 191, 36), font_size=12, bold=True)
        else:
            UIBadge.draw(self.window, sb_x + sb_w - 45, 28, "Ready", (35, 42, 60), TEXT_SECONDARY, font_size=12, bold=False)

        # Agent 1 Section
        lbl_font = FontManager.get_font(13, bold=True)
        a0_lbl = lbl_font.render("Agent 1", True, TEXT_PRIMARY)
        self.window.blit(a0_lbl, (sb_x, 68))
        UIBadge.draw(self.window, sb_x + 130, 75, "Red Hat 🔴", COLOR_AGENT_RED_BG, COLOR_AGENT_RED, font_size=11, bold=True)
        self.dropdown_agent_0.draw(self.window)

        # Agent 2 Section
        a1_lbl = lbl_font.render("Agent 2", True, TEXT_PRIMARY)
        self.window.blit(a1_lbl, (sb_x, 153))
        UIBadge.draw(self.window, sb_x + 130, 160, "Blue Hat 🔵", COLOR_AGENT_BLUE_BG, COLOR_AGENT_BLUE, font_size=11, bold=True)
        self.dropdown_agent_1.draw(self.window)

        # Layout Section
        lay_lbl = lbl_font.render("Layout", True, TEXT_PRIMARY)
        self.window.blit(lay_lbl, (sb_x, 238))
        hint_font = FontManager.get_font(11, bold=False)
        hint_surf = hint_font.render("(live preview)", True, TEXT_MUTED)
        self.window.blit(hint_surf, (sb_x + 60, 240))
        self.dropdown_layout.draw(self.window)

        # Telemetry & Status Card
        card_y = 315
        card_h = 135
        UICard.draw(self.window, (sb_x, card_y, sb_w, card_h), bg_color=BG_CARD, border_color=BORDER_DEFAULT)

        card_title_font = FontManager.get_font(12, bold=True)
        card_t = card_title_font.render("RUN TELEMETRY & STATS", True, TEXT_MUTED)
        self.window.blit(card_t, (sb_x + 12, card_y + 10))

        # Metrics values
        metric_font = FontManager.get_font(14, bold=False)
        m_step = metric_font.render(f"Step: {self.step_count} / {self.horizon}", True, TEXT_PRIMARY)
        m_score = metric_font.render(f"Score: {int(self.cumulative_score)}", True, (52, 211, 153))
        m_speed = metric_font.render(f"Framerate: {self.fps} FPS", True, TEXT_SECONDARY)
        self.window.blit(m_step, (sb_x + 12, card_y + 34))
        self.window.blit(m_score, (sb_x + 12, card_y + 58))
        self.window.blit(m_speed, (sb_x + 12, card_y + 82))

        # Progress bar
        prog_pct = min(1.0, self.step_count / self.horizon) if self.horizon > 0 else 0
        pbar_rect = pygame.Rect(sb_x + 12, card_y + 110, sb_w - 24, 8)
        pygame.draw.rect(self.window, (20, 24, 34), pbar_rect, border_radius=4)
        if prog_pct > 0:
            fill_rect = pygame.Rect(sb_x + 12, card_y + 110, int((sb_w - 24) * prog_pct), 8)
            p_color = COLOR_DONE_BADGE if self.state == AppState.DONE else (99, 102, 241)
            pygame.draw.rect(self.window, p_color, fill_rect, border_radius=4)

        # Controls & Help Card
        help_y = card_y + card_h + 14
        help_h = 105
        UICard.draw(self.window, (sb_x, help_y, sb_w, help_h), bg_color=(24, 28, 40), border_color=BORDER_SUBTLE)
        help_t = card_title_font.render("KEYBOARD CONTROLS", True, TEXT_MUTED)
        self.window.blit(help_t, (sb_x + 12, help_y + 8))

        h_f = FontManager.get_font(11, bold=False)
        h1 = h_f.render("• WASD / Arrows : Move Chef 0 (Red)", True, TEXT_SECONDARY)
        h2 = h_f.render("• Space / Enter / F : Pick up / Drop / Cook", True, TEXT_SECONDARY)
        h3 = h_f.render("• [P] : Pause / Resume   • [R] : Restart", True, TEXT_MUTED)
        self.window.blit(h1, (sb_x + 12, help_y + 30))
        self.window.blit(h2, (sb_x + 12, help_y + 50))
        self.window.blit(h3, (sb_x + 12, help_y + 72))

        # Warning / Info Message if any
        if self.warning_message:
            warn_font = FontManager.get_font(11, bold=False)
            warn_surf = warn_font.render(self.warning_message, True, (251, 191, 36))
            self.window.blit(warn_surf, (sb_x, help_y + help_h + 10))

        # Update button text states
        self.btn_pause.text = "Resume" if self.state == AppState.PAUSED else "Pause"
        self.btn_run.text = "Restart" if self.state in (AppState.RUNNING, AppState.PAUSED) else "Run"

        # Draw Bottom Action Buttons
        self.btn_pause.draw(self.window)
        self.btn_run.draw(self.window)

        # Draw View Graph button (prominently visible when done or when run data exists)
        if self.state == AppState.DONE or self.recorded_rows:
            self.btn_view_graph.is_visible = True
            self.btn_view_graph.draw(self.window)
        else:
            self.btn_view_graph.is_visible = False

        # ----------------------------------------------------
        # 3. Floating Overlay Pass (Draw open dropdown popups on top)
        # ----------------------------------------------------
        for dd in (self.dropdown_layout, self.dropdown_agent_1, self.dropdown_agent_0):
            dd.draw_overlay(self.window)

        pygame.display.flip()

    def run(self) -> None:
        """Main application execution loop."""
        print("[GUI] Overcooked-AI Interactive Runner is ready.")
        try:
            while self.running:
                self.handle_events()
                self._step_simulation()
                self._draw_frame()
                self.clock.tick(self.fps if self.state == AppState.RUNNING else 30)
        finally:
            pygame.display.quit()
            pygame.quit()
            if self.streamlit_proc is not None:
                try:
                    self.streamlit_proc.terminate()
                except Exception:
                    pass
            print("[GUI] Closed cleanly.")
