"""Main interactive application window for Overcooked-AI Runner.

Provides:
- Live Overcooked gameplay rendering filling the entire left pane.
- Red Hat banner for Chef 0 and Blue Hat banner for Chef 1 matching the UI button style.
- Real-time layout preview that updates instantly when a layout is selected from the dropdown.
- Mid-run locking of dropdowns (no changing agents or maps mid-run).
- Pause and Run / Restart controls.
- "Done!" celebration state where the Streamlit [View Graph] button appears.
- Telemetry logging to CSV compatible with dashboard/run_telemetry.py.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import webbrowser
from enum import Enum, auto
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pygame

from gui.theme import (
    WINDOW_DEFAULT_WIDTH,
    WINDOW_DEFAULT_HEIGHT,
    SIDEBAR_WIDTH,
    BG_WINDOW,
    BG_GAME_FRAME,
    BG_SIDEBAR,
    BG_CARD,
    BG_INPUT,
    BORDER_DEFAULT,
    BORDER_FOCUS,
    BORDER_SUBTLE,
    DIVIDER_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    COLOR_AGENT_RED,
    COLOR_AGENT_BLUE,
    COLOR_RUN,
    COLOR_RUN_HOVER,
    COLOR_RUN_TEXT,
    COLOR_PAUSE,
    COLOR_PAUSE_HOVER,
    COLOR_PAUSE_TEXT,
    COLOR_RESET,
    COLOR_RESET_HOVER,
    COLOR_RESET_TEXT,
    COLOR_GRAPH,
    COLOR_GRAPH_HOVER,
    COLOR_GRAPH_TEXT,
    COLOR_DONE_BADGE,
    FontManager,
)
from gui.widgets import UIButton, UIDropdown, UICard, UIBanner
from gui.agent_manager import (
    REPO_ROOT,
    AGENT_EVAL_DIR,
    InteractiveHumanAgent,
    create_agent,
    get_available_layouts,
    is_ppo_supported,
)
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
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
    DONE = auto()       # Horizon reached or episode completed


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
        self.current_layout = (
            initial_layout if initial_layout in self.available_layouts else self.available_layouts[0]
        )

        self.agent_options = [
            ("Human (Keyboard)", "human"),
            ("PPO (Pretrained RL)", "ppo"),
            ("Greedy Agent (Upstream)", "greedy"),
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

        # Build initial environment and visualizer
        self._load_environment(self.current_layout)

        # Setup GUI widgets
        self._setup_widgets()

    @property
    def is_mid_run(self) -> bool:
        """Return True if simulation is actively running or paused."""
        return self.state in (AppState.RUNNING, AppState.PAUSED)

    def _load_environment(self, layout_name: str) -> None:
        """Load or reload the Overcooked environment and calculate optimal high-res tile size."""
        try:
            self.current_layout = layout_name
            evaluator = AgentEvaluator.from_layout_name(
                {"layout_name": layout_name, "old_dynamics": True},
                {"horizon": self.horizon},
            )
            self.env = evaluator.env
            self.mdp = self.env.mdp
            self.step_count = 0
            self.cumulative_score = 0.0
            self.recorded_rows.clear()
            self.status_message = f"Layout '{layout_name}' loaded."
            self.warning_message = None

            # Calculate optimal tile size so the visualizer generates crisp sprites
            grid_w = len(self.mdp.terrain_mtx[0])
            grid_h = len(self.mdp.terrain_mtx)
            tile_w = self.game_pane_width / grid_w
            tile_h = (self.window_height - 100) / grid_h
            optimal_tile_size = max(40, min(180, int(min(tile_w, tile_h))))

            self.visualizer = StateVisualizer(
                tile_size=optimal_tile_size,
                player_colors=["red", "blue"],  # Agent 1 = Red Hat, Agent 2 = Blue Hat
                is_rendering_hud=True,
            )

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

        # Bottom Action Buttons
        btn_y = self.window_height - 64
        half_w = (sb_w - 12) // 2

        self.btn_pause = UIButton(
            rect=(sb_x, btn_y, half_w, 44),
            text="Pause",
            on_click=self.toggle_pause,
            bg_color=COLOR_PAUSE,
            hover_color=COLOR_PAUSE_HOVER,
            text_color=COLOR_PAUSE_TEXT,
            font_size=14,
        )

        self.btn_reset = UIButton(
            rect=(sb_x, btn_y, half_w, 44),
            text="Reset",
            on_click=self.reset_to_setup,
            bg_color=COLOR_RESET,
            hover_color=COLOR_RESET_HOVER,
            text_color=COLOR_RESET_TEXT,
            font_size=14,
        )

        self.btn_run = UIButton(
            rect=(sb_x, btn_y, sb_w, 44),
            text="Run",
            on_click=self.handle_run_or_restart_click,
            bg_color=COLOR_RUN,
            hover_color=COLOR_RUN_HOVER,
            text_color=COLOR_RUN_TEXT,
            font_size=14,
        )

        # View Graph button (ONLY appears and functions when DONE)
        self.btn_view_graph = UIButton(
            rect=(sb_x, btn_y - 54, sb_w, 44),
            text="View Graph (Streamlit)",
            on_click=self.open_streamlit_dashboard,
            bg_color=COLOR_GRAPH,
            hover_color=COLOR_GRAPH_HOVER,
            text_color=COLOR_GRAPH_TEXT,
            font_size=13,
        )
        self.btn_view_graph.is_visible = False

    def _on_agent_0_changed(self, new_val: str) -> None:
        if self.is_mid_run:
            return  # Locked mid-run
        self.selected_agent_0_type = new_val
        self.warning_message = None
        if new_val == "ppo" and not is_ppo_supported(self.current_layout):
            self.warning_message = f"Note: PPO is not bundled for '{self.current_layout}'. Will use Greedy."
        if self.state == AppState.DONE:
            self.state = AppState.SETUP
            self.btn_view_graph.is_visible = False

    def _on_agent_1_changed(self, new_val: str) -> None:
        if self.is_mid_run:
            return  # Locked mid-run
        self.selected_agent_1_type = new_val
        self.warning_message = None
        if new_val == "ppo" and not is_ppo_supported(self.current_layout):
            self.warning_message = f"Note: PPO is not bundled for '{self.current_layout}'. Will use Greedy."
        if self.state == AppState.DONE:
            self.state = AppState.SETUP
            self.btn_view_graph.is_visible = False

    def _on_layout_changed(self, new_layout: str) -> None:
        """Handle live layout switching from dropdown when not mid-run."""
        if self.is_mid_run:
            return  # Locked mid-run
        print(f"[GUI] Layout changed live to: {new_layout}")
        self._load_environment(new_layout)
        if self.state == AppState.DONE:
            self.state = AppState.SETUP
            self.btn_view_graph.is_visible = False

    def handle_run_or_restart_click(self) -> None:
        """Handle primary run button click."""
        if self.is_mid_run:
            self.reset_to_setup()
        else:
            self.start_game()

    def reset_to_setup(self) -> None:
        """Reset the current simulation and return to SETUP state so user can choose different agents and layouts."""
        self._load_environment(self.current_layout)
        self.state = AppState.SETUP
        self.status_message = "Ready. Select agents & layout, then click RUN!"
        self.warning_message = None
        self.btn_view_graph.is_visible = False
        print("[GUI] Reset to SETUP mode.")

    def start_game(self) -> None:
        """Start simulation episode with the selected agents and layout."""
        if self.mdp is None or self.env is None:
            return

        print(f"[GUI] Starting episode on {self.current_layout} (Agent 0: {self.selected_agent_0_type}, Agent 1: {self.selected_agent_1_type})")
        if self.selected_agent_0_type == "ppo" or self.selected_agent_1_type == "ppo":
            self.status_message = "Loading PPO neural network weights..."
        else:
            self.status_message = "Initializing agents..."
        self._draw_frame()

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
        self.btn_view_graph.is_visible = False

    def toggle_pause(self) -> None:
        """Toggle simulation pause state."""
        if self.state == AppState.RUNNING:
            self.state = AppState.PAUSED
            self.status_message = "Simulation paused."
        elif self.state == AppState.PAUSED:
            self.state = AppState.RUNNING
            self.status_message = "Simulation resumed."

    def open_streamlit_dashboard(self) -> None:
        """Launch and display the Streamlit telemetry dashboard. Only accessible when DONE."""
        if self.state != AppState.DONE:
            return

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

        # Mid-run lock: dropdowns cannot be opened or changed mid-run
        is_locked = self.is_mid_run
        self.dropdown_agent_0.is_enabled = not is_locked
        self.dropdown_agent_1.is_enabled = not is_locked
        self.dropdown_layout.is_enabled = not is_locked
        if is_locked:
            self.dropdown_agent_0.is_open = False
            self.dropdown_agent_1.is_open = False
            self.dropdown_layout.is_open = False

        dropdowns = [self.dropdown_agent_0, self.dropdown_agent_1, self.dropdown_layout]

        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                return

            elif event.type == pygame.VIDEORESIZE:
                self.window_width = max(900, event.w)
                self.window_height = max(600, event.h)
                self.game_pane_width = self.window_width - SIDEBAR_WIDTH
                self.window = pygame.display.set_mode(
                    (self.window_width, self.window_height), pygame.RESIZABLE
                )
                self._load_environment(self.current_layout)
                self._setup_widgets()

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                    return

                elif event.key == pygame.K_p and self.is_mid_run:
                    self.toggle_pause()

                elif event.key == pygame.K_r:
                    if self.is_mid_run or self.state == AppState.DONE:
                        self.reset_to_setup()
                    else:
                        self.start_game()

                # Dispatch keyboard input to interactive human agents
                if self.state == AppState.RUNNING and self.agents:
                    two_humans = sum(isinstance(a, InteractiveHumanAgent) for a in self.agents) > 1
                    for a in self.agents:
                        if isinstance(a, InteractiveHumanAgent):
                            a.handle_keydown(event.key, is_two_player=two_humans)

            # Dropdown events (only processed when enabled)
            if not is_locked:
                open_dd = next((dd for dd in dropdowns if dd.is_open), None)
                if open_dd is not None:
                    consumed = open_dd.handle_event(event)
                    if consumed:
                        continue

                for dd in dropdowns:
                    if dd is not open_dd:
                        dd.handle_event(event)
            else:
                # When locked (mid-run or paused), clicking on a dropdown provides immediate feedback
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for dd in dropdowns:
                        if dd.rect.collidepoint(event.pos):
                            mode_str = "PAUSED" if self.state == AppState.PAUSED else "RUNNING"
                            self.status_message = f"Locked while {mode_str}. Click RESET to change values."

            # Buttons events
            if self.is_mid_run:
                self.btn_pause.handle_event(event)
                self.btn_reset.handle_event(event)
            elif self.state == AppState.SETUP:
                self.btn_run.handle_event(event)
            elif self.state == AppState.DONE:
                self.btn_view_graph.handle_event(event)
                self.btn_reset.handle_event(event)
                self.btn_run.handle_event(event)

    def _draw_frame(self) -> None:
        """Render the complete application frame with full-sized environment and clean sidebar."""
        self.window.fill(BG_WINDOW)

        # ----------------------------------------------------
        # 1. Left Game Canvas Pane (Entire size)
        # ----------------------------------------------------
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

            # Stretch/scale to completely fill the entire left pane!
            scaled_surface = pygame.transform.smoothscale(
                game_surface, (self.game_pane_width, self.window_height)
            )
            self.window.blit(scaled_surface, (0, 0))

        # Vertical divider line separating game pane and sidebar
        pygame.draw.line(
            self.window,
            DIVIDER_COLOR,
            (self.game_pane_width, 0),
            (self.game_pane_width, self.window_height),
            width=1,
        )

        # ----------------------------------------------------
        # 2. Right Control Sidebar (Modern Dark Theme)
        # ----------------------------------------------------
        sb_start_x = self.game_pane_width
        sb_x = self.game_pane_width + 16
        sb_w = SIDEBAR_WIDTH - 32

        # Sidebar background fill
        pygame.draw.rect(
            self.window, BG_SIDEBAR, (sb_start_x, 0, SIDEBAR_WIDTH, self.window_height)
        )

        # Clean vertical divider separating game canvas and sidebar
        pygame.draw.line(
            self.window,
            DIVIDER_COLOR,
            (sb_start_x, 0),
            (sb_start_x, self.window_height),
            width=1,
        )

        # Header Title
        font_title = FontManager.get_font(18, bold=True)
        title_surf = font_title.render("OVERCOOKED-AI", True, TEXT_PRIMARY)
        self.window.blit(title_surf, (sb_x, 18))
        sub_font = FontManager.get_font(11, bold=False)
        sub_surf = sub_font.render("Interactive Runner", True, TEXT_MUTED)
        self.window.blit(sub_surf, (sb_x + title_surf.get_width() + 10, 24))

        # Agent 1 Section
        lbl_font = FontManager.get_font(12, bold=True)
        a0_lbl = lbl_font.render("Agent 1", True, TEXT_SECONDARY)
        self.window.blit(a0_lbl, (sb_x, 68))
        # Solid Red Hat banner styled right next to the label
        a0_badge_x = sb_x + a0_lbl.get_width() + 10
        UIBanner.draw(self.window, (a0_badge_x, 65, 78, 22), "RED HAT", COLOR_AGENT_RED, font_size=10, bold=True)
        self.dropdown_agent_0.draw(self.window)

        # Agent 2 Section
        a1_lbl = lbl_font.render("Agent 2", True, TEXT_SECONDARY)
        self.window.blit(a1_lbl, (sb_x, 153))
        # Solid Blue Hat banner styled right next to the label
        a1_badge_x = sb_x + a1_lbl.get_width() + 10
        UIBanner.draw(self.window, (a1_badge_x, 150, 78, 22), "BLUE HAT", COLOR_AGENT_BLUE, font_size=10, bold=True)
        self.dropdown_agent_1.draw(self.window)

        # Layout Section
        lay_lbl = lbl_font.render("Layout / Map", True, TEXT_SECONDARY)
        self.window.blit(lay_lbl, (sb_x, 238))
        self.dropdown_layout.draw(self.window)

        # Telemetry & Status Card
        card_y = 315
        card_h = 135
        UICard.draw(self.window, (sb_x, card_y, sb_w, card_h), bg_color=BG_CARD, border_color=BORDER_DEFAULT, border_radius=8)

        card_title_font = FontManager.get_font(11, bold=True)
        card_t = card_title_font.render("RUN TELEMETRY & STATS", True, TEXT_MUTED)
        self.window.blit(card_t, (sb_x + 12, card_y + 10))

        metric_font = FontManager.get_font(13, bold=False)
        m_step = metric_font.render(f"Step: {self.step_count} / {self.horizon}", True, TEXT_PRIMARY)
        m_score = metric_font.render(f"Score: {int(self.cumulative_score)}", True, (52, 211, 153))
        m_speed = metric_font.render(f"Speed: {self.fps} FPS", True, TEXT_SECONDARY)
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
        UICard.draw(self.window, (sb_x, help_y, sb_w, help_h), bg_color=(24, 28, 40), border_color=BORDER_SUBTLE, border_radius=8)
        help_t = card_title_font.render("KEYBOARD CONTROLS", True, TEXT_MUTED)
        self.window.blit(help_t, (sb_x + 12, help_y + 8))

        h_f = FontManager.get_font(11, bold=False)
        h1 = h_f.render("• WASD / Arrows : Move Chef 0 (Red)", True, TEXT_SECONDARY)
        h2 = h_f.render("• Space / Enter / F : Pick up / Drop / Cook", True, TEXT_SECONDARY)
        h3 = h_f.render("• [P] : Pause / Resume   • [R] : Reset", True, TEXT_MUTED)
        self.window.blit(h1, (sb_x + 12, help_y + 30))
        self.window.blit(h2, (sb_x + 12, help_y + 50))
        self.window.blit(h3, (sb_x + 12, help_y + 72))

        # Warning / Info Message if any
        if self.warning_message:
            warn_font = FontManager.get_font(11, bold=False)
            warn_surf = warn_font.render(self.warning_message, True, COLOR_PAUSE)
            self.window.blit(warn_surf, (sb_x, help_y + help_h + 8))

        # ----------------------------------------------------
        # Dynamic Action Buttons Layout
        # ----------------------------------------------------
        btn_y = self.window_height - 64
        half_w = (sb_w - 12) // 2

        if self.state == AppState.SETUP:
            # Full width Run button
            self.btn_run.rect = pygame.Rect(sb_x, btn_y, sb_w, 44)
            self.btn_run.text = "Run"
            self.btn_run.bg_color = COLOR_RUN
            self.btn_run.hover_color = COLOR_RUN_HOVER
            self.btn_run.text_color = COLOR_RUN_TEXT
            self.btn_run.is_visible = True
            self.btn_run.draw(self.window)

        elif self.state in (AppState.RUNNING, AppState.PAUSED):
            # Side by side: Pause/Resume and Reset
            self.btn_pause.rect = pygame.Rect(sb_x, btn_y, half_w, 44)
            self.btn_pause.text = "Resume" if self.state == AppState.PAUSED else "Pause"
            self.btn_pause.is_visible = True
            self.btn_pause.draw(self.window)

            self.btn_reset.rect = pygame.Rect(sb_x + half_w + 12, btn_y, half_w, 44)
            self.btn_reset.text = "Reset"
            self.btn_reset.is_visible = True
            self.btn_reset.draw(self.window)

        elif self.state == AppState.DONE:
            # Streamlit View Graph button ONLY appears when DONE!
            self.btn_view_graph.rect = pygame.Rect(sb_x, btn_y - 54, sb_w, 44)
            self.btn_view_graph.text = "View Graph (Streamlit)"
            self.btn_view_graph.is_visible = True
            self.btn_view_graph.draw(self.window)

            # Two buttons: Reset (to clear and choose different agents/layouts) and Run Again
            self.btn_reset.rect = pygame.Rect(sb_x, btn_y, half_w, 44)
            self.btn_reset.text = "Reset"
            self.btn_reset.is_visible = True
            self.btn_reset.draw(self.window)

            self.btn_run.rect = pygame.Rect(sb_x + half_w + 12, btn_y, half_w, 44)
            self.btn_run.text = "Run Again"
            self.btn_run.bg_color = COLOR_RUN
            self.btn_run.hover_color = COLOR_RUN_HOVER
            self.btn_run.text_color = COLOR_RUN_TEXT
            self.btn_run.is_visible = True
            self.btn_run.draw(self.window)

        # ----------------------------------------------------
        # 3. Floating Overlay Pass (Draw open dropdown popups on top)
        # ----------------------------------------------------
        if not self.is_mid_run:
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
