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
import json
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from enum import Enum, auto
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("SDL_MAC_ALLOW_HIGHDPI", "1")
os.environ.setdefault("SDL_VIDEO_ALLOW_HIGHDPI", "1")
os.environ.setdefault("SDL_HINT_RENDER_SCALE_QUALITY", "0")
os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "0")

import pygame

from gui.theme import (
    WINDOW_DEFAULT_WIDTH,
    WINDOW_DEFAULT_HEIGHT,
    SIDEBAR_WIDTH,
    BG_WINDOW,
    BG_GAME_FRAME,
    BG_SIDEBAR,
    BG_CARD,
    BG_CARD_ALT,
    BG_INPUT,
    BORDER_BLACK,
    SHADOW_BLACK,
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
from gui.widgets import (
    UIButton,
    UIDropdown,
    UICard,
    UIBanner,
    UISwitchbar,
    UICheckbox,
    UITextInput,
    UISlider,
    UIModalConfirm,
)
from gui.agent_manager import (
    REPO_ROOT,
    AGENT_EVAL_DIR,
    InteractiveHumanAgent,
    create_agent,
    get_available_layouts,
    is_ppo_supported,
    get_custom_trained_agents,
    evict_trainer_cache,
)
from gui.agent_database import AgentDatabase
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

RECIPE_INFO: dict[str, dict[str, str]] = {
    "ppo": {
        "title": "PPO (Standard) (ppo)",
        "desc": "100% PPO self-play baseline",
    },
    "ppo_sp": {
        "title": "PPO-SP (Self-Play) (ppo_sp)",
        "desc": "PPO self-play training",
    },
    "ppo_bc": {
        "title": "PPO-BC (Human Model) (ppo_bc)",
        "desc": "PPO trained alongside the behavioral cloning human model (50/50 mix)",
    },
    "greedy": {
        "title": "Greedy Specialist (greedy)",
        "desc": "PPO trained against the heuristic greedy chef",
    },
    "ppo_greedy": {
        "title": "Greedy Specialist (greedy)",
        "desc": "PPO trained against the heuristic greedy chef",
    },
    "ppo_50_50": {
        "title": "PPO 50/50 Mix (SP + Greedy) (ppo_50_50)",
        "desc": "Half self-play, half greedy specialist rollouts",
    },
    "custom": {
        "title": "Custom Blend (Sliders) (custom)",
        "desc": "Custom partner allocation via sliders",
    },
}


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

        self.agent_options = self._get_full_agent_options()

        self.selected_agent_0_type = "human"
        self.selected_agent_1_type = "ppo"

        # Training Mode State
        self.train_view = "setup"  # "setup", "active", "complete", "db"
        self.train_layout = "all_layouts"
        self.train_current_layout = "all_layouts"
        self.train_recipe = "custom"
        self.train_agent_1 = "ppo"
        self.train_agent_1_pct = 50
        self.train_agent_2 = "greedy"
        self.train_agent_2_pct = 50
        self.train_agent_name = "Jeff"
        self.train_last_trained_name = "Jeff"
        self.train_process: subprocess.Popen[Any] | None = None
        self.train_progress_cur = 0
        self.train_progress_total = 500
        self.train_iters = 100
        self.train_eta_seconds = 1000
        self.train_last_poll_time = 0.0
        self.train_eta_str = "16min 40s"
        self.train_reward_mean = 0.0
        self.train_is_paused = False
        self.train_log_file: Any | None = None

        # Game simulation state
        self.mdp: OvercookedGridworld | None = None
        self.env: OvercookedEnv | None = None
        self.visualizer: StateVisualizer | None = None
        self.agents: list[Any] = []
        self.step_count = 0
        self.cumulative_score = 0.0
        self.recorded_rows: list[dict[str, Any]] = []
        self.history: list[tuple[Any, int, float, list[dict[str, Any]]]] = []
        self.status_message: str = "Ready. Select chefs & layout, then click Run!"
        self.warning_message: str | None = None

        # Background Streamlit process tracker
        self.streamlit_proc: subprocess.Popen[Any] | None = None

        # Mode State ('live' or 'matrix')
        self.active_mode = "live"

        # Matrix Benchmark State
        self.matrix_k = 3
        self.matrix_running = False
        self.matrix_done = False
        self.matrix_progress = 0.0
        self.matrix_status = "Select agents and levels, then click Run Benchmark."
        self.matrix_proc: subprocess.Popen[Any] | None = None
        self.matrix_streamlit_proc: subprocess.Popen[Any] | None = None
        self.matrix_thread: threading.Thread | None = None
        self.matrix_agents = {
            "ppo_sp": True,
            "greedy": True,
            "random": True,
        }
        self.matrix_levels = {
            "cramped_room": True,
            "asymmetric_advantages": True,
            "coordination_ring": False,
            "forced_coordination": False,
            "counter_circuit_o_1order": False,
        }

        # Build initial environment and visualizer
        self._load_environment(self.current_layout)

        # Setup GUI widgets
        self._setup_widgets()

    @property
    def is_mid_run(self) -> bool:
        """Return True if simulation is actively running or paused."""
        return self.state in (AppState.RUNNING, AppState.PAUSED)

    def _load_environment(self, layout_name: str) -> None:
        """Load or reload the Overcooked environment and calculate optimal high-res tile size.

        Supports both standard layouts (with old_dynamics) and newer custom layouts
        (such as counter_circuit, tomato layouts, etc.).
        """
        try:
            self.current_layout = layout_name
            try:
                evaluator = AgentEvaluator.from_layout_name(
                    {"layout_name": layout_name, "old_dynamics": True},
                    {"horizon": self.horizon},
                )
            except Exception:
                evaluator = AgentEvaluator.from_layout_name(
                    {"layout_name": layout_name, "old_dynamics": False},
                    {"horizon": self.horizon},
                )
            self.env = evaluator.env
            self.mdp = self.env.mdp
            self.step_count = 0
            self.cumulative_score = 0.0
            self.recorded_rows.clear()
            self.history.clear()
            self.status_message = f"Layout '{layout_name}' loaded."
            self.warning_message = None

            # Calculate optimal tile size so the visualizer generates crisp sprites
            grid_w = len(self.mdp.terrain_mtx[0])
            grid_h = len(self.mdp.terrain_mtx)
            pad = 40
            avail_w = max(100, self.game_pane_width - pad * 2)
            avail_h = max(100, self.window_height - pad * 2)
            optimal_tile_size = max(30, min(180, int(min(avail_w / grid_w, avail_h / grid_h))))

            self.visualizer = StateVisualizer(
                tile_size=optimal_tile_size,
                player_colors=["red", "blue"],  # Agent 1 = Red Hat, Agent 2 = Blue Hat
                is_rendering_hud=False,         # Clean kitchen view; sidebar shows stats cleanly
                background_color=BG_GAME_FRAME,
            )

            # Check PPO availability notification
            if self.selected_agent_1_type == "ppo" or self.selected_agent_0_type == "ppo":
                if not is_ppo_supported(layout_name):
                    self.warning_message = f"Note: PPO is not bundled for '{layout_name}'. Will use Greedy."
        except Exception as e:
            self.status_message = f"Error loading layout: {e}"

    def _get_full_agent_options(self) -> list[tuple[str, str]]:
        custom = get_custom_trained_agents()
        return [
            ("Human (Keyboard)", "human"),
            ("PPO (Pretrained RL)", "ppo"),
            ("BC (Human Model)", "bc"),
            *custom,
            ("Greedy Chef (Upstream)", "greedy"),
            ("Random Chef", "random"),
            ("Stay (Idle)", "stay"),
        ]

    def _refresh_agent_options(self) -> None:
        self.agent_options = self._get_full_agent_options()
        if hasattr(self, "dropdown_agent_0"):
            self.dropdown_agent_0.options = self.agent_options
        if hasattr(self, "dropdown_agent_1"):
            self.dropdown_agent_1.options = self.agent_options
        custom = get_custom_trained_agents()
        train_opts = [
            ("PPO (Pretrained RL)", "ppo"),
            ("BC (Human Model)", "bc"),
            *custom,
            ("Greedy Chef", "greedy"),
            ("Random Chef", "random"),
        ]
        if hasattr(self, "dropdown_train_agent_1"):
            self.dropdown_train_agent_1.options = train_opts
        if hasattr(self, "dropdown_train_agent_2"):
            self.dropdown_train_agent_2.options = train_opts

    def _setup_widgets(self) -> None:
        """Initialize sidebar switchbar, dropdowns, checkboxes, and buttons."""
        sb_x = self.game_pane_width + 16
        sb_w = SIDEBAR_WIDTH - 32
        item_h = 36

        # Top Mode Switchbar (Live Game vs Matrix vs Train)
        self.switchbar = UISwitchbar(
            rect=(sb_x, 50, sb_w, 30),
            options=[("Live Game", "live"), ("Matrix", "matrix"), ("Train", "train")],
            selected_value=self.active_mode,
            on_change=self._on_mode_changed,
        )

        # -----------------------------
        # Live Match Mode Controls
        # -----------------------------
        # Agent 1 dropdown (Red hat)
        self.dropdown_agent_0 = UIDropdown(
            rect=(sb_x, 108, sb_w, item_h),
            options=self.agent_options,
            selected_value=self.selected_agent_0_type,
            on_change=self._on_agent_0_changed,
        )

        # Agent 2 dropdown (Blue hat)
        self.dropdown_agent_1 = UIDropdown(
            rect=(sb_x, 186, sb_w, item_h),
            options=self.agent_options,
            selected_value=self.selected_agent_1_type,
            on_change=self._on_agent_1_changed,
        )

        # Layout dropdown
        layout_opts = [(name, name) for name in self.available_layouts]
        self.dropdown_layout = UIDropdown(
            rect=(sb_x, 264, sb_w, item_h),
            options=layout_opts,
            selected_value=self.current_layout,
            on_change=self._on_layout_changed,
            max_visible_items=8,
        )

        # Live FPS speed adjustment buttons
        card_y = 318
        self.btn_fps_minus = UIButton(
            rect=(sb_x + 135, card_y + 72, 24, 22),
            text="-",
            on_click=lambda: self.adjust_fps(-1),
            font_size=13,
            border_radius=0,
        )
        self.btn_fps_plus = UIButton(
            rect=(sb_x + 165, card_y + 72, 24, 22),
            text="+",
            on_click=lambda: self.adjust_fps(+1),
            font_size=13,
            border_radius=0,
        )

        # Bottom Action Buttons
        btn_y = self.window_height - 64
        half_w = (sb_w - 12) // 2
        tri_w = (sb_w - 16) // 3

        self.btn_pause = UIButton(
            rect=(sb_x, btn_y, half_w, 44),
            text="Pause",
            on_click=self.toggle_pause,
            bg_color=COLOR_PAUSE,
            hover_color=COLOR_PAUSE_HOVER,
            text_color=COLOR_PAUSE_TEXT,
            font_size=14,
        )

        self.btn_step = UIButton(
            rect=(sb_x + tri_w + 8, btn_y, tri_w, 44),
            text="Step >",
            on_click=self.step_forward,
            bg_color=(225, 230, 240),
            hover_color=(210, 218, 230),
            text_color=TEXT_PRIMARY,
            font_size=14,
            is_visible=False,
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

        # -----------------------------
        # Matrix Test Mode Controls
        # -----------------------------
        self.cb_ppo = UICheckbox(
            (sb_x, 108, sb_w, 24),
            "PPO (Self-Play)",
            checked=self.matrix_agents.get("ppo_sp", True),
            on_change=lambda v: self._toggle_matrix_agent("ppo_sp", v),
        )
        self.cb_greedy = UICheckbox(
            (sb_x, 134, sb_w, 24),
            "Greedy Chef",
            checked=self.matrix_agents.get("greedy", True),
            on_change=lambda v: self._toggle_matrix_agent("greedy", v),
        )
        self.cb_random = UICheckbox(
            (sb_x, 160, sb_w, 24),
            "Random Chef",
            checked=self.matrix_agents.get("random", True),
            on_change=lambda v: self._toggle_matrix_agent("random", v),
        )

        self.cb_cramped = UICheckbox(
            (sb_x, 212, sb_w, 24),
            "Cramped Room",
            checked=self.matrix_levels.get("cramped_room", True),
            on_change=lambda v: self._toggle_matrix_level("cramped_room", v),
        )
        self.cb_asym = UICheckbox(
            (sb_x, 238, sb_w, 24),
            "Asymmetric Advantages",
            checked=self.matrix_levels.get("asymmetric_advantages", True),
            on_change=lambda v: self._toggle_matrix_level("asymmetric_advantages", v),
        )
        self.cb_ring = UICheckbox(
            (sb_x, 264, sb_w, 24),
            "Coordination Ring",
            checked=self.matrix_levels.get("coordination_ring", False),
            on_change=lambda v: self._toggle_matrix_level("coordination_ring", v),
        )
        self.cb_forced = UICheckbox(
            (sb_x, 290, sb_w, 24),
            "Forced Coordination",
            checked=self.matrix_levels.get("forced_coordination", False),
            on_change=lambda v: self._toggle_matrix_level("forced_coordination", v),
        )
        self.cb_circuit = UICheckbox(
            (sb_x, 316, sb_w, 24),
            "Counter Circuit",
            checked=self.matrix_levels.get("counter_circuit_o_1order", False),
            on_change=lambda v: self._toggle_matrix_level("counter_circuit_o_1order", v),
        )

        self.btn_k_minus = UIButton(
            rect=(sb_x + 130, 350, 28, 26),
            text="-",
            on_click=self._dec_k,
            font_size=15,
            border_radius=0,
        )
        self.btn_k_plus = UIButton(
            rect=(sb_x + 195, 350, 28, 26),
            text="+",
            on_click=self._inc_k,
            font_size=15,
            border_radius=0,
        )

        self.btn_run_matrix = UIButton(
            rect=(sb_x, btn_y, sb_w, 44),
            text="Run Matrix Benchmark",
            on_click=self.start_matrix_benchmark,
            bg_color=COLOR_RUN,
            hover_color=COLOR_RUN_HOVER,
            text_color=COLOR_RUN_TEXT,
            font_size=14,
        )

        self.btn_matrix_view_graphs = UIButton(
            rect=(sb_x, btn_y - 54, sb_w, 44),
            text="View Graphs (Streamlit)",
            on_click=self.open_matrix_streamlit_dashboard,
            bg_color=COLOR_GRAPH,
            hover_color=COLOR_GRAPH_HOVER,
            text_color=COLOR_GRAPH_TEXT,
            font_size=13,
            is_visible=False,
        )

        # -----------------------------
        # Train Mode Controls (Images 1, 2, 3, 4)
        # -----------------------------
        self.modal_confirm = UIModalConfirm()

        # Image 4: Setup Widgets
        custom = get_custom_trained_agents()
        train_agent_opts = [
            ("PPO (Pretrained RL)", "ppo"),
            ("BC (Human Model)", "bc"),
            *custom,
            ("Greedy Chef", "greedy"),
            ("Random Chef", "random"),
        ]
        train_layout_opts = [
            ("All Maps", "all_layouts"),
            ("Cramped Room", "cramped_room"),
            ("Asymmetric Advantages", "asymmetric_advantages"),
            ("Coordination Ring", "coordination_ring"),
            ("Forced Coordination", "forced_coordination"),
            ("Counter Circuit", "counter_circuit_o_1order"),
        ]
        self.dropdown_train_layout = UIDropdown(
            rect=(sb_x, 96, sb_w, 28),
            options=train_layout_opts,
            selected_value=self.train_layout,
            on_change=self._on_train_layout_changed,
        )

        recipe_opts = [
            ("PPO (Standard)", "ppo"),
            ("PPO-SP (Self-Play)", "ppo_sp"),
            ("PPO-BC (Human Model)", "ppo_bc"),
            ("Greedy Specialist", "greedy"),
            ("PPO 50/50 Mix (SP + Greedy)", "ppo_50_50"),
            ("Custom Blend (Sliders)", "custom"),
        ]
        self.dropdown_train_recipe = UIDropdown(
            rect=(sb_x, 148, sb_w, 28),
            options=recipe_opts,
            selected_value=self.train_recipe,
            on_change=self._on_recipe_changed,
        )
        self.dropdown_train_agent_1 = UIDropdown(
            rect=(sb_x, 200, sb_w, 28),
            options=train_agent_opts,
            selected_value=self.train_agent_1,
            on_change=self._on_train_agent_1_changed,
        )
        self.slider_agent_1 = UISlider(
            rect=(sb_x, 248, sb_w, 18),
            value=self.train_agent_1_pct,
            min_val=0,
            max_val=100,
            step=10,
            on_change=self._on_slider_1_changed,
        )
        self.dropdown_train_agent_2 = UIDropdown(
            rect=(sb_x, 290, sb_w, 28),
            options=train_agent_opts,
            selected_value=self.train_agent_2,
            on_change=self._on_train_agent_2_changed,
        )
        self.slider_agent_2 = UISlider(
            rect=(sb_x, 338, sb_w, 18),
            value=self.train_agent_2_pct,
            min_val=0,
            max_val=100,
            step=10,
            on_change=self._on_slider_2_changed,
        )
        self.input_agent_name = UITextInput(
            rect=(sb_x, 380, sb_w, 28),
            text="Jeff",
            placeholder="Agent Name",
            on_change=lambda t: setattr(self, "train_agent_name", t),
        )
        self.btn_train_start = UIButton(
            rect=(sb_x, 418, sb_w, 38),
            text="TRAIN",
            on_click=self._start_training,
            bg_color=COLOR_RUN,
            hover_color=COLOR_RUN_HOVER,
            text_color=COLOR_RUN_TEXT,
            font_size=15,
        )
        self.btn_view_all_agents = UIButton(
            rect=(sb_x + sb_w - 120, self.window_height - 38, 120, 24),
            text="view all agents",
            on_click=lambda: setattr(self, "train_view", "db"),
            font_size=11,
            border_radius=0,
        )

        # Image 3: Active Training Widgets
        half_train_btn_w = (sb_w - 8) // 2
        self.btn_view_training_live = UIButton(
            rect=(sb_x, 284, sb_w, 34),
            text="View Training Live",
            on_click=self._open_live_training_terminal,
            bg_color=(235, 238, 245),
            hover_color=(220, 225, 235),
            text_color=(0, 0, 0),
            font_size=13,
        )
        self.btn_pause_training = UIButton(
            rect=(sb_x, 326, half_train_btn_w, 36),
            text="Pause training",
            on_click=self._toggle_pause_training,
            bg_color=COLOR_PAUSE,
            hover_color=COLOR_PAUSE_HOVER,
            text_color=COLOR_PAUSE_TEXT,
            font_size=13,
        )
        self.btn_stop_training = UIButton(
            rect=(sb_x + half_train_btn_w + 8, 326, half_train_btn_w, 36),
            text="Stop training",
            on_click=self._stop_training,
            bg_color=(245, 215, 215),
            hover_color=(235, 195, 195),
            text_color=(150, 20, 20),
            font_size=13,
        )

        # Image 2: Complete Widgets
        self.btn_view_jeff = UIButton(
            rect=(sb_x, 330, sb_w, 42),
            text="View Jeff",
            on_click=self._view_trained_agent_live,
            bg_color=COLOR_RUN,
            hover_color=COLOR_RUN_HOVER,
            text_color=COLOR_RUN_TEXT,
            font_size=14,
        )
        self.btn_view_agent_db = UIButton(
            rect=(sb_x, 390, sb_w, 42),
            text="View agent DB",
            on_click=lambda: setattr(self, "train_view", "db"),
            font_size=14,
        )

        # Image 1: Database Widgets
        self.input_search_agent = UITextInput(
            rect=(sb_x + 95, 114, sb_w - 95, 28),
            text="",
            placeholder="Search...",
        )
        self.btn_back_to_train = UIButton(
            rect=(sb_x, self.window_height - 38, sb_w, 28),
            text="< Back to Train",
            on_click=lambda: setattr(self, "train_view", "setup"),
            font_size=12,
        )

    def _on_train_layout_changed(self, new_layout: str) -> None:
        self.train_layout = new_layout
        if new_layout != "all_layouts" and new_layout in self.available_layouts:
            self._load_environment(new_layout)

    def _on_recipe_changed(self, new_recipe: str) -> None:
        self.train_recipe = new_recipe
        if new_recipe in ("ppo", "ppo_sp"):
            self.train_agent_1 = "ppo"
            self.dropdown_train_agent_1.selected_value = "ppo"
            self.train_agent_1_pct = 100
            self.slider_agent_1.value = 100

            self.train_agent_2 = "ppo"
            self.dropdown_train_agent_2.selected_value = "ppo"
            self.train_agent_2_pct = 0
            self.slider_agent_2.value = 0

        elif new_recipe in ("ppo_bc", "bc"):
            self.train_agent_1 = "ppo"
            self.dropdown_train_agent_1.selected_value = "ppo"
            self.train_agent_1_pct = 50
            self.slider_agent_1.value = 50

            self.train_agent_2 = "bc"
            self.dropdown_train_agent_2.selected_value = "bc"
            self.train_agent_2_pct = 50
            self.slider_agent_2.value = 50

        elif new_recipe in ("greedy", "ppo_greedy"):
            self.train_agent_1 = "ppo"
            self.dropdown_train_agent_1.selected_value = "ppo"
            self.train_agent_1_pct = 0
            self.slider_agent_1.value = 0

            self.train_agent_2 = "greedy"
            self.dropdown_train_agent_2.selected_value = "greedy"
            self.train_agent_2_pct = 100
            self.slider_agent_2.value = 100

        elif new_recipe == "ppo_50_50":
            self.train_agent_1 = "ppo"
            self.dropdown_train_agent_1.selected_value = "ppo"
            self.train_agent_1_pct = 50
            self.slider_agent_1.value = 50

            self.train_agent_2 = "greedy"
            self.dropdown_train_agent_2.selected_value = "greedy"
            self.train_agent_2_pct = 50
            self.slider_agent_2.value = 50

    def _on_train_agent_1_changed(self, new_val: str) -> None:
        self.train_agent_1 = new_val
        self._sync_recipe_from_agents()

    def _on_train_agent_2_changed(self, new_val: str) -> None:
        self.train_agent_2 = new_val
        self._sync_recipe_from_agents()

    def _on_slider_1_changed(self, new_val: int) -> None:
        self.train_agent_1_pct = new_val
        self._sync_recipe_from_agents()

    def _on_slider_2_changed(self, new_val: int) -> None:
        self.train_agent_2_pct = new_val
        self._sync_recipe_from_agents()

    def _sync_recipe_from_agents(self) -> None:
        a1 = self.train_agent_1
        a2 = self.train_agent_2
        s1 = self.train_agent_1_pct
        s2 = self.train_agent_2_pct

        if a1 == "ppo" and a2 == "ppo":
            matched = "ppo" if self.train_recipe == "ppo" else "ppo_sp"
        elif a1 == "ppo" and a2 == "bc" and s1 == 50 and s2 == 50:
            matched = "ppo_bc"
        elif (a1 == "ppo" and a2 == "greedy" and s1 == 0 and s2 == 100) or (a1 == "greedy" and a2 == "ppo" and s1 == 100 and s2 == 0):
            matched = "greedy" if self.train_recipe == "greedy" else "ppo_greedy"
        elif a1 == "ppo" and a2 == "greedy" and s1 == 50 and s2 == 50:
            matched = "ppo_50_50"
        else:
            matched = "custom"

        self.train_recipe = matched
        if hasattr(self, "dropdown_train_recipe"):
            self.dropdown_train_recipe.selected_value = matched

    def _start_training(self) -> None:
        agent_name = self.train_agent_name.strip() or "Jeff"
        self.train_last_trained_name = agent_name
        self.btn_view_jeff.text = f"View {agent_name}"

        # Determine partner type and prob
        recipe = self.train_recipe
        partner_type = "greedy"
        partner_prob = 0.5

        if recipe in ("ppo", "ppo_sp"):
            partner_type = "ppo"
            partner_prob = 0.0
        elif recipe in ("ppo_bc", "bc"):
            partner_type = "bc"
            partner_prob = 0.5
        elif recipe in ("greedy", "ppo_greedy"):
            partner_type = "greedy"
            partner_prob = 1.0
        elif recipe == "ppo_50_50":
            partner_type = "greedy"
            partner_prob = 0.5
        else:
            # Custom blend based on selected agents and sliders
            a1 = self.train_agent_1
            a2 = self.train_agent_2
            s1 = self.train_agent_1_pct
            s2 = self.train_agent_2_pct

            if a2 in ("bc", "greedy", "random"):
                partner_type = a2
                total = max(1, s1 + s2)
                partner_prob = round(s2 / total, 2)
            elif a1 in ("bc", "greedy", "random") and a2 == "ppo":
                partner_type = a1
                total = max(1, s1 + s2)
                partner_prob = round(s1 / total, 2)
            elif a1 == "ppo" and a2 == "ppo":
                partner_type = "ppo"
                partner_prob = 0.0
            else:
                partner_type = "greedy"
                partner_prob = round(s2 / max(1, s1 + s2), 2)

        out_dir = REPO_ROOT / "models" / agent_name
        venv_python = AGENT_EVAL_DIR / ".venv" / "bin" / "python"
        python_bin = str(venv_python) if venv_python.is_file() else sys.executable
        iters_per_layout = getattr(self, "train_iters", 100)
        target_layout = getattr(self, "train_layout", "all_layouts")
        cmd = [
            python_bin,
            "-u",
            "-m", "training.train_agent",
            "--recipe", recipe,
            "--partner-type", partner_type,
            "--partner-prob", str(partner_prob),
            "--layout", target_layout,
            "--iters", str(iters_per_layout),
            "--out-dir", str(out_dir),
        ]
        status_file = REPO_ROOT / "results" / "training_status.json"
        if status_file.is_file():
            try:
                status_file.unlink()
            except Exception:
                pass

        if hasattr(self, "train_log_file") and self.train_log_file is not None:
            try:
                self.train_log_file.close()
            except Exception:
                pass
            self.train_log_file = None

        log_path = REPO_ROOT / "results" / "training_live.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.train_log_file = open(log_path, "w", buffering=1, encoding="utf-8")

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"

        self.train_process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=self.train_log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        self.train_is_paused = False
        self.btn_pause_training.text = "Pause training"
        self.btn_pause_training.bg_color = COLOR_PAUSE
        self.btn_pause_training.hover_color = COLOR_PAUSE_HOVER
        self.train_view = "active"
        self.train_progress_cur = 0
        if target_layout in ("all", "all_layouts", "all_maps"):
            total_it = 5 * iters_per_layout
        else:
            total_it = iters_per_layout
        self.train_progress_total = total_it
        initial_eta = total_it * 2  # Baseline estimate (~2s per iteration with 4 rollout workers)
        self.train_eta_seconds = initial_eta
        self.train_eta_str = self._format_eta(initial_eta)
        self.train_last_poll_time = time.time()

    @staticmethod
    def _format_eta(total_seconds: int) -> str:
        s = max(0, int(total_seconds))
        hrs = s // 3600
        mins = (s % 3600) // 60
        secs = s % 60
        if hrs > 0:
            return f"{hrs}hr {mins:02d}min"
        elif mins > 0:
            return f"{mins}min {secs:02d}s"
        else:
            return f"{secs}s"

    def _open_live_training_terminal(self) -> None:
        log_file = REPO_ROOT / "results" / "training_live.log"
        if not log_file.exists():
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text("Waiting for training process output...\n", encoding="utf-8")
        escaped_path = str(log_file.resolve()).replace("'", "\\'")
        cmd_script = f"clear; echo 'Streaming live training logs ({log_file.name})...'; tail -f '{escaped_path}'"
        apple_script = f'tell application "Terminal" to do script "{cmd_script}"\ntell application "Terminal" to activate'
        try:
            subprocess.Popen(["osascript", "-e", apple_script])
        except Exception as e:
            print(f"[GUI] Failed to open live terminal: {e}")

    def _toggle_pause_training(self) -> None:
        if not self.train_process or self.train_process.poll() is not None:
            return

        if not self.train_is_paused:
            try:
                os.kill(self.train_process.pid, signal.SIGSTOP)
                self.train_is_paused = True
                self.btn_pause_training.text = "Resume training"
                self.btn_pause_training.bg_color = (210, 245, 220)
                self.btn_pause_training.hover_color = (190, 235, 205)
            except Exception as e:
                print(f"[GUI] Error pausing training: {e}")
        else:
            try:
                os.kill(self.train_process.pid, signal.SIGCONT)
                self.train_is_paused = False
                self.train_last_poll_time = time.time()
                self.btn_pause_training.text = "Pause training"
                self.btn_pause_training.bg_color = COLOR_PAUSE
                self.btn_pause_training.hover_color = COLOR_PAUSE_HOVER
            except Exception as e:
                print(f"[GUI] Error resuming training: {e}")

    def _stop_training(self) -> None:
        if self.train_process and self.train_process.poll() is None:
            if self.train_is_paused:
                try:
                    os.kill(self.train_process.pid, signal.SIGCONT)
                except Exception:
                    pass
            try:
                self.train_process.terminate()
                self.train_process.wait(timeout=2)
            except Exception:
                try:
                    self.train_process.kill()
                except Exception:
                    pass
        self.train_process = None
        self.train_is_paused = False
        if hasattr(self, "train_log_file") and self.train_log_file is not None:
            try:
                self.train_log_file.close()
            except Exception:
                pass
            self.train_log_file = None
        self.btn_pause_training.text = "Pause training"
        self.btn_pause_training.bg_color = COLOR_PAUSE
        self.btn_pause_training.hover_color = COLOR_PAUSE_HOVER
        self.train_view = "setup"

    def _poll_training_progress(self) -> None:
        if self.train_view != "active":
            return

        now = time.time()

        if self.train_is_paused:
            self.train_last_poll_time = now

        status_file = REPO_ROOT / "results" / "training_status.json"

        if status_file.is_file():
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.train_progress_cur = data.get("current_iter", 0)
                self.train_progress_total = data.get("total_iters", getattr(self, "train_progress_total", 500))
                self.train_reward_mean = data.get("reward_mean", 0.0)
                self.train_current_layout = data.get("current_layout", self.train_layout)

                eta_sec = data.get("eta_seconds", None)
                if eta_sec is not None:
                    self.train_eta_seconds = max(0, int(eta_sec))
                    self.train_last_poll_time = now

                if data.get("status") == "completed" or (self.train_process and self.train_process.poll() == 0):
                    self.train_eta_seconds = 0
                    self.train_eta_str = "0s"
                    model_dir = REPO_ROOT / "models" / self.train_last_trained_name
                    registered_layout = data.get("layout", getattr(self, "train_layout", "all_layouts"))
                    AgentDatabase.add_agent(
                        name=self.train_last_trained_name,
                        layout=registered_layout,
                        checkpoint_path=model_dir,
                        agent_1=self.train_agent_1,
                        agent_1_pct=self.train_agent_1_pct,
                        agent_2=self.train_agent_2,
                        agent_2_pct=self.train_agent_2_pct,
                        reward_mean=self.train_reward_mean,
                    )
                    self._refresh_agent_options()
                    self.train_view = "complete"
                    self.train_process = None
                    self.train_is_paused = False
                    if hasattr(self, "train_log_file") and self.train_log_file is not None:
                        try:
                            self.train_log_file.close()
                        except Exception:
                            pass
                        self.train_log_file = None
                    return
            except Exception:
                pass

        if self.train_process and self.train_process.poll() is not None:
            ret = self.train_process.poll()
            if ret != 0:
                print(f"[GUI] Training process exited unexpectedly with code {ret}.")
                self.train_process = None
                self.train_is_paused = False
                if hasattr(self, "train_log_file") and self.train_log_file is not None:
                    try:
                        self.train_log_file.close()
                    except Exception:
                        pass
                    self.train_log_file = None
                self.train_view = "setup"
                return

        # Calculate smooth countdown between iteration updates
        if hasattr(self, "train_last_poll_time") and self.train_last_poll_time > 0 and not self.train_is_paused:
            elapsed_since_poll = now - self.train_last_poll_time
            current_eta = max(0, int(self.train_eta_seconds - elapsed_since_poll))
        else:
            current_eta = self.train_eta_seconds

        self.train_eta_str = self._format_eta(current_eta)

    def _view_trained_agent_live(self) -> None:
        self._refresh_agent_options()
        self.active_mode = "live"
        self.switchbar.selected_value = "live"
        target = self.train_last_trained_name.lower()
        for label, val in self.agent_options:
            if target in label.lower() or val.endswith(target):
                self.selected_agent_0_type = val
                self.dropdown_agent_0.selected_value = val
                break
        self.selected_agent_1_type = "greedy"
        self.dropdown_agent_1.selected_value = "greedy"
        self.reset_simulation()
        self.start_or_resume()

    def _confirm_delete_agent(self, agent_id: str, agent_name: str) -> None:
        self.modal_confirm.show(
            title="Are you Sure?",
            message=f"Delete agent '{agent_name}'?",
            on_confirm=lambda: self._execute_delete_agent(agent_id),
        )

    def _execute_delete_agent(self, agent_id: str) -> None:
        AgentDatabase.delete_agent(agent_id)
        evict_trainer_cache(agent_id)
        self._refresh_agent_options()

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
        self.history.clear()

        # Save initial state in history
        import copy
        self.history.append((copy.deepcopy(self.env.state), 0, 0.0, []))

        # Instantiate agents
        a0, warn0 = create_agent(self.selected_agent_0_type, 0, self.mdp, self.current_layout)
        a1, warn1 = create_agent(self.selected_agent_1_type, 1, self.mdp, self.current_layout)
        self.agents = [a0, a1]

        if warn0 or warn1:
            self.warning_message = warn0 or warn1

        self.state = AppState.RUNNING
        self.status_message = f"Playing {self.current_layout} ({self.fps} FPS)!"
        self.btn_view_graph.is_visible = False

    def toggle_pause(self) -> None:
        """Toggle simulation pause state."""
        if self.state == AppState.RUNNING:
            self.state = AppState.PAUSED
            self.status_message = "Simulation paused. Use Arrow keys: Up/Down for FPS, Right for Step."
        elif self.state == AppState.PAUSED:
            self.state = AppState.RUNNING
            self.status_message = f"Simulation resumed ({self.fps} FPS)."

    def adjust_fps(self, delta: int) -> None:
        """Adjust simulation playback speed in FPS and update displayed telemetry."""
        new_fps = max(1, min(60, self.fps + delta))
        if new_fps != self.fps:
            self.fps = new_fps
            self.status_message = f"Speed set to {self.fps} FPS."
            print(f"[GUI] Simulation speed: {self.fps} FPS")

    def step_forward(self) -> None:
        """Advance simulation by exactly one step (useful when paused)."""
        if self.state == AppState.SETUP:
            self.start_game()
            self.state = AppState.PAUSED

        if self.state != AppState.PAUSED:
            return

        if self.step_count >= self.horizon:
            self.state = AppState.DONE
            self.status_message = f"Episode Finished! Final Score: {int(self.cumulative_score)}"
            self._save_telemetry()
            return

        self._step_simulation()
        self.status_message = f"Paused at Step {self.step_count} / {self.horizon}."

    def step_backward(self) -> None:
        """Step backward by one step if history is available (when paused)."""
        if self.state != AppState.PAUSED or not self.history:
            return

        prev_state, prev_step, prev_score, prev_rows = self.history.pop()
        self.env.state = prev_state
        self.step_count = prev_step
        self.cumulative_score = prev_score
        self.recorded_rows = list(prev_rows)
        self.status_message = f"Stepped back to Step {self.step_count} / {self.horizon}."

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

        # Open web browser once Streamlit health endpoint is ready (avoids connection error modal)
        self._open_browser_when_ready("http://localhost:8501", 8501)

    def _open_browser_when_ready(self, url: str, port: int) -> None:
        """Poll Streamlit health endpoint in a background thread and open browser once ready."""
        def _worker() -> None:
            import time
            import urllib.request
            health_url = f"http://127.0.0.1:{port}/_stcore/health"
            start_time = time.time()
            while time.time() - start_time < 8.0:
                try:
                    with urllib.request.urlopen(health_url, timeout=0.3) as resp:
                        if resp.status == 200:
                            break
                except Exception:
                    time.sleep(0.1)
            time.sleep(0.15)
            webbrowser.open(url)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _on_mode_changed(self, new_mode: str) -> None:
        self.active_mode = new_mode

    def _toggle_matrix_agent(self, key: str, val: bool) -> None:
        self.matrix_agents[key] = val

    def _toggle_matrix_level(self, key: str, val: bool) -> None:
        self.matrix_levels[key] = val

    def _inc_k(self) -> None:
        self.matrix_k = min(20, self.matrix_k + 1)

    def _dec_k(self) -> None:
        self.matrix_k = max(1, self.matrix_k - 1)

    def start_matrix_benchmark(self) -> None:
        if self.matrix_running:
            return
        selected_agents = [k for k, v in self.matrix_agents.items() if v]
        selected_levels = [k for k, v in self.matrix_levels.items() if v]
        if not selected_agents:
            self.matrix_status = "Select at least 1 agent!"
            return
        if not selected_levels:
            self.matrix_status = "Select at least 1 level!"
            return

        self.matrix_running = True
        self.matrix_done = False
        self.matrix_progress = 0.0
        self.matrix_status = "Starting benchmark engine..."
        self.btn_matrix_view_graphs.is_visible = False

        thread = threading.Thread(
            target=self._run_matrix_worker,
            args=(selected_agents, selected_levels, self.matrix_k),
            daemon=True,
        )
        self.matrix_thread = thread
        thread.start()

    def _run_matrix_worker(self, agents: list[str], levels: list[str], k: int) -> None:
        venv_python = AGENT_EVAL_DIR / ".venv" / "bin" / "python"
        python_bin = str(venv_python) if venv_python.is_file() else sys.executable
        script_path = REPO_ROOT / "evaluation" / "eval_matrix.py"

        cmd = [
            python_bin,
            "-u",
            str(script_path),
            "--agents", *agents,
            "--levels", *levels,
            "-k", str(k),
            "--horizon", "200",
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(REPO_ROOT),
            )
            self.matrix_proc = proc

            if proc.stdout:
                for line in iter(proc.stdout.readline, ""):
                    line_clean = line.strip()
                    if "[" in line_clean and "/" in line_clean and "]" in line_clean:
                        self.matrix_status = line_clean
                        try:
                            bracket = line_clean.split("[")[1].split("]")[0]
                            c, t = bracket.split("/")
                            self.matrix_progress = min(1.0, max(0.0, float(c) / float(t)))
                        except Exception:
                            pass
                proc.stdout.close()

            proc.wait()
            self.matrix_running = False
            self.matrix_done = True
            self.matrix_progress = 1.0
            self.matrix_status = "Benchmark Complete! Click 'View Graphs' below."
            self.btn_matrix_view_graphs.is_visible = True
        except Exception as e:
            self.matrix_running = False
            self.matrix_status = f"Benchmark Error: {e}"

    def open_matrix_streamlit_dashboard(self) -> None:
        """Launch Streamlit matrix dashboard on port 8502 and open browser. Only accessible when completely done."""
        if not self.matrix_done or self.matrix_running:
            return

        venv_python = AGENT_EVAL_DIR / ".venv" / "bin" / "python"
        python_bin = str(venv_python) if venv_python.is_file() else sys.executable
        dashboard_script = REPO_ROOT / "dashboard" / "coordination_matrix.py"

        if self.matrix_streamlit_proc is None or self.matrix_streamlit_proc.poll() is not None:
            cmd = [
                python_bin,
                "-m",
                "streamlit",
                "run",
                str(dashboard_script),
                "--server.headless",
                "true",
                "--server.port",
                "8502",
            ]
            try:
                self.matrix_streamlit_proc = subprocess.Popen(
                    cmd, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except Exception as e:
                print(f"[GUI] Failed to start matrix Streamlit: {e}")

        self._open_browser_when_ready("http://localhost:8502", 8502)

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
        if self.state not in (AppState.RUNNING, AppState.PAUSED) or not self.agents or self.env is None:
            return

        import copy
        self.history.append((
            copy.deepcopy(self.env.state),
            self.step_count,
            self.cumulative_score,
            list(self.recorded_rows),
        ))
        if len(self.history) > 500:
            self.history.pop(0)

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

        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                return

            # Modal confirmation dialog intercepts all events if visible
            if hasattr(self, "modal_confirm") and self.modal_confirm.is_visible:
                if self.modal_confirm.handle_event(event):
                    continue

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

                if self.active_mode == "live":
                    if event.key == pygame.K_p and self.is_mid_run:
                        self.toggle_pause()

                    elif event.key == pygame.K_r:
                        if self.is_mid_run or self.state == AppState.DONE:
                            self.reset_to_setup()
                        else:
                            self.start_game()

                    # Global FPS adjustment keys (+/- or =/_)
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.adjust_fps(+1)

                    elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                        self.adjust_fps(-1)

                    # Controls when simulation is PAUSED
                    elif self.state == AppState.PAUSED:
                        if event.key == pygame.K_UP:
                            self.adjust_fps(+1)
                        elif event.key == pygame.K_DOWN:
                            self.adjust_fps(-1)
                        elif event.key == pygame.K_RIGHT:
                            self.step_forward()
                        elif event.key == pygame.K_LEFT:
                            self.step_backward()

                        # Interactive human agent can also act turn-by-turn when paused
                        elif self.agents:
                            two_humans = sum(isinstance(a, InteractiveHumanAgent) for a in self.agents) > 1
                            consumed = False
                            for a in self.agents:
                                if isinstance(a, InteractiveHumanAgent):
                                    if a.handle_keydown(event.key, is_two_player=two_humans):
                                        consumed = True
                            if consumed:
                                self.step_forward()

                    # Controls when simulation is actively RUNNING
                    elif self.state == AppState.RUNNING:
                        # Arrow keys adjust FPS if no human agent is using them
                        has_human = any(isinstance(a, InteractiveHumanAgent) for a in self.agents)
                        if not has_human:
                            if event.key == pygame.K_UP:
                                self.adjust_fps(+1)
                            elif event.key == pygame.K_DOWN:
                                self.adjust_fps(-1)

                        # Dispatch keyboard input to interactive human agents
                        if self.agents:
                            two_humans = sum(isinstance(a, InteractiveHumanAgent) for a in self.agents) > 1
                            for a in self.agents:
                                if isinstance(a, InteractiveHumanAgent):
                                    a.handle_keydown(event.key, is_two_player=two_humans)

            # Top Switchbar always receives events
            if self.switchbar.handle_event(event):
                continue

            # ---------------------------------------------
            # Dispatch based on active mode
            # ---------------------------------------------
            if self.active_mode == "live":
                is_locked = self.is_mid_run
                self.dropdown_agent_0.is_enabled = not is_locked
                self.dropdown_agent_1.is_enabled = not is_locked
                self.dropdown_layout.is_enabled = not is_locked
                if is_locked:
                    self.dropdown_agent_0.is_open = False
                    self.dropdown_agent_1.is_open = False
                    self.dropdown_layout.is_open = False

                dropdowns = [self.dropdown_agent_0, self.dropdown_agent_1, self.dropdown_layout]

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
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        for dd in dropdowns:
                            if dd.rect.collidepoint(event.pos):
                                mode_str = "PAUSED" if self.state == AppState.PAUSED else "RUNNING"
                                self.status_message = f"Locked while {mode_str}. Click RESET to change values."

                # Buttons events in Live Mode
                self.btn_fps_minus.handle_event(event)
                self.btn_fps_plus.handle_event(event)

                if self.state == AppState.PAUSED:
                    self.btn_pause.handle_event(event)
                    self.btn_step.handle_event(event)
                    self.btn_reset.handle_event(event)
                elif self.state == AppState.RUNNING:
                    self.btn_pause.handle_event(event)
                    self.btn_reset.handle_event(event)
                elif self.state == AppState.SETUP:
                    self.btn_run.handle_event(event)
                elif self.state == AppState.DONE:
                    self.btn_view_graph.handle_event(event)
                    self.btn_reset.handle_event(event)
                    self.btn_run.handle_event(event)

            elif self.active_mode == "matrix":
                if not self.matrix_running:
                    self.cb_ppo.handle_event(event)
                    self.cb_greedy.handle_event(event)
                    self.cb_random.handle_event(event)
                    self.cb_cramped.handle_event(event)
                    self.cb_asym.handle_event(event)
                    self.cb_ring.handle_event(event)
                    self.cb_forced.handle_event(event)
                    self.cb_circuit.handle_event(event)
                    self.btn_k_minus.handle_event(event)
                    self.btn_k_plus.handle_event(event)
                    self.btn_run_matrix.handle_event(event)

                if self.matrix_done and not self.matrix_running:
                    self.btn_matrix_view_graphs.handle_event(event)

            elif self.active_mode == "train":
                self._handle_train_events(event)

    def _handle_train_events(self, event: pygame.event.Event) -> None:
        if self.train_view == "setup":
            dropdowns = [
                self.dropdown_train_layout,
                self.dropdown_train_recipe,
                self.dropdown_train_agent_1,
                self.dropdown_train_agent_2,
            ]
            open_dd = next((dd for dd in dropdowns if dd.is_open), None)
            if open_dd is not None:
                if open_dd.handle_event(event):
                    return
            for dd in dropdowns:
                if dd is not open_dd and dd.handle_event(event):
                    return

            if self.slider_agent_1.handle_event(event):
                return
            if self.slider_agent_2.handle_event(event):
                return
            if self.input_agent_name.handle_event(event):
                return
            if self.btn_train_start.handle_event(event):
                return
            if self.btn_view_all_agents.handle_event(event):
                return

        elif self.train_view == "active":
            if self.btn_view_training_live.handle_event(event):
                return
            if self.btn_pause_training.handle_event(event):
                return
            if self.btn_stop_training.handle_event(event):
                return
            if self.btn_view_all_agents.handle_event(event):
                return

        elif self.train_view == "complete":
            if self.btn_view_jeff.handle_event(event):
                return
            if self.btn_view_agent_db.handle_event(event):
                return

        elif self.train_view == "db":
            if self.input_search_agent.handle_event(event):
                return
            if self.btn_back_to_train.handle_event(event):
                return

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                sb_x = self.game_pane_width + 16
                sb_w = SIDEBAR_WIDTH - 32
                table_y = 156
                records = AgentDatabase.search(self.input_search_agent.text)
                for i, r in enumerate(records[:12]):
                    ry = table_y + 25 + i * 24
                    del_rect = pygame.Rect(sb_x + sb_w - 32, ry + 2, 26, 20)
                    if del_rect.collidepoint(event.pos):
                        self._confirm_delete_agent(r["id"], r.get("name", "agent"))
                        return

    def _draw_frame(self) -> None:
        """Render the complete application frame with full-sized environment and clean sidebar."""
        self.window.fill(BG_WINDOW)

        # ----------------------------------------------------
        # 1. Left Game Canvas Pane
        # ----------------------------------------------------
        pygame.draw.rect(
            self.window, BG_GAME_FRAME, (0, 0, self.game_pane_width, self.window_height)
        )

        if self.active_mode in ("live", "train"):
            # Live environment visualization or training layout preview
            if self.env is not None and self.visualizer is not None:
                game_surface = self.visualizer.render_state(
                    self.env.state, self.env.mdp.terrain_mtx
                )

                # Preserve exact square tile aspect ratio (no stretching or distortion)
                orig_w, orig_h = game_surface.get_size()
                pad = 24
                avail_w = max(10, self.game_pane_width - pad * 2)
                avail_h = max(10, self.window_height - pad * 2 - (50 if self.active_mode == "train" else 0))
                scale = min(avail_w / orig_w, avail_h / orig_h)
                new_w = max(1, int(orig_w * scale))
                new_h = max(1, int(orig_h * scale))

                # Nearest-neighbor scaling to keep pixel art completely crisp (avoid smoothscale blur)
                if (new_w, new_h) != (orig_w, orig_h):
                    scaled_surface = pygame.transform.scale(game_surface, (new_w, new_h))
                else:
                    scaled_surface = game_surface

                # Center the kitchen inside the left canvas frame
                offset_x = (self.game_pane_width - new_w) // 2
                offset_y = (self.window_height - new_h) // 2 + (20 if self.active_mode == "train" else 0)
                self.window.blit(scaled_surface, (offset_x, offset_y))

                if self.active_mode == "train":
                    banner_w = min(480, self.game_pane_width - 40)
                    banner_rect = pygame.Rect((self.game_pane_width - banner_w) // 2, 16, banner_w, 32)
                    if self.train_view == "active":
                        cur_l = getattr(self, "train_current_layout", self.train_layout)
                        if self.train_layout in ("all", "all_layouts", "all_maps"):
                            b_text = f"TRAINING: {self.train_last_trained_name.upper()} [ALL MAPS: {cur_l.upper()}] ({self.train_progress_cur}/{self.train_progress_total})"
                        else:
                            b_text = f"TRAINING: {self.train_last_trained_name.upper()} ({self.train_progress_cur}/{self.train_progress_total})"
                        UIBanner.draw(self.window, banner_rect, b_text, bg_color=(254, 226, 226), text_color=(220, 38, 38), font_size=12, bold=True)
                    elif self.train_view == "complete":
                        b_text = f"TRAINED: {self.train_last_trained_name.upper()} READY TO TEST"
                        UIBanner.draw(self.window, banner_rect, b_text, bg_color=(220, 252, 231), text_color=(22, 101, 52), font_size=12, bold=True)
                    else:
                        tgt = "ALL 5 BENCHMARK MAPS" if self.train_layout in ("all", "all_layouts", "all_maps") else self.train_layout.upper()
                        b_text = f"TRAINING TARGET: {tgt}"
                        UIBanner.draw(self.window, banner_rect, b_text, bg_color=BG_CARD_ALT, text_color=TEXT_PRIMARY, font_size=12, bold=True)

        elif self.active_mode == "matrix":
            # Matrix Benchmark Dashboard Canvas in RetroUI Style
            card_w = min(680, self.game_pane_width - 80)
            card_h = min(540, self.window_height - 100)
            card_x = (self.game_pane_width - card_w) // 2
            card_y = (self.window_height - card_h) // 2

            UICard.draw(self.window, (card_x, card_y, card_w, card_h), bg_color=BG_CARD, border_color=BORDER_BLACK, border_radius=0, shadow_offset=4)

            # Title
            f_title = FontManager.get_font(20, bold=True)
            t_surf = f_title.render("Matrix Benchmark Engine", True, TEXT_PRIMARY)
            self.window.blit(t_surf, (card_x + 30, card_y + 30))

            f_sub = FontManager.get_font(13, bold=False)
            sub_surf = f_sub.render("Automated n x n cross-play tournament across selected levels", True, TEXT_SECONDARY)
            self.window.blit(sub_surf, (card_x + 30, card_y + 60))

            # Selected config summary
            n_agents = sum(1 for v in self.matrix_agents.values() if v)
            m_levels = sum(1 for v in self.matrix_levels.values() if v)
            matchups = n_agents * n_agents * m_levels
            episodes = matchups * self.matrix_k

            box_y = card_y + 105
            box_w = (card_w - 60 - 30) // 3
            box_h = 75

            # Box 1: Pairings
            UICard.draw(self.window, (card_x + 30, box_y, box_w, box_h), bg_color=BG_CARD_ALT, border_color=BORDER_BLACK, border_radius=0, shadow_offset=2)
            f_box_lbl = FontManager.get_font(11, bold=True)
            f_box_val = FontManager.get_mono_font(18, bold=True)
            self.window.blit(f_box_lbl.render("AGENT PAIRINGS", True, TEXT_MUTED), (card_x + 42, box_y + 14))
            self.window.blit(f_box_val.render(f"{n_agents} x {n_agents} ({n_agents * n_agents})", True, TEXT_PRIMARY), (card_x + 42, box_y + 38))

            # Box 2: Levels
            UICard.draw(self.window, (card_x + 30 + box_w + 15, box_y, box_w, box_h), bg_color=BG_CARD_ALT, border_color=BORDER_BLACK, border_radius=0, shadow_offset=2)
            self.window.blit(f_box_lbl.render("ACTIVE LEVELS", True, TEXT_MUTED), (card_x + 30 + box_w + 27, box_y + 14))
            self.window.blit(f_box_val.render(f"{m_levels} Layouts", True, TEXT_PRIMARY), (card_x + 30 + box_w + 27, box_y + 38))

            # Box 3: Total Games
            UICard.draw(self.window, (card_x + 30 + (box_w + 15) * 2, box_y, box_w, box_h), bg_color=BG_CARD_ALT, border_color=BORDER_BLACK, border_radius=0, shadow_offset=2)
            self.window.blit(f_box_lbl.render("TOTAL GAMES", True, TEXT_MUTED), (card_x + 30 + (box_w + 15) * 2 + 12, box_y + 14))
            self.window.blit(f_box_val.render(f"{episodes} Plays", True, (45, 135, 80)), (card_x + 30 + (box_w + 15) * 2 + 12, box_y + 38))

            # Status section
            status_y = box_y + box_h + 35
            f_sec = FontManager.get_font(12, bold=True)
            self.window.blit(f_sec.render("BENCHMARK STATUS", True, TEXT_MUTED), (card_x + 30, status_y))

            f_stat = FontManager.get_font(14, bold=False)
            stat_color = (45, 135, 80) if self.matrix_done else TEXT_PRIMARY
            self.window.blit(f_stat.render(self.matrix_status, True, stat_color), (card_x + 30, status_y + 25))

            # Progress bar
            p_bar_y = status_y + 60
            p_bar_rect = pygame.Rect(card_x + 30, p_bar_y, card_w - 60, 16)
            UICard.draw(self.window, p_bar_rect, bg_color=(245, 245, 245), border_color=BORDER_BLACK, border_radius=0, shadow_offset=2)
            if self.matrix_progress > 0:
                fill_w = max(2, int((p_bar_rect.width - 4) * self.matrix_progress))
                fill_rect = pygame.Rect(p_bar_rect.x + 2, p_bar_rect.y + 2, fill_w, p_bar_rect.height - 4)
                p_color = COLOR_DONE_BADGE if self.matrix_done else (100, 105, 115)
                pygame.draw.rect(self.window, p_color, fill_rect)

            pct_font = FontManager.get_mono_font(11, bold=True)
            pct_surf = pct_font.render(f"{int(self.matrix_progress * 100)}%", True, TEXT_MUTED)
            self.window.blit(pct_surf, (p_bar_rect.right - pct_surf.get_width(), p_bar_y - 18))

            # Bottom guidance hint
            hint_y = p_bar_y + 35
            f_hint = FontManager.get_font(12, bold=False)
            if self.matrix_done:
                hint_txt = "Benchmark complete! Click 'View Graphs' in the right sidebar to open Streamlit heatmaps."
                hint_surf = f_hint.render(hint_txt, True, (45, 135, 80))
            elif self.matrix_running:
                hint_txt = "Simulating games headless in memory... Windows stay smooth and responsive."
                hint_surf = f_hint.render(hint_txt, True, TEXT_SECONDARY)
            else:
                hint_txt = "Configure agents and layouts in the right sidebar, then click 'Run Matrix Benchmark'."
                hint_surf = f_hint.render(hint_txt, True, TEXT_MUTED)
            self.window.blit(hint_surf, (card_x + 30, hint_y))

        # Vertical divider line separating game pane and sidebar
        pygame.draw.line(
            self.window,
            BORDER_BLACK,
            (self.game_pane_width, 0),
            (self.game_pane_width, self.window_height),
            width=2,
        )

        # ----------------------------------------------------
        # 2. Right Control Sidebar (RetroUI Style)
        # ----------------------------------------------------
        sb_start_x = self.game_pane_width
        sb_x = self.game_pane_width + 16
        sb_w = SIDEBAR_WIDTH - 32

        # Sidebar background fill
        pygame.draw.rect(
            self.window, BG_SIDEBAR, (sb_start_x, 0, SIDEBAR_WIDTH, self.window_height)
        )

        # Header Title
        font_title = FontManager.get_font(18, bold=True)
        title_surf = font_title.render("OVERCOOKED-AI", True, TEXT_PRIMARY)
        self.window.blit(title_surf, (sb_x, 14))
        sub_font = FontManager.get_font(11, bold=False)
        sub_surf = sub_font.render("Retro Runner", True, TEXT_MUTED)
        self.window.blit(sub_surf, (sb_x + title_surf.get_width() + 10, 20))

        # Draw Top Switchbar (Live Match vs Matrix Test)
        self.switchbar.draw(self.window)

        # ----------------------------------------------------
        # Mode Specific Sidebar Content
        # ----------------------------------------------------
        btn_y = self.window_height - 64
        half_w = (sb_w - 12) // 2

        if self.active_mode == "live":
            # Chef 0 Section
            lbl_font = FontManager.get_font(12, bold=True)
            a0_lbl = lbl_font.render("Chef 0", True, TEXT_PRIMARY)
            self.window.blit(a0_lbl, (sb_x, 88))
            a0_badge_x = sb_x + a0_lbl.get_width() + 10
            UIBanner.draw(self.window, (a0_badge_x, 85, 78, 20), "RED HAT", COLOR_AGENT_RED, font_size=10, bold=True)
            self.dropdown_agent_0.draw(self.window)

            # Chef 1 Section
            a1_lbl = lbl_font.render("Chef 1", True, TEXT_PRIMARY)
            self.window.blit(a1_lbl, (sb_x, 166))
            a1_badge_x = sb_x + a1_lbl.get_width() + 10
            UIBanner.draw(self.window, (a1_badge_x, 163, 78, 20), "BLUE HAT", COLOR_AGENT_BLUE, font_size=10, bold=True)
            self.dropdown_agent_1.draw(self.window)

            # Layout Section
            lay_lbl = lbl_font.render("Layout / Map", True, TEXT_PRIMARY)
            self.window.blit(lay_lbl, (sb_x, 244))
            self.dropdown_layout.draw(self.window)

            # Telemetry & Status Card
            card_y = 318
            card_h = 125
            UICard.draw(self.window, (sb_x, card_y, sb_w, card_h), bg_color=BG_CARD, border_color=BORDER_BLACK, border_radius=0, shadow_offset=3)

            card_title_font = FontManager.get_font(11, bold=True)
            card_t = card_title_font.render("RUN TELEMETRY & STATS", True, TEXT_MUTED)
            self.window.blit(card_t, (sb_x + 12, card_y + 10))

            metric_font = FontManager.get_mono_font(12, bold=False)
            m_step = metric_font.render(f"Step  : {self.step_count} / {self.horizon}", True, TEXT_PRIMARY)
            m_score = metric_font.render(f"Score : {int(self.cumulative_score)}", True, (45, 135, 80))
            m_speed = metric_font.render(f"Speed : {self.fps} FPS", True, TEXT_SECONDARY)
            self.window.blit(m_step, (sb_x + 12, card_y + 32))
            self.window.blit(m_score, (sb_x + 12, card_y + 54))
            self.window.blit(m_speed, (sb_x + 12, card_y + 76))
            self.btn_fps_minus.rect = pygame.Rect(sb_x + 135, card_y + 72, 24, 22)
            self.btn_fps_plus.rect = pygame.Rect(sb_x + 165, card_y + 72, 24, 22)
            self.btn_fps_minus.draw(self.window)
            self.btn_fps_plus.draw(self.window)

            # Progress bar
            prog_pct = min(1.0, self.step_count / self.horizon) if self.horizon > 0 else 0
            pbar_rect = pygame.Rect(sb_x + 12, card_y + 102, sb_w - 24, 10)
            UICard.draw(self.window, pbar_rect, bg_color=(245, 245, 245), border_color=BORDER_BLACK, border_radius=0, shadow_offset=1)
            if prog_pct > 0:
                fill_rect = pygame.Rect(sb_x + 13, card_y + 103, int((sb_w - 26) * prog_pct), 8)
                p_color = COLOR_DONE_BADGE if self.state == AppState.DONE else (100, 105, 115)
                pygame.draw.rect(self.window, p_color, fill_rect)

            # Controls & Help Card
            help_y = card_y + card_h + 10
            help_h = 105
            UICard.draw(self.window, (sb_x, help_y, sb_w, help_h), bg_color=BG_CARD_ALT, border_color=BORDER_BLACK, border_radius=0, shadow_offset=2)
            help_t = card_title_font.render("KEYBOARD CONTROLS", True, TEXT_MUTED)
            self.window.blit(help_t, (sb_x + 12, help_y + 8))

            h_f = FontManager.get_font(11, bold=False)
            h1 = h_f.render("- Up / Down     : Adjust FPS speed", True, TEXT_SECONDARY)
            h2 = h_f.render("- Right / Left  : Step forward / back (Paused)", True, TEXT_SECONDARY)
            h3 = h_f.render("- WASD / Arrows : Move Chef 0 (Red)", True, TEXT_MUTED)
            h4 = h_f.render("- Space / Enter : Interact  [P] Pause  [R] Reset", True, TEXT_MUTED)
            self.window.blit(h1, (sb_x + 12, help_y + 26))
            self.window.blit(h2, (sb_x + 12, help_y + 44))
            self.window.blit(h3, (sb_x + 12, help_y + 62))
            self.window.blit(h4, (sb_x + 12, help_y + 80))

            if self.warning_message:
                warn_font = FontManager.get_font(11, bold=False)
                warn_surf = warn_font.render(self.warning_message, True, COLOR_PAUSE)
                self.window.blit(warn_surf, (sb_x, help_y + help_h + 8))

            # Live Action Buttons
            if self.state == AppState.SETUP:
                self.btn_step.is_visible = False
                self.btn_run.rect = pygame.Rect(sb_x, btn_y, sb_w, 44)
                self.btn_run.text = "Run"
                self.btn_run.bg_color = COLOR_RUN
                self.btn_run.hover_color = COLOR_RUN_HOVER
                self.btn_run.text_color = COLOR_RUN_TEXT
                self.btn_run.is_visible = True
                self.btn_run.draw(self.window)

            elif self.state == AppState.PAUSED:
                tri_w = (sb_w - 16) // 3
                self.btn_pause.rect = pygame.Rect(sb_x, btn_y, tri_w, 44)
                self.btn_pause.text = "Resume"
                self.btn_pause.is_visible = True
                self.btn_pause.draw(self.window)

                self.btn_step.rect = pygame.Rect(sb_x + tri_w + 8, btn_y, tri_w, 44)
                self.btn_step.is_visible = True
                self.btn_step.draw(self.window)

                self.btn_reset.rect = pygame.Rect(sb_x + (tri_w + 8) * 2, btn_y, tri_w, 44)
                self.btn_reset.text = "Reset"
                self.btn_reset.is_visible = True
                self.btn_reset.draw(self.window)

            elif self.state == AppState.RUNNING:
                self.btn_step.is_visible = False
                self.btn_pause.rect = pygame.Rect(sb_x, btn_y, half_w, 44)
                self.btn_pause.text = "Pause"
                self.btn_pause.is_visible = True
                self.btn_pause.draw(self.window)

                self.btn_reset.rect = pygame.Rect(sb_x + half_w + 12, btn_y, half_w, 44)
                self.btn_reset.text = "Reset"
                self.btn_reset.is_visible = True
                self.btn_reset.draw(self.window)

            elif self.state == AppState.DONE:
                self.btn_step.is_visible = False
                self.btn_view_graph.rect = pygame.Rect(sb_x, btn_y - 54, sb_w, 44)
                self.btn_view_graph.text = "View Graph (Streamlit)"
                self.btn_view_graph.is_visible = True
                self.btn_view_graph.draw(self.window)

                self.btn_reset.rect = pygame.Rect(sb_x, btn_y, half_w, 44)
                self.btn_reset.text = "Reset"
                self.btn_reset.is_visible = True
                self.btn_reset.draw(self.window)

                self.btn_run.rect = pygame.Rect(sb_x + half_w + 12, btn_y, half_w, 44)
                self.btn_run.text = "Run Again"
                self.btn_run.is_visible = True
                self.btn_run.draw(self.window)

        elif self.active_mode == "matrix":
            # ---------------------------------------------
            # Matrix Mode Sidebar
            # ---------------------------------------------
            f_sec = FontManager.get_font(12, bold=True)

            # Agents Section
            self.window.blit(f_sec.render("AGENTS TO BENCHMARK", True, TEXT_PRIMARY), (sb_x, 88))
            self.cb_ppo.draw(self.window)
            self.cb_greedy.draw(self.window)
            self.cb_random.draw(self.window)

            # Levels Section
            self.window.blit(f_sec.render("LEVELS TO INCLUDE", True, TEXT_PRIMARY), (sb_x, 188))
            self.cb_cramped.draw(self.window)
            self.cb_asym.draw(self.window)
            self.cb_ring.draw(self.window)
            self.cb_forced.draw(self.window)
            self.cb_circuit.draw(self.window)

            # K Stepper Section
            self.window.blit(f_sec.render("TRIALS / PAIR (K)", True, TEXT_PRIMARY), (sb_x, 355))
            self.btn_k_minus.draw(self.window)
            f_k = FontManager.get_mono_font(15, bold=True)
            k_surf = f_k.render(str(self.matrix_k), True, TEXT_PRIMARY)
            k_x = (sb_x + 130 + 28) + (195 - (130 + 28) - k_surf.get_width()) // 2
            self.window.blit(k_surf, (k_x, 354))
            self.btn_k_plus.draw(self.window)

            # Matrix Benchmark Buttons
            if self.matrix_running:
                # Disabled running button while benchmark is actively executing
                self.btn_matrix_view_graphs.is_visible = False
                self.btn_run_matrix.text = "Simulating..."
                self.btn_run_matrix.bg_color = (210, 214, 220)
                self.btn_run_matrix.hover_color = (210, 214, 220)
                self.btn_run_matrix.rect = pygame.Rect(sb_x, btn_y, sb_w, 44)
                self.btn_run_matrix.draw(self.window)

            elif self.matrix_done:
                # View Graphs button ONLY appears after the run is completely done
                self.btn_matrix_view_graphs.is_visible = True
                self.btn_matrix_view_graphs.rect = pygame.Rect(sb_x, btn_y - 54, sb_w, 44)
                self.btn_matrix_view_graphs.text = "View Graphs (Streamlit)"
                self.btn_matrix_view_graphs.draw(self.window)

                # Run Again button
                self.btn_run_matrix.text = "Run Again"
                self.btn_run_matrix.bg_color = COLOR_RUN
                self.btn_run_matrix.hover_color = COLOR_RUN_HOVER
                self.btn_run_matrix.rect = pygame.Rect(sb_x, btn_y, sb_w, 44)
                self.btn_run_matrix.draw(self.window)

            else:
                # Initial setup state before running: View Graphs is hidden
                self.btn_matrix_view_graphs.is_visible = False
                self.btn_run_matrix.text = "Run Matrix Benchmark"
                self.btn_run_matrix.bg_color = COLOR_RUN
                self.btn_run_matrix.hover_color = COLOR_RUN_HOVER
                self.btn_run_matrix.rect = pygame.Rect(sb_x, btn_y, sb_w, 44)
                self.btn_run_matrix.draw(self.window)

        elif self.active_mode == "train":
            self._draw_train_mode(sb_x, sb_w)

        # ----------------------------------------------------
        # 3. Floating Overlay Pass (Draw open dropdown popups on top in Live Mode)
        # ----------------------------------------------------
        if self.active_mode == "live" and not self.is_mid_run:
            for dd in (self.dropdown_layout, self.dropdown_agent_1, self.dropdown_agent_0):
                dd.draw_overlay(self.window)
        elif self.active_mode == "train" and self.train_view == "setup":
            for dd in (self.dropdown_train_agent_2, self.dropdown_train_agent_1, self.dropdown_train_recipe):
                dd.draw_overlay(self.window)

        # Modal Confirmation (Draws on top of everything when visible)
        if hasattr(self, "modal_confirm") and self.modal_confirm.is_visible:
            self.modal_confirm.draw(self.window)

        pygame.display.flip()

    def get_recipe_info(self) -> dict[str, str]:
        """Return the dictionary containing title and description for the current recipe."""
        return RECIPE_INFO.get(
            self.train_recipe,
            {
                "title": f"Recipe ({self.train_recipe})",
                "desc": "Custom partner allocation via sliders",
            },
        )

    def _draw_recipe_info_card(self, sb_x: int, sb_w: int, card_y: int) -> None:
        """Render recipe explanation card at the bottom of the training sidebar."""
        info = self.get_recipe_info()
        title_text = info["title"]
        desc_text = info["desc"]

        card_h = 88
        card_rect = pygame.Rect(sb_x, card_y, sb_w, card_h)
        UICard.draw(
            self.window,
            card_rect,
            bg_color=(255, 255, 255),
            border_color=BORDER_BLACK,
            shadow_offset=2,
        )

        f_tag = FontManager.get_font(9, bold=True)
        tag_surf = f_tag.render("RECIPE INFO:", True, TEXT_MUTED)
        self.window.blit(tag_surf, (sb_x + 10, card_y + 8))

        f_title = FontManager.get_font(11, bold=True)
        title_surf = f_title.render(title_text, True, TEXT_PRIMARY)
        self.window.blit(title_surf, (sb_x + 10, card_y + 24))

        f_desc = FontManager.get_font(11, bold=False)
        max_text_w = sb_w - 20
        words = desc_text.split(" ")
        lines: list[str] = []
        curr_line: list[str] = []
        for w in words:
            test_line = " ".join(curr_line + [w])
            if f_desc.size(test_line)[0] <= max_text_w:
                curr_line.append(w)
            else:
                if curr_line:
                    lines.append(" ".join(curr_line))
                curr_line = [w]
        if curr_line:
            lines.append(" ".join(curr_line))

        line_y = card_y + 44
        for line in lines[:2]:
            l_surf = f_desc.render(line, True, TEXT_SECONDARY)
            self.window.blit(l_surf, (sb_x + 10, line_y))
            line_y += 16

    def _draw_train_mode(self, sb_x: int, sb_w: int) -> None:
        """Render the 4 Train Mode views matching wireframe drawings."""
        f_sec = FontManager.get_font(12, bold=True)
        f_lbl = FontManager.get_font(11, bold=False)
        f_title = FontManager.get_font(14, bold=True)

        if self.train_view == "setup":
            # Training layout selector
            self.window.blit(f_sec.render("training layout / map:", True, TEXT_PRIMARY), (sb_x, 80))
            self.dropdown_train_layout.draw(self.window)

            # Recipe Selector
            self.window.blit(f_sec.render("agent type / recipe:", True, TEXT_PRIMARY), (sb_x, 132))
            self.dropdown_train_recipe.draw(self.window)

            # Select Agent 1
            self.window.blit(f_sec.render("select agent 1:", True, TEXT_PRIMARY), (sb_x, 184))
            self.dropdown_train_agent_1.draw(self.window)

            # Slider 1
            s1_lbl = f"*Drag to the desired amount: {self.train_agent_1_pct}%"
            self.window.blit(f_lbl.render(s1_lbl, True, TEXT_MUTED), (sb_x, 234))
            self.slider_agent_1.draw(self.window)

            # Select Agent 2
            self.window.blit(f_sec.render("select agent 2:", True, TEXT_PRIMARY), (sb_x, 274))
            self.dropdown_train_agent_2.draw(self.window)

            # Slider 2
            s2_lbl = f"*Drag to the desired amount: {self.train_agent_2_pct}%"
            self.window.blit(f_lbl.render(s2_lbl, True, TEXT_MUTED), (sb_x, 324))
            self.slider_agent_2.draw(self.window)

            # Name Agent
            self.window.blit(f_sec.render("name agent:", True, TEXT_PRIMARY), (sb_x, 364))
            self.input_agent_name.draw(self.window)

            # Train Button
            self.btn_train_start.draw(self.window)

            # Recipe Info Card on the bottom of the menu
            card_y = min(464, self.window_height - 38 - 88 - 8)
            self._draw_recipe_info_card(sb_x, sb_w, card_y)

            # View All Agents link button
            self.btn_view_all_agents.draw(self.window)

        elif self.train_view == "active":
            # Image 3: Active Training View
            card_rect = pygame.Rect(sb_x, 100, sb_w, 60)
            UICard.draw(self.window, card_rect, bg_color=(255, 255, 255), border_color=BORDER_BLACK, shadow_offset=2)
            title_text = "Training (Paused) . . ." if self.train_is_paused else "Training . . ."
            t_surf = f_title.render(title_text, True, TEXT_PRIMARY)
            self.window.blit(t_surf, t_surf.get_rect(center=(card_rect.centerx, card_rect.centery - 10)))
            f_sub = FontManager.get_font(10, bold=False)
            recipe_display = self.train_recipe.upper().replace("_", " ")
            cur_l = getattr(self, "train_current_layout", self.train_layout)
            if self.train_layout in ("all", "all_layouts", "all_maps"):
                layout_display = f"ALL MAPS ({cur_l.upper()})"
            else:
                layout_display = cur_l.upper()
            r_surf = f_sub.render(f"Recipe: {recipe_display} | Map: {layout_display}", True, TEXT_MUTED)
            self.window.blit(r_surf, r_surf.get_rect(center=(card_rect.centerx, card_rect.centery + 14)))

            fraction = max(0.0, min(1.0, self.train_progress_cur / max(1, self.train_progress_total)))
            pct = int(round(fraction * 100))
            pb_lbl = f"Progress Bar: {self.train_progress_cur}/{self.train_progress_total} ({pct}%)"
            self.window.blit(f_sec.render(pb_lbl, True, TEXT_PRIMARY), (sb_x, 172))

            pb_rect = pygame.Rect(sb_x, 194, sb_w, 32)
            pygame.draw.rect(self.window, SHADOW_BLACK, pb_rect.move(2, 2))
            pygame.draw.rect(self.window, (255, 255, 255), pb_rect)

            fill_w = int(pb_rect.width * fraction)
            if fill_w > 0:
                fill_r = pygame.Rect(pb_rect.x, pb_rect.y, fill_w, pb_rect.height)
                pygame.draw.rect(self.window, (248, 113, 113), fill_r)
                for x in range(pb_rect.x + 4, pb_rect.x + fill_w, 10):
                    pygame.draw.line(self.window, (220, 38, 38), (x, pb_rect.bottom - 2), (min(pb_rect.x + fill_w, x + 8), pb_rect.y + 2), width=2)

            pygame.draw.rect(self.window, BORDER_BLACK, pb_rect, width=2)

            eta_lbl = f"Estimated Time: {self.train_eta_str}"
            self.window.blit(f_lbl.render(eta_lbl, True, TEXT_PRIMARY), (sb_x, 236))

            rew_lbl = f"Mean Reward: {self.train_reward_mean:+.2f}"
            self.window.blit(f_lbl.render(rew_lbl, True, TEXT_PRIMARY), (sb_x, 258))

            self.btn_view_training_live.draw(self.window)
            self.btn_pause_training.draw(self.window)
            self.btn_stop_training.draw(self.window)

            # Recipe Info Card on the bottom of the active menu
            active_info_y = min(374, self.window_height - 38 - 88 - 8)
            self._draw_recipe_info_card(sb_x, sb_w, active_info_y)

            self.btn_view_all_agents.draw(self.window)

        elif self.train_view == "complete":
            # Image 2: Training Complete View
            card_rect = pygame.Rect(sb_x, 100, sb_w, 60)
            UICard.draw(self.window, card_rect, bg_color=(255, 255, 255), border_color=BORDER_BLACK, shadow_offset=2)
            t_surf = f_title.render("Training Complete!", True, (22, 163, 74))
            self.window.blit(t_surf, t_surf.get_rect(center=card_rect.center))

            pb_lbl = f"Progress Bar: {self.train_progress_total}/{self.train_progress_total} (100%)"
            self.window.blit(f_sec.render(pb_lbl, True, TEXT_PRIMARY), (sb_x, 180))

            pb_rect = pygame.Rect(sb_x, 204, sb_w, 36)
            pygame.draw.rect(self.window, SHADOW_BLACK, pb_rect.move(2, 2))
            pygame.draw.rect(self.window, (248, 113, 113), pb_rect)
            for x in range(pb_rect.x + 4, pb_rect.right, 10):
                pygame.draw.line(self.window, (220, 38, 38), (x, pb_rect.bottom - 2), (min(pb_rect.right, x + 8), pb_rect.y + 2), width=2)
            pygame.draw.rect(self.window, BORDER_BLACK, pb_rect, width=2)

            self.window.blit(f_lbl.render("Estimated Time: 0s", True, TEXT_PRIMARY), (sb_x, 252))

            self.btn_view_jeff.draw(self.window)
            self.btn_view_agent_db.draw(self.window)

            # Recipe Info Card on the bottom of the complete menu
            complete_info_y = min(448, self.window_height - 38 - 88 - 8)
            self._draw_recipe_info_card(sb_x, sb_w, complete_info_y)

        elif self.train_view == "db":
            # Image 1: Agent Database View
            banner_rect = pygame.Rect(sb_x, 88, sb_w, 28)
            UIBanner.draw(self.window, banner_rect, "Agent DataBase", bg_color=(243, 244, 246), text_color=TEXT_PRIMARY, font_size=13)

            self.window.blit(f_sec.render("Search agent:", True, TEXT_PRIMARY), (sb_x, 126))
            self.input_search_agent.rect.topleft = (sb_x + 95, 122)
            self.input_search_agent.draw(self.window)

            table_y = 156
            table_h = self.window_height - table_y - 50
            table_rect = pygame.Rect(sb_x, table_y, sb_w, table_h)
            pygame.draw.rect(self.window, SHADOW_BLACK, table_rect.move(2, 2))
            pygame.draw.rect(self.window, (255, 255, 255), table_rect)
            pygame.draw.rect(self.window, BORDER_BLACK, table_rect, width=2)

            header_rect = pygame.Rect(sb_x, table_y, sb_w, 24)
            pygame.draw.rect(self.window, (243, 244, 246), header_rect)
            pygame.draw.line(self.window, BORDER_BLACK, (sb_x, table_y + 24), (sb_x + sb_w, table_y + 24), width=1)

            col_x1 = sb_x + 6
            col_x2 = sb_x + 72
            col_x3 = sb_x + 140
            col_x4 = sb_x + sb_w - 34

            f_th = FontManager.get_font(10, bold=True)
            self.window.blit(f_th.render("Name", True, TEXT_PRIMARY), (col_x1, table_y + 6))
            self.window.blit(f_th.render("date trained", True, TEXT_PRIMARY), (col_x2, table_y + 6))
            self.window.blit(f_th.render("Time", True, TEXT_PRIMARY), (col_x3, table_y + 6))
            self.window.blit(f_th.render("Delete", True, TEXT_PRIMARY), (col_x4, table_y + 6))

            pygame.draw.line(self.window, (200, 205, 215), (col_x2 - 4, table_y), (col_x2 - 4, table_y + table_h), width=1)
            pygame.draw.line(self.window, (200, 205, 215), (col_x3 - 4, table_y), (col_x3 - 4, table_y + table_h), width=1)
            pygame.draw.line(self.window, (200, 205, 215), (col_x4 - 4, table_y), (col_x4 - 4, table_y + table_h), width=1)

            records = AgentDatabase.search(self.input_search_agent.text)
            f_td = FontManager.get_font(10)
            max_rows = (table_h - 26) // 24

            if not records:
                empty_lbl = f_lbl.render("No agents trained yet.", True, TEXT_MUTED)
                self.window.blit(empty_lbl, (sb_x + 20, table_y + 36))
            else:
                for i, r in enumerate(records[:max_rows]):
                    ry = table_y + 25 + i * 24
                    pygame.draw.line(self.window, (230, 233, 238), (sb_x, ry + 23), (sb_x + sb_w, ry + 23), width=1)

                    name_txt = r.get("name", "agent")
                    if len(name_txt) > 10:
                        name_txt = name_txt[:9] + ".."
                    self.window.blit(f_td.render(name_txt, True, TEXT_PRIMARY), (col_x1, ry + 5))
                    self.window.blit(f_td.render(r.get("date_trained", ""), True, TEXT_SECONDARY), (col_x2, ry + 5))
                    self.window.blit(f_td.render(r.get("time_trained", ""), True, TEXT_SECONDARY), (col_x3, ry + 5))

                    del_btn_rect = pygame.Rect(sb_x + sb_w - 32, ry + 2, 26, 20)
                    mouse_pos = pygame.mouse.get_pos()
                    is_hov = del_btn_rect.collidepoint(mouse_pos)
                    btn_col = (254, 226, 226) if is_hov else (243, 244, 246)
                    pygame.draw.rect(self.window, btn_col, del_btn_rect)
                    pygame.draw.rect(self.window, BORDER_BLACK, del_btn_rect, width=1)
                    f_x = FontManager.get_font(11, bold=True)
                    x_surf = f_x.render("X", True, (220, 38, 38))
                    self.window.blit(x_surf, x_surf.get_rect(center=del_btn_rect.center))

            self.btn_back_to_train.draw(self.window)

    def run(self) -> None:
        """Main application execution loop."""
        print("[GUI] Overcooked-AI Interactive Runner is ready.")
        sim_accumulator = 0.0
        try:
            while self.running:
                dt = min(self.clock.tick(60) / 1000.0, 0.2)
                self._poll_training_progress()
                self.handle_events()
                if self.state == AppState.RUNNING:
                    sim_accumulator += dt
                    sim_step_interval = 1.0 / max(1, self.fps)
                    while sim_accumulator >= sim_step_interval:
                        self._step_simulation()
                        sim_accumulator -= sim_step_interval
                        if self.state != AppState.RUNNING:
                            sim_accumulator = 0.0
                            break
                else:
                    sim_accumulator = 0.0

                self._draw_frame()
        finally:
            pygame.display.quit()
            pygame.quit()
            if self.streamlit_proc is not None:
                try:
                    self.streamlit_proc.terminate()
                except Exception:
                    pass
            if self.train_process is not None and self.train_process.poll() is None:
                try:
                    if getattr(self, "train_is_paused", False):
                        os.kill(self.train_process.pid, signal.SIGCONT)
                    self.train_process.terminate()
                except Exception:
                    pass
            if hasattr(self, "train_log_file") and self.train_log_file is not None:
                try:
                    self.train_log_file.close()
                except Exception:
                    pass
            print("[GUI] Closed cleanly.")
