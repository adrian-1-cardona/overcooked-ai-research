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
import threading
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
from gui.widgets import UIButton, UIDropdown, UICard, UIBanner, UISwitchbar, UICheckbox
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
            pad = 40
            avail_w = max(100, self.game_pane_width - pad * 2)
            avail_h = max(100, self.window_height - pad * 2)
            optimal_tile_size = max(40, min(180, int(min(avail_w / grid_w, avail_h / grid_h))))

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

    def _setup_widgets(self) -> None:
        """Initialize sidebar switchbar, dropdowns, checkboxes, and buttons."""
        sb_x = self.game_pane_width + 16
        sb_w = SIDEBAR_WIDTH - 32
        item_h = 36

        # Top Mode Switchbar (Live Match vs Matrix Test)
        self.switchbar = UISwitchbar(
            rect=(sb_x, 50, sb_w, 30),
            options=[("Live Match", "live"), ("Matrix Test", "matrix")],
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

        # -----------------------------
        # Matrix Test Mode Controls
        # -----------------------------
        self.cb_ppo = UICheckbox(
            (sb_x, 108, sb_w, 22),
            "PPO (Self-Play)",
            checked=self.matrix_agents.get("ppo_sp", True),
            on_change=lambda v: self._toggle_matrix_agent("ppo_sp", v),
        )
        self.cb_greedy = UICheckbox(
            (sb_x, 132, sb_w, 22),
            "Greedy Agent",
            checked=self.matrix_agents.get("greedy", True),
            on_change=lambda v: self._toggle_matrix_agent("greedy", v),
        )
        self.cb_random = UICheckbox(
            (sb_x, 156, sb_w, 22),
            "Random Agent",
            checked=self.matrix_agents.get("random", True),
            on_change=lambda v: self._toggle_matrix_agent("random", v),
        )

        self.cb_cramped = UICheckbox(
            (sb_x, 210, sb_w, 20),
            "Cramped Room",
            checked=self.matrix_levels.get("cramped_room", True),
            on_change=lambda v: self._toggle_matrix_level("cramped_room", v),
        )
        self.cb_asym = UICheckbox(
            (sb_x, 232, sb_w, 20),
            "Asymmetric Advantages",
            checked=self.matrix_levels.get("asymmetric_advantages", True),
            on_change=lambda v: self._toggle_matrix_level("asymmetric_advantages", v),
        )
        self.cb_ring = UICheckbox(
            (sb_x, 254, sb_w, 20),
            "Coordination Ring",
            checked=self.matrix_levels.get("coordination_ring", False),
            on_change=lambda v: self._toggle_matrix_level("coordination_ring", v),
        )
        self.cb_forced = UICheckbox(
            (sb_x, 276, sb_w, 20),
            "Forced Coordination",
            checked=self.matrix_levels.get("forced_coordination", False),
            on_change=lambda v: self._toggle_matrix_level("forced_coordination", v),
        )
        self.cb_circuit = UICheckbox(
            (sb_x, 298, sb_w, 20),
            "Counter Circuit",
            checked=self.matrix_levels.get("counter_circuit_o_1order", False),
            on_change=lambda v: self._toggle_matrix_level("counter_circuit_o_1order", v),
        )

        self.btn_k_minus = UIButton(
            rect=(sb_x + 130, 328, 28, 26),
            text="-",
            on_click=self._dec_k,
            font_size=15,
            border_radius=0,
        )
        self.btn_k_plus = UIButton(
            rect=(sb_x + 195, 328, 28, 26),
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
        )

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
                            self.matrix_progress = min(1.0, max(0.0, int(c) / int(t)))
                        except Exception:
                            pass
                proc.stdout.close()

            proc.wait()
            self.matrix_running = False
            self.matrix_done = True
            self.matrix_progress = 1.0
            self.matrix_status = "Benchmark Complete! Click 'View Graphs' below."
        except Exception as e:
            self.matrix_running = False
            self.matrix_status = f"Benchmark Error: {e}"

    def open_matrix_streamlit_dashboard(self) -> None:
        """Launch Streamlit matrix dashboard on port 8502 and open browser."""
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

                if self.active_mode == "live":
                    if event.key == pygame.K_p and self.is_mid_run:
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
                if self.is_mid_run:
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

                if self.matrix_done:
                    self.btn_matrix_view_graphs.handle_event(event)

    def _draw_frame(self) -> None:
        """Render the complete application frame with full-sized environment and clean sidebar."""
        self.window.fill(BG_WINDOW)

        # ----------------------------------------------------
        # 1. Left Game Canvas Pane
        # ----------------------------------------------------
        pygame.draw.rect(
            self.window, BG_GAME_FRAME, (0, 0, self.game_pane_width, self.window_height)
        )

        if self.active_mode == "live":
            # Live environment visualization
            if self.env is not None and self.visualizer is not None:
                game_surface = self.visualizer.render_state(
                    self.env.state, self.env.mdp.terrain_mtx
                )

                # Preserve exact square tile aspect ratio (no stretching or distortion)
                orig_w, orig_h = game_surface.get_size()
                pad = 24
                avail_w = max(10, self.game_pane_width - pad * 2)
                avail_h = max(10, self.window_height - pad * 2)
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
                offset_y = (self.window_height - new_h) // 2
                self.window.blit(scaled_surface, (offset_x, offset_y))

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
                fill_w = int((p_bar_rect.width - 4) * self.matrix_progress)
                fill_rect = pygame.Rect(p_bar_rect.x + 2, p_bar_rect.y + 2, fill_w, p_bar_rect.height - 4)
                p_color = COLOR_DONE_BADGE if self.matrix_done else (100, 105, 115)
                pygame.draw.rect(self.window, p_color, fill_rect)

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
            # Agent 1 Section
            lbl_font = FontManager.get_font(12, bold=True)
            a0_lbl = lbl_font.render("Agent 1", True, TEXT_PRIMARY)
            self.window.blit(a0_lbl, (sb_x, 88))
            a0_badge_x = sb_x + a0_lbl.get_width() + 10
            UIBanner.draw(self.window, (a0_badge_x, 85, 78, 20), "RED HAT", COLOR_AGENT_RED, font_size=10, bold=True)
            self.dropdown_agent_0.draw(self.window)

            # Agent 2 Section
            a1_lbl = lbl_font.render("Agent 2", True, TEXT_PRIMARY)
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

            # Progress bar
            prog_pct = min(1.0, self.step_count / self.horizon) if self.horizon > 0 else 0
            pbar_rect = pygame.Rect(sb_x + 12, card_y + 102, sb_w - 24, 10)
            UICard.draw(self.window, pbar_rect, bg_color=(245, 245, 245), border_color=BORDER_BLACK, border_radius=0, shadow_offset=1)
            if prog_pct > 0:
                fill_rect = pygame.Rect(sb_x + 13, card_y + 103, int((sb_w - 26) * prog_pct), 8)
                p_color = COLOR_DONE_BADGE if self.state == AppState.DONE else (100, 105, 115)
                pygame.draw.rect(self.window, p_color, fill_rect)

            # Controls & Help Card
            help_y = card_y + card_h + 12
            help_h = 95
            UICard.draw(self.window, (sb_x, help_y, sb_w, help_h), bg_color=BG_CARD_ALT, border_color=BORDER_BLACK, border_radius=0, shadow_offset=2)
            help_t = card_title_font.render("KEYBOARD CONTROLS", True, TEXT_MUTED)
            self.window.blit(help_t, (sb_x + 12, help_y + 8))

            h_f = FontManager.get_font(11, bold=False)
            h1 = h_f.render("- WASD / Arrows : Move Chef 0 (Red)", True, TEXT_SECONDARY)
            h2 = h_f.render("- Space / Enter : Pick up / Drop / Cook", True, TEXT_SECONDARY)
            h3 = h_f.render("- [P] Pause / Resume   - [R] Reset", True, TEXT_MUTED)
            self.window.blit(h1, (sb_x + 12, help_y + 28))
            self.window.blit(h2, (sb_x + 12, help_y + 48))
            self.window.blit(h3, (sb_x + 12, help_y + 68))

            if self.warning_message:
                warn_font = FontManager.get_font(11, bold=False)
                warn_surf = warn_font.render(self.warning_message, True, COLOR_PAUSE)
                self.window.blit(warn_surf, (sb_x, help_y + help_h + 8))

            # Live Action Buttons
            if self.state == AppState.SETUP:
                self.btn_run.rect = pygame.Rect(sb_x, btn_y, sb_w, 44)
                self.btn_run.text = "Run"
                self.btn_run.bg_color = COLOR_RUN
                self.btn_run.hover_color = COLOR_RUN_HOVER
                self.btn_run.text_color = COLOR_RUN_TEXT
                self.btn_run.is_visible = True
                self.btn_run.draw(self.window)

            elif self.state in (AppState.RUNNING, AppState.PAUSED):
                self.btn_pause.rect = pygame.Rect(sb_x, btn_y, half_w, 44)
                self.btn_pause.text = "Resume" if self.state == AppState.PAUSED else "Pause"
                self.btn_pause.is_visible = True
                self.btn_pause.draw(self.window)

                self.btn_reset.rect = pygame.Rect(sb_x + half_w + 12, btn_y, half_w, 44)
                self.btn_reset.text = "Reset"
                self.btn_reset.is_visible = True
                self.btn_reset.draw(self.window)

            elif self.state == AppState.DONE:
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
            self.window.blit(f_sec.render("TRIALS PER PAIR (K)", True, TEXT_PRIMARY), (sb_x, 328))
            self.btn_k_minus.draw(self.window)
            f_k = FontManager.get_mono_font(15, bold=True)
            k_surf = f_k.render(str(self.matrix_k), True, TEXT_PRIMARY)
            self.window.blit(k_surf, (sb_x + 172, 332))
            self.btn_k_plus.draw(self.window)

            # Matrix Benchmark Buttons
            if self.matrix_running:
                # Disabled running button
                self.btn_run_matrix.text = "Simulating..."
                self.btn_run_matrix.bg_color = (210, 214, 220)
                self.btn_run_matrix.hover_color = (210, 214, 220)
                self.btn_run_matrix.rect = pygame.Rect(sb_x, btn_y, sb_w, 44)
                self.btn_run_matrix.draw(self.window)

            elif self.matrix_done:
                # View Graphs button (opens Streamlit port 8502)
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
                self.btn_run_matrix.text = "Run Matrix Benchmark"
                self.btn_run_matrix.bg_color = COLOR_RUN
                self.btn_run_matrix.hover_color = COLOR_RUN_HOVER
                self.btn_run_matrix.rect = pygame.Rect(sb_x, btn_y, sb_w, 44)
                self.btn_run_matrix.draw(self.window)

        # ----------------------------------------------------
        # 3. Floating Overlay Pass (Draw open dropdown popups on top in Live Mode)
        # ----------------------------------------------------
        if self.active_mode == "live" and not self.is_mid_run:
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
