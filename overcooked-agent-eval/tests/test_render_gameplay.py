"""Unit tests for the Overcooked-AI gameplay rendering command."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure paths
TEST_DIR = Path(__file__).resolve().parent
AGENT_EVAL_DIR = TEST_DIR.parent
EXPERIMENTS_DIR = AGENT_EVAL_DIR / "experiments"
EXTERNAL_OVERCOOKED = AGENT_EVAL_DIR.parent / "external" / "overcooked_ai" / "src"

if str(EXTERNAL_OVERCOOKED) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_OVERCOOKED))
if str(AGENT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_EVAL_DIR))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import pygame

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from experiments.render_gameplay import (
    HumanAgent,
    StayAgent,
    create_agent,
    load_replay_actions,
    run_frames_export,
    run_video_export,
    surface_to_bgr,
)


class TestRenderGameplay(unittest.TestCase):
    def setUp(self) -> None:
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=10, info_level=0)
        self.visualizer = StateVisualizer(tile_size=40)

    def test_agent_creation(self) -> None:
        random_agent = create_agent("random", 0, self.mdp)
        stay_agent = create_agent("stay", 1, self.mdp)
        human_agent = create_agent("human", 0, self.mdp)

        self.assertEqual(random_agent.agent_index, 0)
        self.assertEqual(stay_agent.agent_index, 1)
        self.assertIsInstance(stay_agent, StayAgent)
        self.assertIsInstance(human_agent, HumanAgent)

        state = self.env.state
        action, _ = stay_agent.action(state)
        self.assertEqual(action, Action.STAY)

        human_agent.set_action(Direction.NORTH)
        action, _ = human_agent.action(state)
        self.assertEqual(action, Direction.NORTH)

        # After acting, resets to STAY
        action_next, _ = human_agent.action(state)
        self.assertEqual(action_next, Action.STAY)

    def test_surface_to_bgr(self) -> None:
        surface = pygame.Surface((50, 40))
        surface.fill((255, 0, 0))  # Red in RGB
        bgr = surface_to_bgr(surface)

        self.assertEqual(bgr.shape, (40, 50, 3))
        # Blue channel should be 0, Red channel should be 255
        self.assertEqual(bgr[0, 0, 0], 0)  # B
        self.assertEqual(bgr[0, 0, 1], 0)  # G
        self.assertEqual(bgr[0, 0, 2], 255)  # R

    def test_load_replay_actions(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(["episode", "timestep", "agent_0_action", "agent_1_action"])
            writer.writerow([1, 1, "north", "interact"])
            writer.writerow([1, 2, "stay", "west"])

        try:
            detected_layout, actions = load_replay_actions(csv_path, target_episode=1)
            self.assertIsNone(detected_layout)
            self.assertEqual(len(actions), 2)
            self.assertEqual(actions[0], (Direction.NORTH, Action.INTERACT))
            self.assertEqual(actions[1], (Action.STAY, Direction.WEST))
        finally:
            csv_path.unlink(missing_ok=True)

    def test_video_export(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = Path(f.name)

        try:
            agents = [create_agent("stay", 0, self.mdp), create_agent("stay", 1, self.mdp)]
            run_video_export(
                env=self.env,
                agents=agents,
                visualizer=self.visualizer,
                horizon=5,
                fps=5,
                layout_name="cramped_room",
                output_path=video_path,
            )
            self.assertTrue(video_path.exists())
            self.assertGreater(video_path.stat().st_size, 1000)
        finally:
            video_path.unlink(missing_ok=True)

    def test_frames_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            agents = [create_agent("stay", 0, self.mdp), create_agent("stay", 1, self.mdp)]
            run_frames_export(
                env=self.env,
                agents=agents,
                visualizer=self.visualizer,
                horizon=4,
                layout_name="cramped_room",
                output_dir=out_dir,
            )
            frame_files = sorted(out_dir.glob("frame_*.png"))
            self.assertEqual(len(frame_files), 5)  # initial frame + 4 steps

    def test_get_available_layouts(self) -> None:
        from experiments.render_gameplay import get_available_layouts
        layouts = get_available_layouts()
        self.assertIn("cramped_room", layouts)
        self.assertIn("asymmetric_advantages", layouts)
        self.assertIn("coordination_ring", layouts)

    def test_load_replay_actions_invalid_csv(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(["episode", "score", "total_steps"])
            writer.writerow([1, 10, 400])

        try:
            with self.assertRaises(ValueError) as ctx:
                load_replay_actions(csv_path, target_episode=1)
            self.assertIn("does not contain action columns", str(ctx.exception))
        finally:
            csv_path.unlink(missing_ok=True)

    def test_window_rendering_lifecycle(self) -> None:
        from experiments.render_gameplay import run_window_rendering
        pygame.init()
        pygame.display.init()
        # Post key events and QUIT event to exercise the loop
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        pygame.event.post(pygame.event.Event(pygame.QUIT))

        agents = [create_agent("human", 0, self.mdp), create_agent("stay", 1, self.mdp)]
        run_window_rendering(
            env=self.env,
            agents=agents,
            visualizer=self.visualizer,
            horizon=3,
            initial_fps=10,
            layout_name="cramped_room",
        )


if __name__ == "__main__":
    unittest.main()
