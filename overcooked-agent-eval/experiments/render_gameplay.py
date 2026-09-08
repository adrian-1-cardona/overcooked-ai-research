"""Render Overcooked-AI gameplay in real time, export to video, or replay telemetry.

This command brings Overcooked-AI visualization to life by bridging the built-in
StateVisualizer with live episode simulation, interactive keyboard play,
video export, and CSV replay.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure parent and external submodule paths are accessible regardless of cwd
CURRENT_FILE = Path(__file__).resolve()
AGENT_EVAL_DIR = CURRENT_FILE.parents[1]
REPO_ROOT = AGENT_EVAL_DIR.parent
EXTERNAL_OVERCOOKED = REPO_ROOT / "external" / "overcooked_ai" / "src"

if str(EXTERNAL_OVERCOOKED) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_OVERCOOKED))
if str(AGENT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_EVAL_DIR))

import cv2
import numpy as np
import pygame

from overcooked_ai_py.agents.agent import Agent, RandomAgent
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


DEFAULT_LAYOUT = "cramped_room"
DEFAULT_HORIZON = 400
DEFAULT_SEED = 42
DEFAULT_FPS = 10
DEFAULT_TILE_SIZE = 75
DEFAULT_OUTPUT_VIDEO = AGENT_EVAL_DIR / "results" / "gameplay.mp4"
DEFAULT_OUTPUT_FRAMES = AGENT_EVAL_DIR / "results" / "frames"

ACTION_NAME_TO_ACTION: dict[str, Any] = {
    "north": Direction.NORTH,
    "south": Direction.SOUTH,
    "east": Direction.EAST,
    "west": Direction.WEST,
    "stay": Action.STAY,
    "interact": Action.INTERACT,
}


class StayAgent(Agent):
    """An agent that always chooses STAY."""

    def action(self, state: OvercookedState) -> tuple[Any, dict[str, Any]]:
        return Action.STAY, {}

    def actions(self, states: Sequence[OvercookedState]) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
        return tuple(Action.STAY for _ in states), [{} for _ in states]


class HumanAgent(Agent):
    """An agent controlled via keyboard in the Pygame window."""

    def __init__(self) -> None:
        super().__init__()
        self.pending_action: Any = Action.STAY

    def set_action(self, action: Any) -> None:
        self.pending_action = action

    def action(self, state: OvercookedState) -> tuple[Any, dict[str, Any]]:
        act = self.pending_action
        self.pending_action = Action.STAY
        return act, {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Overcooked-AI gameplay in a live window, export to MP4 video, or replay CSV."
    )
    parser.add_argument(
        "--layout",
        type=str,
        default=DEFAULT_LAYOUT,
        help=f"Layout name (default: {DEFAULT_LAYOUT})",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Max timesteps per episode (default: {DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Playback / video framerate in FPS (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "window", "video", "frames"],
        default="auto",
        help="Rendering mode: window (live GUI), video (MP4), frames (PNGs), or auto (default: auto)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for exported video or frames directory (default: results/gameplay.mp4 or results/frames)",
    )
    parser.add_argument(
        "--agent-0",
        type=str,
        choices=["random", "stay", "human"],
        default="random",
        help="Policy for Agent 0 (default: random)",
    )
    parser.add_argument(
        "--agent-1",
        type=str,
        choices=["random", "stay"],
        default="random",
        help="Policy for Agent 1 (default: random)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"Grid cell pixel size (default: {DEFAULT_TILE_SIZE})",
    )
    parser.add_argument(
        "--replay-csv",
        type=Path,
        default=None,
        help="Path to recorded CSV telemetry file to replay",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=1,
        help="Episode number to replay if CSV contains multiple episodes (default: 1)",
    )
    args = parser.parse_args()

    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if args.fps < 1:
        parser.error("--fps must be at least 1")

    return args


def create_agent(name: str, index: int, mdp: OvercookedGridworld) -> Agent:
    if name == "random":
        agent = RandomAgent(all_actions=True)
    elif name == "stay":
        agent = StayAgent()
    elif name == "human":
        agent = HumanAgent()
    else:
        raise ValueError(f"Unknown agent type: {name}")

    agent.set_agent_index(index)
    agent.set_mdp(mdp)
    return agent


def is_gui_available() -> bool:
    """Check whether a graphical window display is available."""
    try:
        pygame.display.init()
        # Test creating a minimal surface window
        test_surf = pygame.display.set_mode((1, 1))
        pygame.display.quit()
        return True
    except Exception:
        return False


def surface_to_bgr(surface: pygame.Surface) -> np.ndarray:
    """Convert a Pygame Surface to an OpenCV BGR ndarray (height, width, 3)."""
    rgb = pygame.surfarray.array3d(surface)
    rgb = np.transpose(rgb, (1, 0, 2))  # (height, width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def get_available_layouts() -> list[str]:
    """Return a sorted list of layout names found in the Overcooked-AI submodule."""
    from overcooked_ai_py.static import LAYOUTS_DIR
    layout_dir = Path(LAYOUTS_DIR)
    if not layout_dir.exists():
        return []
    return sorted(p.stem for p in layout_dir.glob("*.layout"))


def load_replay_actions(
    csv_path: Path, target_episode: int = 1
) -> tuple[str | None, list[tuple[Any, Any]]]:
    """Load recorded actions from a single-episode or multi-episode CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Replay CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Replay CSV is empty: {csv_path}")

    first_row = rows[0]
    if "agent_0_action" not in first_row and "agent_1_action" not in first_row:
        raise ValueError(
            f"Replay CSV '{csv_path.name}' does not contain action columns ('agent_0_action', 'agent_1_action'). "
            f"Found columns: {list(first_row.keys())}. "
            "Make sure you are passing a timestep telemetry CSV (e.g. random_baseline_*.csv or multi_episode_*.csv), "
            "not an episode_metrics summary file."
        )

    detected_layout: str | None = None
    actions: list[tuple[Any, Any]] = []

    for row in rows:
        ep_val = row.get("episode") or row.get("episode_id")
        if ep_val is not None and int(ep_val) != target_episode:
            continue

        if detected_layout is None and "layout_name" in row:
            detected_layout = row["layout_name"]

        a0_str = row.get("agent_0_action", "stay")
        a1_str = row.get("agent_1_action", "stay")
        a0 = ACTION_NAME_TO_ACTION.get(a0_str, Action.STAY)
        a1 = ACTION_NAME_TO_ACTION.get(a1_str, Action.STAY)
        actions.append((a0, a1))

    return detected_layout, actions


def run_window_rendering(
    env: OvercookedEnv,
    agents: list[Agent],
    visualizer: StateVisualizer,
    horizon: int,
    initial_fps: int,
    layout_name: str,
    replay_actions: list[tuple[Any, Any]] | None = None,
) -> None:
    """Render gameplay live in an interactive Pygame desktop window."""
    pygame.init()
    pygame.display.init()
    clock = pygame.time.Clock()

    state = env.state
    cumulative_score = 0
    step_count = 0
    fps = initial_fps
    is_paused = False
    human_mode = any(isinstance(a, HumanAgent) for a in agents)
    human_agent: HumanAgent | None = next((a for a in agents if isinstance(a, HumanAgent)), None)

    # Initial render
    hud_data = StateVisualizer.default_hud_data(
        state,
        score=cumulative_score,
        time_left=max(0, horizon - step_count),
    )
    initial_surface = visualizer.render_state(state, env.mdp.terrain_mtx, hud_data=hud_data)
    window = pygame.display.set_mode(initial_surface.get_size())

    def update_caption() -> None:
        mode_str = "PAUSED" if is_paused else f"{fps} FPS"
        caption = f"Overcooked-AI: {layout_name} | Step {step_count}/{horizon} | Score: {cumulative_score} [{mode_str}]"
        pygame.display.set_caption(caption)

    update_caption()

    print("\n" + "=" * 60)
    print(f"  Overcooked-AI Live Gameplay Rendering [{layout_name}]")
    print("=" * 60)
    print("  Controls:")
    print("    [SPACE / P]  : Pause / Resume simulation")
    print("    [RIGHT ARROW]: Step 1 timestep forward (when paused)")
    print("    [UP / DOWN]  : Increase / Decrease FPS speed")
    print("    [R]          : Restart episode")
    print("    [ESC / Q]    : Quit cleanly")
    if human_mode:
        print("  Human Agent (Chef 0) Controls:")
        print("    [WASD / Arrows]: Move North, South, West, East")
        print("    [SPACE / ENTER / F]: Interact / Pick up / Drop")
        print("    [P]: Pause simulation")
    print("=" * 60 + "\n")

    running = True
    while running:
        should_advance_step = not is_paused
        human_action = Action.STAY

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                    break

                # Restart
                elif event.key == pygame.K_r:
                    env.reset()
                    state = env.state
                    cumulative_score = 0
                    step_count = 0
                    is_paused = False
                    update_caption()

                # Speed control
                elif event.key == pygame.K_UP and not human_mode:
                    fps = min(60, fps + 2)
                    update_caption()
                elif event.key == pygame.K_DOWN and not human_mode:
                    fps = max(1, fps - 2)
                    update_caption()

                # Pause / Resume
                elif (event.key == pygame.K_p) or (event.key == pygame.K_SPACE and not human_mode):
                    is_paused = not is_paused
                    update_caption()

                # Step forward when paused
                elif event.key == pygame.K_RIGHT and is_paused:
                    should_advance_step = True

                # Human agent controls
                if human_mode and human_agent is not None:
                    if event.key in (pygame.K_w, pygame.K_UP):
                        human_action = Direction.NORTH
                    elif event.key in (pygame.K_s, pygame.K_DOWN):
                        human_action = Direction.SOUTH
                    elif event.key in (pygame.K_a, pygame.K_LEFT):
                        human_action = Direction.WEST
                    elif event.key in (pygame.K_d, pygame.K_RIGHT):
                        human_action = Direction.EAST
                    elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_f):
                        human_action = Action.INTERACT

        if not running:
            break

        if human_mode and human_agent is not None:
            human_agent.set_action(human_action)

        # Step environment
        if should_advance_step and step_count < horizon:
            if replay_actions is not None:
                if step_count < len(replay_actions):
                    joint_action = replay_actions[step_count]
                else:
                    joint_action = (Action.STAY, Action.STAY)
            else:
                joint_action = tuple(agent.action(state)[0] for agent in agents)

            next_state, sparse_reward, done, _ = env.step(joint_action)
            cumulative_score += sparse_reward
            state = next_state
            step_count += 1
            update_caption()

            if done or step_count >= horizon:
                is_paused = True
                update_caption()
                print(f"Episode finished at step {step_count}! Final Score: {cumulative_score}")

        # Render current frame
        hud_data = StateVisualizer.default_hud_data(
            state,
            score=cumulative_score,
            time_left=max(0, horizon - step_count),
        )
        surface = visualizer.render_state(state, env.mdp.terrain_mtx, hud_data=hud_data)
        window.blit(surface, (0, 0))
        pygame.display.flip()

        clock.tick(fps)

    pygame.display.quit()
    pygame.quit()
    print("Window closed. Exiting.")


def run_video_export(
    env: OvercookedEnv,
    agents: list[Agent],
    visualizer: StateVisualizer,
    horizon: int,
    fps: int,
    layout_name: str,
    output_path: Path,
    replay_actions: list[tuple[Any, Any]] | None = None,
) -> None:
    """Render gameplay and save as an MP4 video file using OpenCV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state = env.state
    cumulative_score = 0
    step_count = 0

    hud_data = StateVisualizer.default_hud_data(
        state,
        score=cumulative_score,
        time_left=max(0, horizon - step_count),
    )
    first_surface = visualizer.render_state(state, env.mdp.terrain_mtx, hud_data=hud_data)
    w, h = first_surface.get_size()
    # Video codecs require even dimensions
    out_w = w + (w % 2)
    out_h = h + (h % 2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (out_w, out_h))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open OpenCV VideoWriter for output: {output_path}")

    def write_frame(surf: pygame.Surface) -> None:
        bgr = surface_to_bgr(surf)
        if bgr.shape[1] != out_w or bgr.shape[0] != out_h:
            bgr = cv2.copyMakeBorder(
                bgr, 0, out_h - bgr.shape[0], 0, out_w - bgr.shape[1],
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        writer.write(bgr)

    # Initial frame
    write_frame(first_surface)

    print(f"Rendering {horizon} steps to video: {output_path.resolve()} ({fps} FPS)...")

    while step_count < horizon:
        if replay_actions is not None:
            if step_count < len(replay_actions):
                joint_action = replay_actions[step_count]
            else:
                joint_action = (Action.STAY, Action.STAY)
        else:
            joint_action = tuple(agent.action(state)[0] for agent in agents)

        next_state, sparse_reward, done, _ = env.step(joint_action)
        cumulative_score += sparse_reward
        state = next_state
        step_count += 1

        hud_data = StateVisualizer.default_hud_data(
            state,
            score=cumulative_score,
            time_left=max(0, horizon - step_count),
        )
        surface = visualizer.render_state(state, env.mdp.terrain_mtx, hud_data=hud_data)
        write_frame(surface)

        if step_count % 50 == 0 or step_count == horizon:
            print(f"  Rendered {step_count}/{horizon} frames (Score: {cumulative_score})...")

        if done:
            break

    writer.release()
    file_size_kb = output_path.stat().st_size / 1024
    print("\nVideo rendering complete!")
    print(f"Layout       : {layout_name}")
    print(f"Frames       : {step_count + 1}")
    print(f"Final Score  : {cumulative_score}")
    print(f"Output Video : {output_path.resolve()} ({file_size_kb:.1f} KB)")


def run_frames_export(
    env: OvercookedEnv,
    agents: list[Agent],
    visualizer: StateVisualizer,
    horizon: int,
    layout_name: str,
    output_dir: Path,
    replay_actions: list[tuple[Any, Any]] | None = None,
) -> None:
    """Render gameplay and save consecutive PNG image frames to a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    state = env.state
    cumulative_score = 0
    step_count = 0

    print(f"Rendering {horizon} steps as PNG frames into: {output_dir.resolve()}...")

    hud_data = StateVisualizer.default_hud_data(
        state,
        score=cumulative_score,
        time_left=max(0, horizon - step_count),
    )
    first_surface = visualizer.render_state(state, env.mdp.terrain_mtx, hud_data=hud_data)
    pygame.image.save(first_surface, str(output_dir / "frame_0000.png"))

    while step_count < horizon:
        if replay_actions is not None:
            if step_count < len(replay_actions):
                joint_action = replay_actions[step_count]
            else:
                joint_action = (Action.STAY, Action.STAY)
        else:
            joint_action = tuple(agent.action(state)[0] for agent in agents)

        next_state, sparse_reward, done, _ = env.step(joint_action)
        cumulative_score += sparse_reward
        state = next_state
        step_count += 1

        hud_data = StateVisualizer.default_hud_data(
            state,
            score=cumulative_score,
            time_left=max(0, horizon - step_count),
        )
        surface = visualizer.render_state(state, env.mdp.terrain_mtx, hud_data=hud_data)
        frame_filename = output_dir / f"frame_{step_count:04d}.png"
        pygame.image.save(surface, str(frame_filename))

        if step_count % 50 == 0 or step_count == horizon:
            print(f"  Saved {step_count}/{horizon} frames...")

        if done:
            break

    print("\nFrames rendering complete!")
    print(f"Total Frames Saved: {step_count + 1}")
    print(f"Output Directory  : {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    layout = args.layout
    replay_actions: list[tuple[Any, Any]] | None = None

    if args.replay_csv is not None:
        detected_layout, replay_actions = load_replay_actions(
            args.replay_csv, target_episode=args.episode
        )
        if detected_layout is not None and args.layout == DEFAULT_LAYOUT:
            layout = detected_layout
        if len(replay_actions) < args.horizon:
            args.horizon = len(replay_actions)
        print(f"[Replay Mode] Loaded {len(replay_actions)} actions from {args.replay_csv}")

    available_layouts = get_available_layouts()
    if available_layouts and layout not in available_layouts:
        common = [
            "cramped_room",
            "asymmetric_advantages",
            "coordination_ring",
            "forced_coordination",
            "counter_circuit",
        ]
        print(f"\n[ERROR] Layout '{layout}' not found.")
        print(f"Popular choices: {', '.join(common)}")
        print(f"All available ({len(available_layouts)} layouts): {', '.join(available_layouts[:15])}...\n")
        sys.exit(1)

    # Build environment and visualizer
    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=args.horizon, info_level=0)
    visualizer = StateVisualizer(tile_size=args.tile_size)

    # Agents
    agents = [
        create_agent(args.agent_0, 0, mdp),
        create_agent(args.agent_1, 1, mdp),
    ]

    # Resolve mode
    mode = args.mode
    if mode == "auto":
        mode = "window" if is_gui_available() else "video"

    if mode == "window":
        if not is_gui_available():
            print(
                "[WARNING] No graphical display detected. Falling back to video export (--mode video)."
            )
            mode = "video"
        else:
            run_window_rendering(
                env=env,
                agents=agents,
                visualizer=visualizer,
                horizon=args.horizon,
                initial_fps=args.fps,
                layout_name=layout,
                replay_actions=replay_actions,
            )
            return

    if mode == "video":
        output_video = args.output or DEFAULT_OUTPUT_VIDEO
        run_video_export(
            env=env,
            agents=agents,
            visualizer=visualizer,
            horizon=args.horizon,
            fps=args.fps,
            layout_name=layout,
            output_path=output_video,
            replay_actions=replay_actions,
        )
    elif mode == "frames":
        output_frames = args.output or DEFAULT_OUTPUT_FRAMES
        run_frames_export(
            env=env,
            agents=agents,
            visualizer=visualizer,
            horizon=args.horizon,
            layout_name=layout,
            output_dir=output_frames,
            replay_actions=replay_actions,
        )


if __name__ == "__main__":
    main()
