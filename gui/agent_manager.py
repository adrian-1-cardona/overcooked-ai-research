"""Agent loading, factory, and layout discovery for the Overcooked GUI."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Sequence

# Set legacy Keras compatibility for bundled TF assets
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_EVAL_DIR = REPO_ROOT / "overcooked-agent-eval"
EXTERNAL_OVERCOOKED = REPO_ROOT / "external" / "overcooked_ai" / "src"

for p in (REPO_ROOT, AGENT_EVAL_DIR, EXTERNAL_OVERCOOKED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pygame
from overcooked_ai_py.agents.agent import Agent, RandomAgent
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState


# Pretrained PPO layout mapping
LAYOUT_TO_ASSET_NAME: dict[str, str] = {
    "cramped_room": "CrampedRoom",
    "asymmetric_advantages": "AsymmetricAdvantages",
    "coordination_ring": "CoordinationRing",
    "forced_coordination": "ForcedCoordination",
    "counter_circuit_o_1order": "CounterCircuit1Order",
}

# Trainer cache so loading PPO is only done once per layout
_TRAINER_CACHE: dict[str, Any] = {}


class StayAgent(Agent):
    """An agent that always remains stationary."""

    def action(self, state: OvercookedState) -> tuple[Any, dict[str, Any]]:
        return Action.STAY, {}

    def actions(self, states: Sequence[OvercookedState]) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
        return tuple(Action.STAY for _ in states), [{} for _ in states]


class InteractiveHumanAgent(Agent):
    """Keyboard-controlled human agent supporting player 0 and player 1 keybindings."""

    def __init__(self, agent_index: int = 0) -> None:
        super().__init__()
        self.set_agent_index(agent_index)
        self.pending_action: Any = Action.STAY

    def set_action(self, action: Any) -> None:
        self.pending_action = action

    def action(self, state: OvercookedState) -> tuple[Any, dict[str, Any]]:
        act = self.pending_action
        self.pending_action = Action.STAY
        return act, {}

    def handle_keydown(self, key: int, is_two_player: bool = False) -> bool:
        """Handle a Pygame KEYDOWN event. Returns True if this agent consumed the key."""
        if self.agent_index == 0:
            # Primary controls: WASD and Arrow Keys (if 1 player)
            if key in (pygame.K_w, pygame.K_UP if not is_two_player else -1):
                self.pending_action = Direction.NORTH
                return True
            elif key in (pygame.K_s, pygame.K_DOWN if not is_two_player else -1):
                self.pending_action = Direction.SOUTH
                return True
            elif key in (pygame.K_a, pygame.K_LEFT if not is_two_player else -1):
                self.pending_action = Direction.WEST
                return True
            elif key in (pygame.K_d, pygame.K_RIGHT if not is_two_player else -1):
                self.pending_action = Direction.EAST
                return True
            elif key in (pygame.K_SPACE, pygame.K_f, pygame.K_RETURN if not is_two_player else -1):
                self.pending_action = Action.INTERACT
                return True
        else:
            # Secondary controls (Chef 1): Arrow Keys or IJKL
            if key in (pygame.K_UP, pygame.K_i):
                self.pending_action = Direction.NORTH
                return True
            elif key in (pygame.K_DOWN, pygame.K_k):
                self.pending_action = Direction.SOUTH
                return True
            elif key in (pygame.K_LEFT, pygame.K_j):
                self.pending_action = Direction.WEST
                return True
            elif key in (pygame.K_RIGHT, pygame.K_l):
                self.pending_action = Direction.EAST
                return True
            elif key in (pygame.K_RETURN, pygame.K_RSHIFT, pygame.K_o):
                self.pending_action = Action.INTERACT
                return True

        return False


def get_available_layouts() -> list[str]:
    """Return all available Overcooked layouts, prioritizing the main research layouts."""
    priority_order = [
        "cramped_room",
        "asymmetric_advantages",
        "coordination_ring",
        "forced_coordination",
        "counter_circuit_o_1order",
        "counter_circuit",
        "bottleneck",
        "corridor",
        "large_room",
    ]
    try:
        from overcooked_ai_py.static import LAYOUTS_DIR
        layout_dir = Path(LAYOUTS_DIR)
        all_found = sorted(p.stem for p in layout_dir.glob("*.layout"))
    except Exception:
        all_found = list(priority_order)

    # Put priority layouts at top
    result: list[str] = [name for name in priority_order if name in all_found]
    remaining = [name for name in all_found if name not in result]
    result.extend(remaining)
    return result


def is_ppo_supported(layout_name: str) -> bool:
    """Check if a bundled pretrained PPO model exists for this layout."""
    return layout_name in LAYOUT_TO_ASSET_NAME


def get_ppo_agent(layout_name: str, agent_index: int, mdp: OvercookedGridworld) -> Agent:
    """Load or retrieve cached PPO agent for the specified layout and agent index."""
    if not is_ppo_supported(layout_name):
        raise ValueError(f"Pretrained PPO model is not bundled for layout '{layout_name}'.")

    from experiments.run_pretrained_policy_comparison import bundled_checkpoint, load_trainer
    from human_aware_rl.rllib.rllib import get_agent_from_trainer

    if layout_name not in _TRAINER_CACHE:
        checkpoint_path = bundled_checkpoint(layout_name, "SP")
        print(f"[PPO Loader] Initializing PPO policy from: {checkpoint_path.name}...")
        trainer = load_trainer(checkpoint_path)
        _TRAINER_CACHE[layout_name] = trainer

    trainer = _TRAINER_CACHE[layout_name]
    agent = get_agent_from_trainer(trainer, policy_id="ppo", agent_index=agent_index)
    agent.set_mdp(mdp)
    agent.reset()
    return agent


def create_agent(agent_type: str, agent_index: int, mdp: OvercookedGridworld, layout_name: str) -> tuple[Agent, str | None]:
    """
    Factory creating an Agent instance.

    Returns:
        (agent_instance, warning_message_if_any)
    """
    warning: str | None = None
    agent_type_lower = agent_type.lower().strip()

    if agent_type_lower in ("human", "human (keyboard)", "human (chef 0)", "human (chef 1)"):
        agent = InteractiveHumanAgent(agent_index=agent_index)

    elif agent_type_lower in ("ppo", "ppo (self-play)", "ppo_sp"):
        if is_ppo_supported(layout_name):
            try:
                agent = get_ppo_agent(layout_name, agent_index, mdp)
            except Exception as e:
                print(f"[Warning] Failed to load PPO ({e}); falling back to Upstream Greedy.")
                warning = f"PPO load failed: {e}. Falling back to Upstream Greedy."
                from overcooked_ai_py.agents.agent import GreedyHumanModel
                from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
                mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS)
                agent = GreedyHumanModel(mlam)
        else:
            warning = f"No bundled PPO model for '{layout_name}'. Using Upstream Greedy agent."
            from overcooked_ai_py.agents.agent import GreedyHumanModel
            from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
            mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS)
            agent = GreedyHumanModel(mlam)

    elif agent_type_lower in ("greedy", "greedy agent", "greedy agent (upstream)", "upstream", "upstream (baseline)", "greedy_human_model"):
        from overcooked_ai_py.agents.agent import GreedyHumanModel
        from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
        mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS)
        agent = GreedyHumanModel(mlam)

    elif agent_type_lower in ("random", "random agent"):
        agent = RandomAgent(all_actions=True)

    elif agent_type_lower in ("stay", "stay agent", "idle"):
        agent = StayAgent()

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent.set_agent_index(agent_index)
    agent.set_mdp(mdp)
    return agent, warning
