"""Agent loading, factory, and layout discovery for the Overcooked GUI."""

from __future__ import annotations

import contextlib
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

# Set legacy Keras compatibility and suppress TF C++ logs
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@contextlib.contextmanager
def suppress_c_stderr():
    """Silence noisy C-level and Python-level deprecation warnings during TF/Ray model loading."""
    has_dup = False
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_stderr = os.dup(2)
        os.dup2(null_fd, 2)
        os.close(null_fd)
        has_dup = True
    except Exception:
        has_dup = False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        if has_dup:
            os.dup2(save_stderr, 2)
            os.close(save_stderr)

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


def evict_trainer_cache(key_substr: str | None = None) -> None:
    """Evict cached trainers from memory when agents are deleted or reset."""
    if key_substr is None:
        _TRAINER_CACHE.clear()
    else:
        keys_to_del = [k for k in _TRAINER_CACHE if key_substr in k]
        for k in keys_to_del:
            _TRAINER_CACHE.pop(k, None)


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
    """Return all available Overcooked layouts, prioritizing the main research layouts.

    Filters out test/single-player/4-player layouts so that only verified playable
    2-player layouts are presented in the GUI.
    """
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
    # Unplayable layouts that are not 2-player cooperative layouts
    unplayable = {
        "cramped_room_single",     # 1-player layout
        "multiplayer_schelling",   # 4-player layout
        "old_dynamics_cook_test",  # 1-player test layout
        "old_dynamics_put_test",   # 1-player test layout
    }
    try:
        from overcooked_ai_py.static import LAYOUTS_DIR
        layout_dir = Path(LAYOUTS_DIR)
        all_found = sorted(p.stem for p in layout_dir.glob("*.layout"))
    except Exception:
        all_found = list(priority_order)

    # Filter unplayable
    all_found = [name for name in all_found if name not in unplayable]

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

    with suppress_c_stderr():
        from experiments.run_pretrained_policy_comparison import bundled_checkpoint, load_trainer
        from human_aware_rl.rllib.rllib import get_agent_from_trainer

    if layout_name not in _TRAINER_CACHE:
        checkpoint_path = bundled_checkpoint(layout_name, "SP")
        print(f"[PPO Loader] Initializing PPO policy from: {checkpoint_path.name}...")
        with suppress_c_stderr():
            trainer = load_trainer(checkpoint_path)
        _TRAINER_CACHE[layout_name] = trainer
        print(f"[PPO Loader] PPO policy loaded successfully.")

    trainer = _TRAINER_CACHE[layout_name]
    with suppress_c_stderr():
        agent = get_agent_from_trainer(trainer, policy_id="ppo", agent_index=agent_index)
        agent.set_mdp(mdp)
        agent.reset()
    return agent


def is_upstream_greedy_supported(mdp: OvercookedGridworld) -> bool:
    """Check if upstream GreedyHumanModel / MLAM supports this layout (strictly 3-onion soup only)."""
    try:
        orders = getattr(mdp, "start_all_orders", None)
        if not orders or len(orders) != 1:
            return False
        first = orders[0]
        ingrs = first.get("ingredients") if isinstance(first, dict) else first
        return list(ingrs) == ["onion", "onion", "onion"]
    except Exception:
        return False


def _create_fallback_greedy_agent(mdp: OvercookedGridworld, agent_index: int) -> Agent:
    """Create a reliable greedy agent (Upstream GreedyHumanModel or fallback GreedySymbolSearchAgent)."""
    if is_upstream_greedy_supported(mdp):
        try:
            from overcooked_ai_py.agents.agent import GreedyHumanModel
            from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
            mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS)
            agent = GreedyHumanModel(mlam)
        except Exception:
            from agents.greedy_symbol_search import GreedySymbolSearchAgent
            agent = GreedySymbolSearchAgent()
    else:
        from agents.greedy_symbol_search import GreedySymbolSearchAgent
        agent = GreedySymbolSearchAgent()
    agent.set_agent_index(agent_index)
    agent.set_mdp(mdp)
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
                print(f"[Warning] Failed to load PPO ({e}); falling back to Greedy.")
                warning = f"PPO load failed: {e}. Falling back to Greedy."
                agent = _create_fallback_greedy_agent(mdp, agent_index)
        else:
            warning = f"No bundled PPO model for '{layout_name}'. Using Greedy agent."
            agent = _create_fallback_greedy_agent(mdp, agent_index)

    elif agent_type_lower in ("greedy", "greedy agent", "greedy agent (upstream)", "upstream", "upstream (baseline)", "greedy_human_model"):
        agent = _create_fallback_greedy_agent(mdp, agent_index)

    elif agent_type_lower in ("random", "random agent"):
        agent = RandomAgent(all_actions=True)

    elif agent_type_lower in ("stay", "stay agent", "idle"):
        agent = StayAgent()

    elif agent_type_lower in ("bc", "bc agent", "human model", "human_model", "behavior cloning"):
        if is_ppo_supported(layout_name):
            try:
                with suppress_c_stderr():
                    from experiments.run_pretrained_policy_comparison import bundled_checkpoint, load_trainer
                    from human_aware_rl.rllib.rllib import get_agent_from_trainer
                    bc_key = f"bc_{layout_name}"
                    if bc_key not in _TRAINER_CACHE:
                        checkpoint_path = bundled_checkpoint(layout_name, "BC")
                        trainer = load_trainer(checkpoint_path)
                        _TRAINER_CACHE[bc_key] = trainer
                    trainer = _TRAINER_CACHE[bc_key]
                    agent = get_agent_from_trainer(trainer, policy_id="bc", agent_index=agent_index)
                    agent.set_agent_index(agent_index)
                    agent.set_mdp(mdp)
                    agent.reset()
                    agent.set_agent_index(agent_index)
                    agent.set_mdp(mdp)
                return agent, None
            except Exception as e:
                warning = f"BC load failed: {e}; using Greedy Chef."
                agent = _create_fallback_greedy_agent(mdp, agent_index)
                return agent, warning
        else:
            warning = f"BC human model is not bundled for '{layout_name}'; using Greedy Chef."
            agent = _create_fallback_greedy_agent(mdp, agent_index)
            return agent, warning

    elif agent_type.startswith("custom:"):
        model_path = Path(agent_type[len("custom:"):])
        if not model_path.exists():
            warning = f"Custom agent directory not found: '{model_path.name}'. Using Greedy agent."
            return _create_fallback_greedy_agent(mdp, agent_index), warning

        meta_file = model_path / "metadata.json"
        warning = None
        if meta_file.is_file():
            import json
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
            recipe = meta.get("recipe", "ppo_greedy")

            # Check if this model was trained on all maps or has layout subdirectories
            layout_target_dir = model_path / layout_name
            if layout_target_dir.is_dir() and any(layout_target_dir.glob("checkpoint_*")):
                active_model_dir = layout_target_dir
                active_meta_file = active_model_dir / "metadata.json"
                if active_meta_file.is_file():
                    try:
                        with open(active_meta_file, "r", encoding="utf-8") as f:
                            active_meta = json.load(f)
                        recipe = active_meta.get("recipe", recipe)
                    except Exception:
                        pass
                layout = layout_name
            else:
                # Check if model_path directly contains checkpoints (single-layout model)
                direct_ckpts = [p for p in model_path.glob("checkpoint_*") if p.is_dir()]
                if direct_ckpts:
                    active_model_dir = model_path
                    layout = meta.get("layout_name", layout_name)
                    if layout not in ("all_layouts", "all_maps", layout_name):
                        try:
                            trained_shape = OvercookedGridworld.from_layout_name(layout).shape
                            current_shape = mdp.shape
                            if trained_shape != current_shape:
                                warning = (
                                    f"Custom agent '{model_path.name}' trained on '{layout}' "
                                    f"({trained_shape[0]}x{trained_shape[1]}) is incompatible with "
                                    f"'{layout_name}' ({current_shape[0]}x{current_shape[1]}). Using Greedy agent."
                                )
                                return _create_fallback_greedy_agent(mdp, agent_index), warning
                        except Exception:
                            pass
                        warning = f"Custom agent '{model_path.name}' was trained on '{layout}', not '{layout_name}'."
                else:
                    warning = f"Custom agent '{model_path.name}' has no checkpoint for '{layout_name}'. Using Greedy agent."
                    return _create_fallback_greedy_agent(mdp, agent_index), warning

            checkpoint_dirs = sorted(p for p in active_model_dir.glob("checkpoint_*") if p.is_dir())
            if not checkpoint_dirs:
                warning = f"No checkpoint directory found in {active_model_dir}. Using Greedy agent."
                return _create_fallback_greedy_agent(mdp, agent_index), warning
            latest_ckpt_dir = checkpoint_dirs[-1]
            ckpt_files = [p for p in latest_ckpt_dir.glob("checkpoint-*") if p.is_file() and not p.name.endswith(".tune_metadata")]
            if not ckpt_files:
                warning = f"No checkpoint file found in {latest_ckpt_dir}. Using Greedy agent."
                return _create_fallback_greedy_agent(mdp, agent_index), warning

            try:
                with suppress_c_stderr():
                    from training.train_agent import build_trainer
                    from human_aware_rl.rllib.rllib import RlLibAgent

                    trainer_key = f"{str(model_path.resolve())}_{layout}"
                    if trainer_key not in _TRAINER_CACHE:
                        partner_type = meta.get("partner_type", "greedy")
                        partner_prob = meta.get("partner_prob", None)
                        trainer, _ = build_trainer(
                            recipe=recipe,
                            layout_name=layout,
                            partner_type=partner_type,
                            partner_prob=partner_prob,
                            num_workers=0,
                        )
                        trainer.restore(str(ckpt_files[0]))
                        _TRAINER_CACHE[trainer_key] = trainer

                    trainer = _TRAINER_CACHE[trainer_key]
                    policy = trainer.get_policy("ppo")
                    featurize_fn = lambda state, debug=False: mdp.lossless_state_encoding(state, 400)
                    agent = RlLibAgent(policy, agent_index=agent_index, featurize_fn=featurize_fn)
                    agent.set_agent_index(agent_index)
                    agent.set_mdp(mdp)
                    agent.reset()
                    agent.set_agent_index(agent_index)
                    agent.set_mdp(mdp)
                return agent, warning
            except Exception as e:
                warning = f"Failed to load custom agent '{model_path.name}': {e}. Using Greedy agent."
                return _create_fallback_greedy_agent(mdp, agent_index), warning
        else:
            try:
                with suppress_c_stderr():
                    from experiments.run_pretrained_policy_comparison import load_trainer
                    from human_aware_rl.rllib.rllib import get_agent_from_trainer
                    trainer = load_trainer(model_path)
                    agent = get_agent_from_trainer(trainer, policy_id="ppo", agent_index=agent_index)
                    agent.set_agent_index(agent_index)
                    agent.set_mdp(mdp)
                    agent.reset()
                    agent.set_agent_index(agent_index)
                    agent.set_mdp(mdp)
                return agent, None
            except Exception as e:
                warning = f"Failed to load agent '{model_path.name}': {e}. Using Greedy agent."
                return _create_fallback_greedy_agent(mdp, agent_index), warning

    else:
        # Check if agent_type corresponds to an entry in AgentDatabase or a folder in models/
        models_dir = REPO_ROOT / "models"
        resolved_path = None
        target_dir = models_dir / agent_type
        if target_dir.is_dir() and (any(target_dir.glob("checkpoint_*")) or any(target_dir.rglob("checkpoint_*")) or (target_dir / "metadata.json").is_file()):
            resolved_path = target_dir.resolve()
        else:
            try:
                from gui.agent_database import AgentDatabase
                for r in AgentDatabase.load():
                    if agent_type.lower() in (r.get("id", "").lower(), r.get("name", "").lower()):
                        ckpt = r.get("checkpoint_path")
                        if ckpt and Path(ckpt).exists():
                            resolved_path = Path(ckpt).resolve()
                            break
            except Exception:
                pass

        if resolved_path is not None:
            return create_agent(f"custom:{resolved_path}", agent_index=agent_index, mdp=mdp, layout_name=layout_name)

        raise ValueError(f"Unknown agent type: {agent_type}")

    agent.set_agent_index(agent_index)
    agent.set_mdp(mdp)
    return agent, warning


def get_custom_trained_agents() -> list[tuple[str, str]]:
    """Scan models/ directory for custom trained agent checkpoints."""
    models_dir = REPO_ROOT / "models"
    if not models_dir.is_dir():
        return []

    results: list[tuple[str, str]] = []
    for p in sorted(models_dir.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        has_checkpoint = (
            any(p.glob("checkpoint_*"))
            or any(p.rglob("checkpoint_*"))
            or (p / "config.pkl").is_file()
            or (p / "metadata.json").is_file()
        )
        if has_checkpoint:
            name = p.name
            meta_file = p / "metadata.json"
            is_all_layouts = False
            if meta_file.is_file():
                try:
                    import json
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("layout_name") in ("all_layouts", "all_maps"):
                        is_all_layouts = True
                except Exception:
                    pass

            map_tag = " (All Maps)" if is_all_layouts else ""
            if name in ("ppo_greedy", "ppo greedy"):
                label = f"PPO Greedy{map_tag}"
            elif name in ("ppo_v_greedy", "ppo v greedy", "ppo_50_50"):
                label = f"PPO v Greedy{map_tag}"
            elif "ppo_sp" in name:
                label = f"PPO Custom SP{map_tag}"
            else:
                label = f"PPO ({name}){map_tag}"
            results.append((label, f"custom:{p.resolve()}"))
    return results

