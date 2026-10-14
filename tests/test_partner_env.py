"""Test OvercookedPartnerMultiAgent environment."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_EVAL_DIR = REPO_ROOT / "overcooked-agent-eval"
EXTERNAL_OVERCOOKED = REPO_ROOT / "external" / "overcooked_ai" / "src"

for p in (REPO_ROOT, AGENT_EVAL_DIR, EXTERNAL_OVERCOOKED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from training.env_wrapper import OvercookedPartnerMultiAgent
from training.greedy_policy import GreedyRllibPolicy, WorkerStateContext


def test_partner_multiagent():
    print("Testing OvercookedPartnerMultiAgent...")
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    base_env = OvercookedEnv.from_mdp(mdp, horizon=400)

    env = OvercookedPartnerMultiAgent(
        base_env=base_env,
        partner_type="greedy",
        partner_prob=1.0,
    )

    obs = env.reset()
    print(f"Agents populated: {env.curr_agents}")
    assert any(a.startswith("ppo") for a in env.curr_agents)
    assert any(a.startswith("greedy") for a in env.curr_agents)

    # Instantiate greedy policy
    greedy_policy = GreedyRllibPolicy(env.greedy_observation_space, env.shared_action_space, {})

    # Compute actions
    actions_dict = {}
    for agent_id in env.curr_agents:
        if agent_id.startswith("ppo"):
            actions_dict[agent_id] = Action.ACTION_TO_INDEX[Direction.NORTH]
        else:
            act, _, _ = greedy_policy.compute_actions([obs[agent_id]])
            actions_dict[agent_id] = int(act[0])

    print(f"Action dict: {actions_dict}")
    next_obs, rewards, dones, infos = env.step(actions_dict)
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
    print("OvercookedPartnerMultiAgent test passed!")


if __name__ == "__main__":
    test_partner_multiagent()
