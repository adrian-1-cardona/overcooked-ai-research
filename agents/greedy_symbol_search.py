##Greedy Symbol‑Search Agent for Overcooked‑AI.
##It repeatedly:
##  1️⃣ Looks at which ingredients are still needed.
##  2️⃣ Runs a BFS (from utils.bfs_find_nearest) to the closest needed tile.
##  3️⃣ Converts the path into primitive move actions (N,S,E,W).
##  4️⃣ When on an ingredient, issues an INTERACT action to pick it up.
##  5️⃣ When carrying something and next to a serving plate, issues INTERACT to drop.
## If no reachable ingredient exists, the agent wanders randomly.

import random
from typing import List, Tuple

from .utils import bfs_find_nearest, plan_moves_from_path, get_needed_ingredients
from overcooked_ai_py.mdp.actions import Action, Direction

class GreedySymbolSearchAgent:
    def __init__(self, env):
        self.env = env
        self.carrying = None  # simple flag for what we hold

    def _ingredients_positions(self, state) -> List[Tuple[int, int]]:
        """Return a list of grid cells that still contain needed ingredients.
        This wrapper uses the generic helper ``get_needed_ingredients`` which
        expects the state to expose an ``ingredients`` attribute.  If the repo
        uses a different field you can adjust this method accordingly.
        """
        return get_needed_ingredients(state)

    def act(self, obs):
        """Return a **single** primitive action for the current timestep.
        ``obs`` is the raw Observation object supplied by the Overcooked
        environment.  We extract the underlying game state via ``obs.state``
        (the standard attribute in the upstream repo).
        """
        state = obs.state
        # 1️⃣  What do we still need?
        needed = self._ingredients_positions(state)
        if not needed:
            # nothing left – just wander
            return random.choice([Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.MOVE_EAST, Action.MOVE_WEST])

        # 2️⃣  Find closest needed tile with BFS
        start = obs.agent_pos  # (row, col) of our chef
        path = bfs_find_nearest(self.env.layout.graph, start, needed)
        if path is None:
            # unreachable -> wander
            return random.choice([Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.MOVE_EAST, Action.MOVE_WEST])

        # 3️⃣  Convert path to primitive moves (ignore first element – it's our position)
        if len(path) > 1:
            move_seq = plan_moves_from_path(path)
            # return the *first* move in the sequence for this timestep
            move_idx = move_seq[0]
            return [Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.MOVE_EAST, Action.MOVE_WEST][move_idx]

        # 4️⃣  We're already on the ingredient – interact to pick up
        if not self.carrying:
            self.carrying = True
            return Action.INTERACT

        # 5️⃣  If we have something, try to drop on nearest plate (simplified)
        #    Here we just issue INTERACT again – the environment will handle the
        #    valid drop if we are adjacent to a plate.
        return Action.INTERACT

