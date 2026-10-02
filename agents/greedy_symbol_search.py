import random
from typing import List, Tuple

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action

# Helper utilities from our local agents package
from .utils import bfs_find_nearest, plan_moves_from_path, get_needed_ingredients

class GreedySymbolSearchAgent(Agent):
    """Greedy symbol‑search agent for Overcooked‑AI.

    The agent repeatedly:
    1. Finds which ingredients are still needed.
    2. Uses BFS to locate the nearest needed ingredient.
    3. Moves toward it using primitive actions (N/S/E/W).
    4. Interacts to pick up an ingredient when on the tile.
    5. Interacts again to drop the held item (the environment will handle
       the drop if a serving plate is adjacent).
    """

    def __init__(self):
        super().__init__()
        self.carrying: bool = False

    # ---------------------------------------------------------------------
    # Helper to obtain positions of still‑needed ingredients.
    # ---------------------------------------------------------------------
    def _ingredients_positions(self, state) -> List[Tuple[int, int]]:
        """Return a list of (row, col) coordinates for needed ingredients.

        The repository provides ``get_needed_ingredients`` which extracts those
        tiles from the ``state`` object. Adjust this method if the state
        representation changes.
        """
        return get_needed_ingredients(state)

    # ---------------------------------------------------------------------
    # Core Agent API method.
    # ---------------------------------------------------------------------
    def action(self, state) -> tuple[Action, dict]:
        """Return a primitive ``Action`` for the current timestep.

        ``state`` is an ``OvercookedState`` instance supplied by the environment.
        The method returns ``(action, {})`` – an empty info dictionary satisfies
        the ``Agent`` interface used throughout the codebase.
        """
        # 1️⃣ Determine which ingredients are still needed.
        needed = self._ingredients_positions(state)
        if not needed:
            # No work left – wander randomly.
            return random.choice([
                Action.MOVE_NORTH,
                Action.MOVE_SOUTH,
                Action.MOVE_EAST,
                Action.MOVE_WEST,
            ]), {}

        # 2️⃣ Locate this agent's current position in the layout graph.
        start = self.mdp.layout.start_positions[self.agent_index]

        # 3️⃣ Find the shortest path to the nearest needed tile.
        path = bfs_find_nearest(self.mdp.layout.graph, start, needed)
        if path is None:
            # Unreachable – fall back to random wandering.
            return random.choice([
                Action.MOVE_NORTH,
                Action.MOVE_SOUTH,
                Action.MOVE_EAST,
                Action.MOVE_WEST,
            ]), {}

        # 4️⃣ Convert the path to a primitive move (ignore the first node – our position).
        if len(path) > 1:
            move_seq = plan_moves_from_path(path)
            move_idx = move_seq[0]
            primitive = [
                Action.MOVE_NORTH,
                Action.MOVE_SOUTH,
                Action.MOVE_EAST,
                Action.MOVE_WEST,
            ][move_idx]
            return primitive, {}

        # 5️⃣ We're on a needed ingredient – pick it up.
        if not self.carrying:
            self.carrying = True
            return Action.INTERACT, {}

        # 6️⃣ Already holding something – attempt to drop (simplified).
        return Action.INTERACT, {}
