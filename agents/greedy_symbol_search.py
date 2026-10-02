import random
from typing import List, Tuple, Optional, Set

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action, Direction
from .utils import a_star_search, DIRECTIONS


class GreedySymbolSearchAgent(Agent):
    """An upgraded rule-based agent for Overcooked-AI featuring:
    
    1. Dynamic Obstacle Marking: Treats partner chef's position as a temporary wall.
    2. Complete Recipe FSM: Executes the full 4-stage cooking lifecycle
       (Fetch Onion -> Put in Pot -> Fetch Plate -> Dish & Serve Soup).
    3. Orientation-Aware A* State: Searches over (x, y, orientation) so the agent
       arrives facing target counters ready for instant INTERACT.
    4. Task Allocation & Conflict Avoidance: Dynamically switches goals if partner
       is closer or already executing a matching task.
    5. A* Search: Uses priority queue with Manhattan distance + turn penalty heuristic.
    """

    def __init__(self):
        super().__init__()

    def _get_partner_info(self, state):
        """Extract partner chef's position, orientation, and held object."""
        if len(state.players) <= 1:
            return None, None, None
        
        partner_idx = 1 - self.agent_index if self.agent_index in (0, 1) else (self.agent_index + 1) % len(state.players)
        partner = state.players[partner_idx]
        p_pos = partner.position
        p_orient = partner.orientation
        p_held = partner.held_object.name if partner.has_object() else None
        return p_pos, p_orient, p_held

    def _select_target_counter(self, state, obstacles: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Recipe Finite State Machine (FSM) with Task Allocation & Conflict Avoidance."""
        our_player = state.players[self.agent_index]
        our_pos = our_player.position
        our_held = our_player.held_object.name if our_player.has_object() else None

        partner_pos, partner_orient, partner_held = self._get_partner_info(state)

        pots = self.mdp.get_pot_locations()
        onion_dispensers = self.mdp.get_onion_dispenser_locations()
        dish_dispensers = self.mdp.get_dish_dispenser_locations()
        serving_counters = self.mdp.get_serving_locations()

        # Categorize pots
        ready_pots = []
        cooking_pots = []
        needy_pots = []  # pots that still need onions

        for p in pots:
            pot_obj = state.objects.get(p)
            if pot_obj is None:
                needy_pots.append(p)
            elif pot_obj.is_ready:
                ready_pots.append(p)
            elif pot_obj.is_cooking:
                cooking_pots.append(p)
            elif pot_obj.is_idle:
                # Pot has 1 or 2 onions, still needs more
                if len(pot_obj.ingredients) < 3:
                    needy_pots.append(p)
                else:
                    cooking_pots.append(p)

        # ---------------------------------------------------------------------
        # STAGE 4: DISH & SERVE (Holding Soup)
        # ---------------------------------------------------------------------
        if our_held == "soup":
            if serving_counters:
                return serving_counters[0]

        # ---------------------------------------------------------------------
        # STAGE 3: DISH SOUP (Holding Plate / Dish)
        # ---------------------------------------------------------------------
        if our_held == "dish":
            if ready_pots:
                return ready_pots[0]
            # If no pot is ready yet, wait near pot or non-full pot
            if pots:
                return pots[0]

        # ---------------------------------------------------------------------
        # STAGE 2: POT ONION (Holding Onion)
        # ---------------------------------------------------------------------
        if our_held == "onion":
            if needy_pots:
                return needy_pots[0]
            # If all pots are cooking/ready, target closest pot
            if pots:
                return pots[0]

        # ---------------------------------------------------------------------
        # STAGE 1: FETCH INGREDIENT OR PLATE (Not holding anything)
        # ---------------------------------------------------------------------
        if our_held is None:
            # Task Allocation 1: If soup is ready or currently cooking, AND partner isn't already dishing, fetch a plate!
            if (ready_pots or cooking_pots) and partner_held != "dish" and dish_dispensers:
                return dish_dispensers[0]

            # Task Allocation 2: Otherwise, fetch an onion!
            if onion_dispensers:
                if len(onion_dispensers) == 1 or partner_pos is None:
                    return onion_dispensers[0]

                # Conflict Avoidance: If multiple onion dispensers, pick the one partner is NOT closest to
                def manhattan(p1, p2):
                    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

                d0_our = manhattan(our_pos, onion_dispensers[0])
                d0_part = manhattan(partner_pos, onion_dispensers[0])
                
                # If partner is closer to dispenser 0, target dispenser 1 (if available)
                if d0_part < d0_our and len(onion_dispensers) > 1:
                    return onion_dispensers[1]
                return onion_dispensers[0]

            if dish_dispensers:
                return dish_dispensers[0]

        return None

    def action(self, state) -> tuple[Action, dict]:
        """Core Agent API method returning (primitive_action, info_dict)."""
        our_player = state.players[self.agent_index]
        start_pos = our_player.position
        start_orient = our_player.orientation

        # 1. Dynamic Obstacle Marking: Treat partner position as obstacle
        partner_pos, _, _ = self._get_partner_info(state)
        obstacles = {partner_pos} if partner_pos is not None else set()

        # 2. Select Target Counter via Recipe FSM & Task Allocation
        target_counter = self._select_target_counter(state, obstacles)
        if target_counter is None:
            return random.choice([Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]), {}

        # 3. Orientation-Aware A* Search (with partner as obstacle)
        path = a_star_search(
            self.mdp.terrain_mtx,
            start_pos,
            start_orient,
            target_counter,
            obstacles=obstacles
        )

        # Fallback: If path is blocked by partner obstacle, retry search without partner obstacle
        if path is None and obstacles:
            path = a_star_search(
                self.mdp.terrain_mtx,
                start_pos,
                start_orient,
                target_counter,
                obstacles=set()
            )

        # If still no path found, wander or stay
        if path is None:
            return Direction.NORTH, {}

        # 4. Execute Path Step or INTERACT
        if len(path) == 0:
            # We are already at target walkable cell facing the counter -> INTERACT!
            return Action.INTERACT, {}

        next_action = path[0]
        return next_action, {}
