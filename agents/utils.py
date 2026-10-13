"""
Utility helpers for rule-based agents in Overcooked-AI.
Provides orientation-aware A* search and grid path planning functions.
"""
import heapq
from typing import List, Tuple, Optional, Set

NORTH = (0, -1)
SOUTH = (0, 1)
EAST = (1, 0)
WEST = (-1, 0)
DIRECTIONS = [NORTH, SOUTH, EAST, WEST]


def a_star_search(
    terrain_mtx: List[List[str]],
    start_pos: Tuple[int, int],
    start_orient: Tuple[int, int],
    target_counter_pos: Tuple[int, int],
    obstacles: Optional[Set[Tuple[int, int]]] = None
) -> Optional[List[Tuple[int, int]]]:
    """Orientation-aware A* search over the Overcooked gridworld.

    State space: (x, y, orientation).
    Target: standing adjacent to target_counter_pos AND facing target_counter_pos.
    Path cost g(n): 1 per translation step, 1 per orientation turn.
    Heuristic h(n): Manhattan distance to target_counter_pos + turn penalty.
    """
    if obstacles is None:
        obstacles = set()

    tx, ty = target_counter_pos
    height = len(terrain_mtx)
    width = len(terrain_mtx[0])

    def is_walkable(x: int, y: int) -> bool:
        if 0 <= x < width and 0 <= y < height:
            if (x, y) in obstacles:
                return False
            return terrain_mtx[y][x] == ' '
        return False

    def heuristic(x: int, y: int, orient: Tuple[int, int]) -> int:
        dist = abs(x - tx) + abs(y - ty)
        if (x + orient[0], y + orient[1]) != (tx, ty):
            dist += 1
        return dist

    start_state = (start_pos[0], start_pos[1], start_orient)
    h_start = heuristic(start_pos[0], start_pos[1], start_orient)
    pq = [(h_start, 0, start_state, [])]
    visited = {}

    while pq:
        f, g, (x, y, orient), path = heapq.heappop(pq)

        # Goal check: standing at walkable cell adjacent to target_counter_pos and facing it
        if (x + orient[0], y + orient[1]) == (tx, ty):
            return path

        state_key = (x, y, orient)
        if state_key in visited and visited[state_key] <= g:
            continue
        visited[state_key] = g

        # 1. Forward movement in current orientation
        nx, ny = x + orient[0], y + orient[1]
        if is_walkable(nx, ny):
            next_state = (nx, ny, orient)
            if next_state not in visited or visited[next_state] > g + 1:
                h = heuristic(nx, ny, orient)
                heapq.heappush(pq, (g + 1 + h, g + 1, next_state, path + [orient]))

        # 2. Orientation turns (staying in same cell)
        for d in DIRECTIONS:
            if d != orient:
                next_state = (x, y, d)
                if next_state not in visited or visited[next_state] > g + 1:
                    h = heuristic(x, y, d)
                    heapq.heappush(pq, (g + 1 + h, g + 1, next_state, path + [d]))

    return None
