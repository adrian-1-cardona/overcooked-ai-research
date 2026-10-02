"""
Utility helpers for simple rule‑based agents.
Provides a minimal BFS implementation that works with the Overcooked
environment map format (a 2‑D list of ints where 0 = empty, 1 = wall).
The functions are deliberately lightweight – just enough for the
GreedySymbolSearchAgent used in the demo.
"""
import collections
from typing import List, Tuple, Optional

def bfs_find_nearest(grid: List[List[int]], start: Tuple[int, int], targets: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    """Return a shortest path from *start* to the closest position in *targets*.
    If no reachable target exists, returns ``None``.
    The path is a list of grid coordinates, **including** the start cell.
    """
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    queue = collections.deque()
    queue.append((start, [start]))
    visited[start[0]][start[1]] = True
    target_set = set(targets)
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) in target_set:
            return path
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None

def plan_moves_from_path(path: List[Tuple[int, int]]) -> List[int]:
    """Convert a coordinate path into primitive Overcooked action indices.
    The Overcooked action space (in the upstream repo) uses the following
    integer mapping:
        0 – ``Direction.NORTH``
        1 – ``Direction.SOUTH``
        2 – ``Direction.EAST``
        3 – ``Direction.WEST``
        4 – ``Action.INTERACT``
        5 – ``Action.STAY``
    This helper only returns movement actions; ``INTERACT`` and ``STAY``
    are added by the agent logic when needed.
    """
    moves = []
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if r2 == r1 - 1:
            moves.append(0)  # NORTH
        elif r2 == r1 + 1:
            moves.append(1)  # SOUTH
        elif c2 == c1 + 1:
            moves.append(2)  # EAST
        elif c2 == c1 - 1:
            moves.append(3)  # WEST
        else:
            raise ValueError("Non‑adjacent steps in path")
    return moves

def get_needed_ingredients(state) -> List[Tuple[int, int]]:
    """Extract the positions of still‑needed ingredients from an Overcooked state.
    The concrete implementation depends on the exact state object used by the
    repo (``state.ingredients`` or similar).  Here we provide a placeholder that
    looks for a ``.ingredients`` attribute containing a list of ``(row, col)``
    tuples.  If the attribute is missing we simply return an empty list – the
    greedy agent will then wander.
    """
    if hasattr(state, "ingredients"):
        return list(state.ingredients)
    return []
