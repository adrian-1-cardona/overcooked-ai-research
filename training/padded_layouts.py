"""
Padded 9x5 representations of the canonical Overcooked-AI benchmark layouts.

All layouts are padded to 9 columns x 5 rows with impassable counter tiles ('X').
This preserves the exact internal coordinate geometry, movement graph, and
start positions while standardizing the spatial observation tensor to (9, 5, C).
"""

from __future__ import annotations

from typing import Dict, List
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

# Canonical layouts padded to width=9 (cols=9), height=5 (rows=5)
PADDED_LAYOUT_GRIDS: Dict[str, List[str]] = {
    "cramped_room": [
        "XXPXXXXXX",
        "O  2OXXXX",
        "X1  XXXXX",
        "XDXSXXXXX",
        "XXXXXXXXX",
    ],
    "asymmetric_advantages": [
        "XXXXXXXXX",
        "O XSXOX S",
        "X   P 1 X",
        "X2  P   X",
        "XXXDXDXXX",
    ],
    "coordination_ring": [
        "XXXPXXXXX",
        "X 1 PXXXX",
        "D2X XXXXX",
        "O   XXXXX",
        "XOSXXXXXX",
    ],
    "forced_coordination": [
        "XXXPXXXXX",
        "O X1PXXXX",
        "O2X XXXXX",
        "D X XXXXX",
        "XXXSXXXXX",
    ],
    "counter_circuit_o_1order": [
        "XXXPPXXXX",
        "X  2   XX",
        "D XXXX SX",
        "X  1   XX",
        "XXXOOXXXX",
    ],
}

_CACHED_MDPS: Dict[str, OvercookedGridworld] = {}


def get_padded_mdp(layout_name: str) -> OvercookedGridworld:
    """Return a cached OvercookedGridworld instance for the padded 9x5 layout."""
    if layout_name not in PADDED_LAYOUT_GRIDS:
        raise ValueError(
            f"Unknown layout '{layout_name}'. Available padded layouts: {list(PADDED_LAYOUT_GRIDS.keys())}"
        )
    if layout_name not in _CACHED_MDPS:
        grid = PADDED_LAYOUT_GRIDS[layout_name]
        _CACHED_MDPS[layout_name] = OvercookedGridworld.from_grid(grid)
    return _CACHED_MDPS[layout_name]


def get_all_padded_mdps() -> Dict[str, OvercookedGridworld]:
    """Return all 5 padded layout MDPs."""
    return {name: get_padded_mdp(name) for name in PADDED_LAYOUT_GRIDS}


def precompute_planners() -> None:
    """Precompute MLAM and MotionPlanner in the driver process to prevent worker pickle race conditions."""
    from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
    for name, mdp in get_all_padded_mdps().items():
        try:
            MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)
        except Exception:
            MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=True)

