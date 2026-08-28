# Episode performance and coordination metrics

The metrics command reads a completed telemetry CSV and writes one summary row per episode. It does not rerun Overcooked-AI. Each summary keeps `run_id` and `episode_id`, so it can be joined back to the source telemetry.

## Run the metrics pipeline

From `overcooked-agent-eval/`:

```bash
python experiments/summarize_episode_metrics.py
```

The default input is `results/multi_episode_random_baseline_cramped_room.csv`. The default output is `results/multi_episode_random_baseline_cramped_room_episode_metrics.csv`. Use `--input` and `--output` for other saved runs.

## Core performance definitions

- **Team score** is the sum of the timestep team reward for the episode.
- **Soups delivered** is the number of `soup_delivery` events attributed to either player. Each delivery event is counted once.
- **Episode length** is the number of contiguous telemetry rows. A valid episode starts at timestep 1 and has `done=True` only on its final row.
- **Distance traveled** is the sum of actual Manhattan distance between each pre-action and post-action position. Failed moves add zero distance.
- **Idle time** counts explicit `stay` actions. A failed directional move is not idle, and `interact` is not idle. Per-agent idle rate divides by episode length; team idle rate divides total agent-idle timesteps by twice the episode length.

## Coordination definitions

- **Wall/terrain collision attempt** means a directional action did not change the agent’s position and the failure was not explained by a teammate block, same-target collision, or swap. This includes counters, walls, and other non-teammate movement constraints.
- **Blocked by teammate** means an agent tried to move into the teammate’s pre-action cell, the teammate did not vacate that cell, and the movement failed. The summary records both the blocked agent and the teammate that caused the block.
- **Same-target collision** means both agents selected directional actions toward the same destination and both failed to move.
- **Swap collision** means both agents tried to enter each other’s pre-action cell and both failed to move.
- **Collision attempt** is attributed to both agents involved in a same-target or swap collision. A team collision event counts the joint event once.
- **Interference timestep** means an agent participated in a teammate block or joint collision, either as the blocked agent or the blocking teammate. A team interference timestep counts the timestep once even when both agents are involved.
- **Repeated interference** means interference involvement on consecutive timesteps. The first timestep in a sequence is interference but is not repeated interference.

## Important limitations

These are observable behavior measures, not claims about agent intent. A block may be strategically reasonable, and an explicit `stay` may be useful waiting. The wall/terrain category is a residual category because telemetry does not encode the map tile that rejected movement. Counts should be compared with score, deliveries, layout, and agent role rather than treated as standalone evidence that an agent is bad at coordination.

## Summary fields

The output includes run, episode, seed, layout, player, and agent identifiers; team score; soups delivered; episode length; per-agent and team distance and idle values; per-agent and team wall, blocking, collision, interference, and repeated-interference counts; same-target and swap event counts; and idle/interference rates.
