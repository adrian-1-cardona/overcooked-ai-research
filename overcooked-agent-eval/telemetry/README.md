# Telemetry Schema and Reproducibility Manifest

The telemetry CSV uses one row per environment timestep. This is the stable input format for future metrics and analysis code. `TelemetryRow` validates values while the experiment runs, and `TelemetryLogger.validate_csv()` validates the saved header and every saved row.

In addition, every experiment run writes a companion JSON reproducibility manifest (`*.manifest.json`) recording the complete execution environment, parameters, seeds, and version metadata (Issue #14).

---

## Missing values and encoding

- Held objects use `none` when an agent has empty hands.
- Event fields use an empty string when no event occurred. Multiple events are sorted and separated with semicolons.
- Pre-action and post-action positions are JSON coordinates such as `[1, 2]`, which avoids Python-specific tuple syntax and supports movement metrics.
- Actions and orientations use `north`, `south`, `east`, `west`, `stay`, or `interact` where applicable.
- Boolean values are written as `True` or `False`.
- All fields are required. Event fields are the only fields that may be empty; there is no generic null value.

---

## Telemetry Field Reference (27 Fields)

| Field | Type | Meaning |
| --- | --- | --- |
| `run_id` | string | Deterministic ID derived from the full experiment configuration. |
| `episode_id` | integer | One-based episode number within the run. |
| `episode_seed` | integer | Seed used for this episode; the base seed plus `episode_id - 1`. |
| `timestep` | integer | One-based timestep after the recorded joint action. |
| `layout_name` | string | Overcooked-AI layout used for the episode. |
| `agent_0_id` | integer | Stable player index `0` for the first agent. |
| `agent_1_id` | integer | Stable player index `1` for the second agent. |
| `agent_0_name` | string | Strategy name for player 0. |
| `agent_1_name` | string | Strategy name for player 1. |
| `agent_0_action` | string | Action selected by player 0. |
| `agent_1_action` | string | Action selected by player 1. |
| `reward` | number | Team sparse reward returned by the environment for this timestep. |
| `agent_0_sparse_reward` | number | Player 0 contribution to timestep sparse reward. |
| `agent_1_sparse_reward` | number | Player 1 contribution to timestep sparse reward. |
| `agent_0_shaped_reward` | number | Player 0 timestep reward-shaping value. |
| `agent_1_shaped_reward` | number | Player 1 timestep reward-shaping value. |
| `done` | boolean | Whether the episode ended after this timestep. |
| `agent_0_previous_position` | JSON array | Player 0 `[x, y]` position before the action. |
| `agent_1_previous_position` | JSON array | Player 1 `[x, y]` position before the action. |
| `agent_0_position` | JSON array | Player 0 `[x, y]` position after the action. |
| `agent_1_position` | JSON array | Player 1 `[x, y]` position after the action. |
| `agent_0_orientation` | string | Player 0 orientation after the action. |
| `agent_1_orientation` | string | Player 1 orientation after the action. |
| `agent_0_held_object` | string | Object held by player 0, or `none`. |
| `agent_1_held_object` | string | Object held by player 1, or `none`. |
| `agent_0_events` | string | Semicolon-separated Overcooked events attributed to player 0. |
| `agent_1_events` | string | Semicolon-separated Overcooked events attributed to player 1. |

---

## Reproducibility Manifest Format (Issue #14)

Beside every telemetry CSV, a JSON manifest is generated:

```json
{
  "agent_0_name": "RandomAgent",
  "agent_1_name": "RandomAgent",
  "base_seed": 42,
  "created_at": "2026-08-31T22:45:00.000000+00:00",
  "episode_seeds": [42, 43, 44, 45, 46],
  "episodes": 5,
  "evaluation_commit": "7d80d183f5abb4eb6f7867668653a087a1784458",
  "horizon": 400,
  "layout": "cramped_room",
  "output_telemetry_file": "results/multi_episode_random_baseline_cramped_room.csv",
  "overcooked_ai_version": "submodule-739950a079cdaed5a44fcc662efc40244c205d06",
  "python_version": "3.10.21",
  "run_id": "random-a1b2c3d4e5f60718",
  "schema_version": 2
}
```

---

## Telemetry CSV Example

```csv
run_id,episode_id,episode_seed,timestep,layout_name,agent_0_id,agent_1_id,agent_0_name,agent_1_name,agent_0_action,agent_1_action,reward,agent_0_sparse_reward,agent_1_sparse_reward,agent_0_shaped_reward,agent_1_shaped_reward,done,agent_0_previous_position,agent_1_previous_position,agent_0_position,agent_1_position,agent_0_orientation,agent_1_orientation,agent_0_held_object,agent_1_held_object,agent_0_events,agent_1_events
random-example,1,42,1,cramped_room,0,1,RandomAgent,RandomAgent,north,stay,0,0,0,0,0,False,"[1, 3]","[3, 2]","[1, 2]","[3, 2]",north,south,none,none,,
```

The runner validates the complete file immediately after saving it, so missing columns or malformed values fail clearly instead of producing silently unreliable data.
