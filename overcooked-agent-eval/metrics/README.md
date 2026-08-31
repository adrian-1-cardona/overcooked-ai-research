# Formal Specification: Episode Performance and Coordination Metrics

This document provides a methods-grade specification for every metric computed by the Overcooked agent evaluation framework. All metrics are computed strictly from saved per-timestep telemetry without rerunning the environment.

---

## 1. Metric Philosophy and Methodological Boundaries

### Observable Behavior vs. Intent and Blame
- **What these metrics measure:** Explicit physical state changes, positions, actions, held objects, and environment-emitted events recorded during gameplay.
- **What these metrics DO NOT measure:** Internal agent intent, blame, malice, or policy quality in isolation.
- **Why this distinction matters:** 
  - An explicit `stay` action might indicate an efficient agent waiting patiently for soup to finish cooking, or it might indicate an unresponsive policy.
  - A teammate blocking event indicates that physical pathing conflicted, not necessarily that the blocking agent was "at fault."
  - Held objects and task categories serve as **role proxies**, not assigned or intentional behavioral roles.
  - Evaluation of coordination quality requires analyzing coordination metrics alongside task performance (score and deliveries), layout structure, and partner compatibility.

---

## 2. Core Performance Metrics (Issue #2)

| Metric Field | Telemetry Inputs | Unit / Range | Mathematical Formula & Definition | Edge Cases / Handling |
|---|---|---|---|---|
| `team_score` | `reward` | Points ($\ge 0$) | $\sum_{t=1}^{T} \text{reward}_t$ — Total sparse reward accumulated by the team across the episode. | In standard Overcooked-AI, deliveries award 20 points each. |
| `soups_delivered` | `agent_0_events`, `agent_1_events` | Integer ($\ge 0$) | $\sum_{t=1}^{T} \sum_{i \in \{0, 1\}} \mathbb{I}(\text{"soup\_delivery"} \in \text{events}_{i, t})$ — Count of completed soup deliveries to the service counter. | Each delivery event emitted by the environment is counted exactly once. |
| `episode_length` | `timestep`, `done` | Timesteps ($T \ge 1$) | $T = \text{Count of contiguous timestep rows}$ | Verified: Timesteps must start at 1, increment contiguously, and have `done=True` only at $t=T$. |

---

## 3. Movement and Space Metrics (Issue #2)

| Metric Field | Telemetry Inputs | Unit / Range | Mathematical Formula & Definition | Edge Cases / Handling |
|---|---|---|---|---|
| `agent_i_distance_traveled` | `agent_i_previous_position`, `agent_i_position` | Grid steps ($\ge 0$) | $\sum_{t=1}^{T} \| \mathbf{p}_{i, t} - \mathbf{p}_{i, t-1} \|_1$ where $\mathbf{p} = [x, y]$. | Failed movement attempts result in $\mathbf{p}_{i, t} = \mathbf{p}_{i, t-1}$ and add 0 distance. |
| `team_distance_traveled` | Distance fields | Grid steps ($\ge 0$) | $D_{\text{team}} = D_{\text{agent\_0}} + D_{\text{agent\_1}}$ | Always strictly equal to the sum of individual agent distances. |
| `agent_i_idle_timesteps` | `agent_i_action` | Timesteps ($[0, T]$) | $\sum_{t=1}^{T} \mathbb{I}(a_{i, t} = \text{"stay"})$ | Only explicit `stay` actions count as idle. Failed directional moves or `interact` do not count as idle. |
| `team_idle_timesteps` | Idle timesteps | Timesteps ($[0, 2T]$) | $I_{\text{team}} = I_{\text{agent\_0}} + I_{\text{agent\_1}}$ | Sum of idle timesteps across both players. |
| `agent_i_idle_rate` | Idle timesteps, $T$ | Rate ($[0.0, 1.0]$) | $\frac{I_{\text{agent\_i}}}{T}$ | Proportion of episode timesteps where agent $i$ explicitly stayed. |
| `team_idle_rate` | Team idle, $T$ | Rate ($[0.0, 1.0]$) | $\frac{I_{\text{team}}}{2T}$ | Normalized across both agent opportunities ($2T$). |

---

## 4. Conflict, Blocking, and Interference Metrics (Issue #3)

Let:
- $\mathbf{p}_{i, t-1}$ be the pre-action position of agent $i$.
- $\mathbf{p}_{i, t}$ be the post-action position of agent $i$.
- $\mathbf{u}(a_{i, t})$ be the directional unit vector for action $a_{i, t} \in \{\text{north, south, east, west}\}$.
- Target cell $\mathbf{g}_{i, t} = \mathbf{p}_{i, t-1} + \mathbf{u}(a_{i, t})$.
- Movement failed $\iff \mathbf{p}_{i, t} = \mathbf{p}_{i, t-1}$.

| Metric Field | Telemetry Inputs | Unit / Range | Mathematical Formula & Definition | Edge Cases / Handling |
|---|---|---|---|---|
| `same_target_collision_events` | Actions, positions | Events ($\ge 0$) | $\mathbb{I}(\text{failed}_{0, t} \land \text{failed}_{1, t} \land \mathbf{g}_{0, t} = \mathbf{g}_{1, t})$ — Both agents attempted to step into the exact same destination cell. | Joint event counted once for the team. |
| `swap_collision_events` | Actions, positions | Events ($\ge 0$) | $\mathbb{I}(\text{failed}_{0, t} \land \text{failed}_{1, t} \land \mathbf{g}_{0, t} = \mathbf{p}_{1, t-1} \land \mathbf{g}_{1, t} = \mathbf{p}_{0, t-1})$ — Both agents attempted to step into each other's cell simultaneously. | Joint event counted once for the team. |
| `agent_i_collision_attempts` | Joint collisions | Attempts ($\ge 0$) | Attributed to each agent participating in a same-target or swap collision at step $t$. | Increments for both agents during joint collisions. |
| `team_collision_events` | Joint collisions | Events ($\ge 0$) | $\sum_{t=1}^{T} (\text{same\_target}_t + \text{swap}_t)$ | Total joint collision events across the episode. |
| `agent_i_blocked_by_teammate` | Actions, positions | Events ($\ge 0$) | $\mathbb{I}(\text{failed}_{i, t} \land \mathbf{g}_{i, t} = \mathbf{p}_{j, t-1} \land \mathbf{p}_{j, t} = \mathbf{p}_{j, t-1})$ where $j = 1 - i$ and step $t$ is not a joint collision. | Agent $i$ attempted to enter teammate $j$'s cell, but $j$ did not vacate. |
| `agent_j_blocking_teammate` | Actions, positions | Events ($\ge 0$) | Attributed to teammate $j$ whenever agent $i$ is blocked by $j$. | Complementary attribution of the blocking event. |
| `team_blocking_events` | Blocking events | Events ($\ge 0$) | Total directional moves blocked by an unvacated teammate. | $\sum_{t=1}^{T} (\text{blocked}_{0, t} + \text{blocked}_{1, t})$. |
| `agent_i_wall_collision_attempts` | Actions, positions | Attempts ($\ge 0$) | $\mathbb{I}(\text{failed}_{i, t} \land \neg \text{joint\_collision}_t \land \neg \text{blocked\_by\_teammate}_{i, t})$ — Movement failed due to terrain, counter, or outer boundary. | Residual failure category separating terrain from teammate interaction. |
| `agent_i_interference_timesteps` | Collisions, blocks | Timesteps ($[0, T]$) | $\sum_{t=1}^{T} \mathbb{I}(\text{collision}_t \lor \text{blocked}_{i, t} \lor \text{blocking}_{i, t})$ | Timesteps where agent $i$ was involved in either a collision or teammate block. |
| `team_interference_timesteps` | Interferences | Timesteps ($[0, T]$) | $\sum_{t=1}^{T} \mathbb{I}(\text{collision}_t \lor \text{blocked}_{0, t} \lor \text{blocked}_{1, t})$ | Timesteps where team experienced movement interference (counted once per step). |
| `team_repeated_interference_timesteps` | Interferences | Timesteps ($[0, T]$) | Timesteps where team experienced interference at both $t$ and $t-1$. | Measures persistent gridlock or recurring spatial deadlock. |
| `team_interference_rate` | Team interference, $T$ | Rate ($[0.0, 1.0]$) | $\frac{\text{team\_interference\_timesteps}}{T}$ | Fraction of episode length affected by spatial conflicts. |

---

## 5. Held-Object and Task-Event Role Proxies (Issue #17)

### Held-Object Time-Shares
For each agent $i \in \{0, 1\}$ and object $k \in \{\text{none}, \text{onion}, \text{tomato}, \text{dish}, \text{soup}\}$:
- **Held Timesteps:** Count of timesteps where `agent_i_held_object == k`.
- **Time-Share:** $\text{share}_{i, k} = \frac{\text{held\_timesteps}_{i, k}}{T}$.
- **Invariant Guarantee:** For each agent $i$, $\sum_{k} \text{held\_timesteps}_{i, k} = T$ and $\sum_{k} \text{share}_{i, k} = 1.0$.

### Task-Event Counts and Rates
- `agent_i_potting_onion_count` / `agent_i_potting_tomato_count`: Number of ingredients placed into cooking pots by agent $i$.
- `agent_i_potting_event_count`: Total ingredients placed into pots by agent $i$.
- `team_potting_event_count`: Total ingredients placed into pots across both agents.
- `team_dish_pickup_count`: Total clean dishes picked up from dispensers or counters.
- `team_soup_pickup_count`: Total cooked soups ladled into dishes.
- `agent_i_soup_delivery_count`: Soups successfully delivered to the service station by agent $i$.
- `agent_i_potting_event_rate`: $\frac{\text{potting\_event\_count}_i}{T}$.
- `agent_i_soup_delivery_rate`: $\frac{\text{soup\_delivery\_count}_i}{T}$.

---

## 6. Task Duplication and Unused Pipeline Metrics (Issue #13)

### Task Classification Model
At each timestep $t$, an agent's observable state is classified into one of five mutually exclusive task categories:
1. **`gather`**: Holding `onion` or `tomato`, or executing `potting_onion`/`potting_tomato`/`onion_pickup`/`tomato_pickup`.
2. **`dish`**: Holding `dish`, or executing `dish_pickup`/`useful_dish_pickup`.
3. **`deliver`**: Holding `soup`, or executing `soup_pickup`/`soup_delivery`.
4. **`idle`**: Action is `stay` while holding `none`.
5. **`other`**: Moving empty-handed without active events.

### Task Duplication
- **Duplicate Gather Timesteps:** Timesteps where both agents are classified as `gather`.
- **Duplicate Dish Timesteps:** Timesteps where both agents are classified as `dish`.
- **Duplicate Deliver Timesteps:** Timesteps where both agents are classified as `deliver`.
- **Team Task Duplication Timesteps:** $\sum_{t=1}^{T} \mathbb{I}(\text{task}_{0, t} = \text{task}_{1, t} \land \text{task}_{0, t} \notin \{\text{"idle"}, \text{"other"}\})$.
- **Team Task Duplication Rate:** $\frac{\text{team\_task\_duplication\_timesteps}}{T}$.

### Unused Pipeline Work
- **Pipeline Active Condition:** A cooked soup has been picked up or is ready, or cooking is complete, establishing a pending delivery stage.
- **Unused Pipeline Timestep:** A timestep where the pipeline is active, yet neither agent is actively working on dish preparation or delivery (`task}_{0, t} \notin \{\text{"dish"}, \text{"deliver"}\} \land \text{task}_{1, t} \notin \{\text{"dish"}, \text{"deliver"}\}$).
- **Team Unused Pipeline Rate:** $\frac{\text{team\_unused\_pipeline\_timesteps}}{T}$.

---

## 7. Batch Aggregation Specification (Issue #15)

For a batch of $N$ episodes on layout $L$ with agent pairing $(A_0, A_1)$, every numeric metric $x$ is aggregated into:
1. **Sample Size ($N$):** Count of completed episodes in the batch ($N \ge 1$).
2. **Mean ($\mu$):** $\mu = \frac{1}{N} \sum_{e=1}^{N} x_e$.
3. **Sample Standard Deviation ($s$):** $s = \sqrt{\frac{1}{N-1} \sum_{e=1}^{N} (x_e - \mu)^2}$ for $N > 1$, or $0.0$ for $N = 1$.
4. **Minimum ($\min$):** $\min_{e} x_e$.
5. **Maximum ($\max$):** $\max_{e} x_e$.

The per-episode CSV remains the granular source of truth, while the batch aggregate provides statistical comparability across different pairings.

---

## 8. Comparative Coordination Case Example

The following table illustrates why score alone is insufficient: two hypothetical agent pairings achieve the identical score on `cramped_room`, yet exhibit fundamentally different coordination structures.

| Dimension | Team Alpha (Specialized Complementary) | Team Beta (Chaotic Duplication) |
|---|---|---|
| **Team Score** | **40.0** (2 deliveries) | **40.0** (2 deliveries) |
| **Soups Delivered** | 2 | 2 |
| **Agent 0 Held Onion Share** | 0.650 (Prep role) | 0.400 |
| **Agent 1 Held Dish/Soup Share** | 0.600 (Deliverer role) | 0.420 |
| **Task Duplication Rate** | **0.020** (Negligible redundant work) | **0.480** (Both chasing onions simultaneously) |
| **Team Blocking Events** | **4** (Fluid navigation) | **92** (Constant hallway bottlenecking) |
| **Team Interference Rate** | **0.015** | **0.310** |
| **Interpretation** | Clean pipeline specialization with fluid spatial coordination. | High spatial contention and redundant effort; score achieved despite severe coordination friction. |

---

## 9. Summary Fields in Episode Metrics CSV

Every row in `*_episode_metrics.csv` contains 68 columns:
1. `run_id`, `episode_id`, `episode_seed`, `layout_name`, `agent_0_id`, `agent_1_id`, `agent_0_name`, `agent_1_name`
2. `team_score`, `soups_delivered`, `episode_length`
3. `agent_0_distance_traveled`, `agent_1_distance_traveled`, `team_distance_traveled`
4. `agent_0_idle_timesteps`, `agent_1_idle_timesteps`, `team_idle_timesteps`
5. `agent_0_idle_rate`, `agent_1_idle_rate`, `team_idle_rate`
6. `agent_0_wall_collision_attempts`, `agent_1_wall_collision_attempts`, `team_wall_collision_attempts`
7. `agent_0_blocked_by_teammate`, `agent_1_blocked_by_teammate`, `agent_0_blocking_teammate`, `agent_1_blocking_teammate`, `team_blocking_events`
8. `agent_0_collision_attempts`, `agent_1_collision_attempts`, `team_collision_events`, `same_target_collision_events`, `swap_collision_events`
9. `agent_0_interference_timesteps`, `agent_1_interference_timesteps`, `team_interference_timesteps`
10. `agent_0_repeated_interference_timesteps`, `agent_1_repeated_interference_timesteps`, `team_repeated_interference_timesteps`
11. `agent_0_interference_rate`, `agent_1_interference_rate`, `team_interference_rate`
12. `agent_0_held_none_timesteps`, `agent_0_held_onion_timesteps`, `agent_0_held_tomato_timesteps`, `agent_0_held_dish_timesteps`, `agent_0_held_soup_timesteps`
13. `agent_1_held_none_timesteps`, `agent_1_held_onion_timesteps`, `agent_1_held_tomato_timesteps`, `agent_1_held_dish_timesteps`, `agent_1_held_soup_timesteps`
14. `agent_0_held_none_share`, `agent_0_held_onion_share`, `agent_0_held_tomato_share`, `agent_0_held_dish_share`, `agent_0_held_soup_share`
15. `agent_1_held_none_share`, `agent_1_held_onion_share`, `agent_1_held_tomato_share`, `agent_1_held_dish_share`, `agent_1_held_soup_share`
16. `agent_0_potting_onion_count`, `agent_1_potting_onion_count`, `team_potting_onion_count`
17. `agent_0_potting_tomato_count`, `agent_1_potting_tomato_count`, `team_potting_tomato_count`
18. `agent_0_potting_event_count`, `agent_1_potting_event_count`, `team_potting_event_count`
19. `agent_0_dish_pickup_count`, `agent_1_dish_pickup_count`, `team_dish_pickup_count`
20. `agent_0_soup_pickup_count`, `agent_1_soup_pickup_count`, `team_soup_pickup_count`
21. `agent_0_soup_delivery_count`, `agent_1_soup_delivery_count`, `team_soup_delivery_count`
22. `agent_0_potting_event_rate`, `agent_1_potting_event_rate`, `agent_0_soup_delivery_rate`, `agent_1_soup_delivery_rate`
23. `duplicate_gather_timesteps`, `duplicate_dish_timesteps`, `duplicate_deliver_timesteps`, `team_task_duplication_timesteps`, `team_task_duplication_rate`
24. `team_unused_pipeline_timesteps`, `team_unused_pipeline_rate`
