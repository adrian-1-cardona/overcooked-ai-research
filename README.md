# Overcooked-AI: Cooperative Multi-Agent Coordination & Evaluation Framework

Overcooked-AI is a research framework for studying emergent collective behavior, human-aware reinforcement learning, and partner compatibility in decentralized cooperative multi-agent environments. Two chefs in a 2D gridworld kitchen must coordinate under severe spatial bottlenecks and physical task asymmetries to prepare, cook, plate, and serve soup dishes before time expires. Coordination is evaluated not merely through total score, but across spatial handoffs, role division, partner compatibility, and deadlock resilience.

Builds on Carroll et al. (*NeurIPS 2019*). Senior project research by Adrian Cardona under the guidance of Prof. Rodrigo Canaan at California Polytechnic State University, San Luis Obispo. Active development 2026.

### Key design:
- **DeepMind Frame Stacking ($k=4$, 104 channels)** — Four-frame observation histories provide temporal velocity and intent context to feed-forward CNNs without the training instability of recurrent LSTMs.
- **Human-Aware RL vs. Symmetric Self-Play** — Rigorous benchmark comparing `ppo[fr,self]` against `ppo[fr,bc]` (trained alongside Behavior Cloning human models on real player trajectories).
- **Interactive Pygame Simulator (`runover.py`)** — Real-time gameplay renderer supporting human-in-the-loop play, live AI watch mode, and dynamic seat swapping (Red Hat Chef 0 vs. Blue Hat Chef 1).
- **Streamlit Coordination Matrix Dashboard** — Non-square cross-play evaluation matrix, side-by-side performance diffs ($\Delta$), and custom agent row/column filtering.
- **Stateless Simulation & Pure Contracts** — Deterministic action handling with clean architectural separation between base game state and agent inference adapters.

---

## Setup

```bash
./setup.sh                                   # auto-detects environment and installs dependencies
source overcooked-agent-eval/.venv/bin/activate
pip install -e ./external/overcooked_ai
```

Requires Python 3.10 (`>=3.10,<3.11`). For proposal details and methodology, see [`project_docs/project_proposal.md`](project_docs/project_proposal.md).

---

## Running

```bash
# Interactive GUI (Human Play / Watch Mode / Live Matrix):
python runover.py
```



## Structure

```
overcooked-ai-research/        ← repo root
│
├── runover.py                 ← CLI entry point & interactive Pygame GUI
├── setup.sh                   ← environment setup & submodule initialization
├── run_tests.sh               ← 5-stage test runner (contracts, evaluation, lifecycle)
│
├── gui/                       ← Pygame GUI & agent interaction
│   ├── app.py                 ← main application loop & live renderer
│   ├── agent_manager.py       ← agent factory, model loader, & alias resolver
│   ├── agent_database.py      ← persistent model registry CRUD manager
│   └── widgets.py             ← UI components (dropdowns, buttons, modal dialogs)
│
├── dashboard/                 ← Streamlit telemetry & matrix analytics
│   ├── coordination_matrix.py ← non-square matrix filtering & side-by-side diffs
│   └── run_telemetry.py       ← per-step movement and reward telemetry visualization
│
├── training/                  ← RLlib training engine & wrappers
│   ├── train_benchmark_models.py ← multi-map benchmark pipeline (Self-Play & BC)
│   ├── train_agent.py         ← single/multi-layout PPO trainer & status updater
│   ├── env_wrapper.py         ← OvercookedPartnerMultiAgent (k=4 frame history)
│   ├── frame_stacked_agent.py ← FrameStackedRlLibAgent inference adapter
│   └── checkpoint_policy.py   ← lightweight checkpoint loader (no trainer overhead)
│
├── evaluation/                ← cross-play benchmarking
│   └── eval_matrix.py         ← multi-agent cross-play matrix generator
│
├── overcooked-agent-eval/     ← evaluation framework & experimental baselines
│   ├── agents/                ← greedy heuristic & baseline agents
│   ├── experiments/           ← pretrained policy comparison scripts
│   └── tests/                 ← evaluation contract unit tests
│
├── external/overcooked_ai/    ← upstream Overcooked-AI submodule
│   ├── src/overcooked_ai_py/  ← core gridworld MDP simulation
│   └── src/human_aware_rl/    ← Carroll et al. PPO & BC implementations
│
├── project_docs/              ← research proposal & literature notes
├── work_done/                 ← milestone reports & baseline reproduction
└── results/                   ← gitignored telemetry, matrix CSVs, & logs
```

---

## Architecture Notes & References

For research background and foundational literature, see:
- **Carroll et al. (NeurIPS 2019)**: *"On the Utility of Learning about Humans for Human-AI Coordination"* — Foundational paper introducing Overcooked-AI. Demonstrated that pure self-play agents fail to coordinate with humans because they assume optimal partners; training alongside Behavior Cloning (BC) human models produces human-compatible policies. [arXiv:1910.05789](https://arxiv.org/abs/1910.05789)
- **Mnih et al. (Nature 2015)**: *"Human-level control through deep reinforcement learning"* — Established four-frame observation histories ($k=4$) for temporal velocity and acceleration representation in feed-forward networks. [Nature 518, 529–533](https://www.nature.com/articles/nature14236)
- **Canaan et al.**: Cooperative Game AI, partner modeling, and quality-diversity search in multi-agent environments.


**Advisor:** Prof. Rodrigo Canaan ([rcanaan@calpoly.edu](mailto:rcanaan@calpoly.edu))
