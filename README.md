# Overcooked-AI Research Workspace

This repository contains my senior project research workspace for cooperative AI agents in Overcooked-AI.

## Project Direction

I am building a software framework on top of the HumanCompatibleAI Overcooked-AI benchmark to create, run, evaluate, and visualize cooperative AI agents in an Overcooked-style environment.

## Research Question

How do different cooperative agent strategies affect team performance, coordination quality, and partner compatibility in Overcooked-AI?

## Repository Structure

- `external/overcooked_ai/` - Overcooked-AI benchmark environment as a Git submodule
- `overcooked-agent-eval/` - My custom evaluation framework
- `overcooked-agent-eval/agents/` - Custom agent strategies
- `overcooked-agent-eval/experiments/` - Experiment runners and configs
- `overcooked-agent-eval/telemetry/` - Logging tools
- `overcooked-agent-eval/metrics/` - Coordination and performance metrics
- `overcooked-agent-eval/results/` - Generated experiment outputs
- `overcooked-agent-eval/notebooks/` - Analysis notebooks
- `overcooked-agent-eval/dashboard/` - Visualization tools
- `overcooked-agent-eval/tests/` - Tests

## Planned Metrics

- Team score
- Soups delivered
- Episode length
- Idle time
- Blocking/collisions
- Task switches
- Distance traveled
- Role specialization
- Partner compatibility

## Research Goal

The goal is not just to train an AI agent to play Overcooked. The goal is to build a reusable evaluation framework for studying cooperative AI behavior, agent coordination, and teammate compatibility.
