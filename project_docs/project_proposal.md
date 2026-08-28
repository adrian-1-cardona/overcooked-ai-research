# Project Proposal: Evaluating Cooperative AI Agents in Overcooked-AI

**Researcher:** Adrian Cardona  
**Advisor:** Prof. Rodrigo Canaan  
**Repository:** https://github.com/adrian-1-cardona/overcooked-ai-research  
**Date:** TBD

## 1. Project Overview

For my senior project, I want to build a software framework for evaluating cooperative AI agents in the Overcooked-AI environment.

In cooperative games, total score does not always explain why two agents worked well together or failed to coordinate. Two agents might get a decent score while still blocking each other, duplicating tasks, wasting time, or failing to specialize. Some agents may not be the highest scoring overall but may work better with certain partners.

This project is not just about training an AI to play Overcooked. The goal is to build a repeatable evaluation system that can run different agent pairings, collect gameplay telemetry, and compare coordination behavior across layouts.

The bigger research direction is:

**How do different cooperative agent strategies affect team performance, coordination quality, and partner compatibility in Overcooked-AI?**

This connects to human-AI coordination, cooperative game AI, multi-agent systems, reinforcement learning, evolutionary computing, and quality-diversity algorithms.

## 2. Project Goal

The goal is to create an Overcooked-AI Agent Evaluation Framework. It should eventually be able to:

- Run different agent pairings in Overcooked-AI.
- Collect per-step gameplay data.
- Measure coordination behavior beyond just final score.
- Compare different teammate strategies.
- Support future agents based on reinforcement learning, evolutionary algorithms, or quality-diversity methods.

For the first stage, I want to focus on building the software foundation before trying to create advanced agents.

## 3. Research Question

Main research question:

**How can we evaluate partner compatibility and coordination quality between different cooperative agents in Overcooked-AI?**

Possible subquestions:

- Do specialized agents coordinate better than general greedy agents?
- Which agent pairings lead to less blocking and idle time?
- How much does layout structure affect coordination?
- Can coordination metrics explain failures that total score hides?
- What kinds of agent behaviors are useful for human-aware coordination?

## 4. Minimum Viable Project

The MVP is a reusable Python framework on top of Overcooked-AI that can run baseline agent pairings, log scores and episode data, save structured results, compute basic coordination metrics, and compare results across pairings and layouts.

The first milestone is intentionally simple:

**Get Overcooked-AI running, run baseline agents, collect basic telemetry, and save repeatable experiment results.**

Before building smarter agents, I need to make sure I can run the environment and collect useful data reliably.

## 5. Initial Technical Milestone

**Milestone 1: Random Baseline Experiment**

Immediate tasks:

1. Verify the Overcooked-AI repo works locally.
2. Run a baseline layout such as `cramped_room`.
3. Run two simple agents, likely random agents or the simplest built-in baseline.
4. Log the layout, episode length, actions, rewards or score, and timesteps.
5. Save the results to a CSV.
6. Add setup and run instructions to the README.
7. Document the work in a `work_done/` folder.

This milestone is mostly a sanity check. It proves the project has a working foundation.

## 6. Possible Metrics

The project should move beyond only using final score. Possible metrics include team score, soups delivered, episode length, idle time, blocking or collisions, distance traveled, task switches, time holding useful objects, role specialization, partner compatibility, and performance across layouts.

These metrics can help explain not just how well agents performed, but how they coordinated.

## 7. Possible Agent Types

After the baseline is working, I could implement several simple agents:

- **RandomAgent** — baseline behavior.
- **GreedyAgent** — tries to maximize immediate progress.
- **RunnerAgent** — focuses on delivering completed soups.
- **PrepAgent** — focuses on gathering, chopping, and cooking ingredients.
- **SupportAgent** — prepares useful objects and avoids interference.
- **RoleSpecialistAgent** — commits to a specific part of the task pipeline.
- **AdaptiveAgent** — adjusts behavior based on teammate actions.

The research would come from comparing how these different agents work with different partners.

## 8. Stretch Goals

Possible stretch goals include reinforcement learning agents, evolutionary agents, quality-diversity search for cooperative play styles, a behavior dashboard, multi-layout comparisons, noisy or human-like teammate behavior, and studying which styles are more compatible with humans.

## 9. Expected Contributions

1. **Software Framework** — a reusable evaluation framework for running and comparing Overcooked-AI agents.
2. **Telemetry Dataset** — structured logs of actions, rewards, scores, and coordination behavior across experiments.
3. **Coordination Analysis** — metrics and findings that help explain why certain pairings coordinate better than others.

## 10. Why This Matters

In cooperative AI, getting a high score is not always enough. If an agent only works well with itself but fails with other partners, it may not be useful for human-AI teamwork.

Overcooked-AI is a good environment for this because agents must coordinate around shared tasks, limited space, timing, and role distribution. This makes it a strong testbed for studying teammate compatibility and coordination behavior.

I also want this project to be software-engineering oriented. The final repo should show clean project structure, reproducible experiments, readable code, and useful tooling, not just a one-off notebook or model.

## 11. Current Repository Status

Repository: https://github.com/adrian-1-cardona/overcooked-ai-research

- Research workspace created.
- Overcooked-AI added as a Git submodule.
- Custom project folder created under `overcooked-agent-eval`.
- README structure started.
- Milestone 1 random baseline and telemetry logging added.

## 12. Questions for Advisor Meeting

1. Should the project be framed more around evaluation, human-AI coordination, or quality-diversity search?
2. What would make this research-worthy beyond just building software?
3. What are the first two or three papers I should read after the Overcooked-AI paper?
4. What is the smallest experiment that would give useful research signal?
5. Which metrics would be most meaningful for measuring coordination?
6. Should I start with rule-based agents before attempting RL or evolutionary methods?
7. What should be the expected deliverable by the end of the quarter?

## 13. End-of-Quarter Target

By the end of the quarter, a realistic goal is a working repo, a running Overcooked-AI environment, the baseline experiment, basic telemetry and initial metrics, a short write-up, and a clear next-step plan for better agents or coordination metrics.
