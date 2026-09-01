# Work Done & Milestone Documentation

This directory tracks the architectural, engineering, and scientific progress of the Overcooked-AI research workspace across all milestones.

---

## 📋 Mandatory Repository Rule: Milestone Folder Standard

To ensure rigorous scientific reproducibility, transparent project history, and publication-ready documentation, **every project milestone must have a dedicated folder (`work_done/milestone_X/`)** containing comprehensive documentation.

### Each Milestone Folder Must Include:
1. **`README.md`**: The primary milestone report covering:
   - **Executive Summary & Research Context**: What was built and where it fits in the research roadmap.
   - **Scientific & Engineering Rationale**: *Why this milestone represents a robust base for research* and what scientific value it unlocks.
   - **Critical Pitfalls Avoided**: Methodological or software traps prevented by this milestone's design.
   - **Technical Deep-Dive**: Exhaustive explanation of components, schemas, metrics, or agents.
   - **Step-by-Step Reproduction Guide**: Exact CLI commands with copy-pasteable snippets.
   - **Verification & Test Coverage**: Passing test counts, telemetry validation, and sample outputs.
   - **Issues Closed & Upstream Alignment**: Cross-references to GitHub issues and PRs.
2. **Supplemental Deep-Dive Notes (Optional)**: Specific sub-milestone reports (e.g. `2a_telemetry_runner.md`, `2b_coordination_metrics.md`) for complex multi-part milestones.

---

## 🗂️ Milestone Directory Index

- **[Milestone 1: Baseline Sanity Check](milestone_1/README.md)**  
  *Status:* Completed & Merged (PR #1)  
  *Focus:* Submodule integration, Python 3.10 environment, and single-episode repeatable random baseline.

- **[Milestone 2: Complete Evaluation Foundation](milestone_2/README.md)**  
  *Status:* Completed & Verified (PR #12)  
  *Focus:* 27-field telemetry schema, 68-field coordination/role/duplication metric suite, multi-episode & multi-layout runner, reproducibility manifests, batch statistical summarizer, and 30-test suite.

- **[Milestone 3: Baseline & Learning Agents](milestone_3/README.md)** *(Upcoming)*  
  *Focus:* Heuristic agents (`Greedy`, `Prep`, `Runner`, `Support`, `RoleSpecialist`) and pretrained MARL adapter (Self-Play PPO / FCP).

- **[Milestone 4: Partner Compatibility Matrix](milestone_4/README.md)** *(Upcoming)*  
  *Focus:* $N \times N$ cross-play pairings across layouts and human trajectory ingestion baseline.

- **[Milestone 5: Compatibility Fingerprints & Significance](milestone_5/README.md)** *(Upcoming)*  
  *Focus:* Multi-dimensional behavioral radar profiles, self-play trap detection, and hypothesis testing.

- **[Milestone 6: Coordination Failure Mode Taxonomy](milestone_6/README.md)** *(Upcoming)*  
  *Focus:* Formal taxonomy of spatial deadlocks, pipeline starvations, and role collisions.

- **[Milestone 7: Robustness & Stress Testing](milestone_7/README.md)** *(Upcoming)*  
  *Focus:* Noisy teammates, partner swapping, and out-of-distribution layout transfer.

- **[Milestone 8: Publication Package & Manuscript](milestone_8/README.md)** *(Upcoming)*  
  *Focus:* Empirical research claims, publication-grade figures, and two-column IEEE/AAAI LaTeX manuscript.

