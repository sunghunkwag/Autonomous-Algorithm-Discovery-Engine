# Autonomous Algorithm Discovery Engine

A sophisticated Evolutionary Algorithm (EA) engine designed to autonomously discover and synthesize algorithms for fundamental computational tasks (SUM, MAX, DOUBLE).

## Key Features

*   **Two-Stage Evolution**: 
    *   **Stage 1**: Structural discovery using Control Flow Graph (CFG) analysis to find loops and branching patterns.
    *   **Stage 2**: Task-specific optimization with curriculum learning and bias feedback.
*   **Virtual Machine (VM)**: A custom-built register-based VM with 8 registers and 64-slot memory.
*   **Meta-Learning Feedback**: Stage 2 results generate opcode biases that improved Stage 1's search efficiency (Self-Guided).
*   **Strict Verification**: 
    *   **Anti-Cheat**: Filters out linear code ("fake passes") by enforcing loop structures (SCCs).
    *   **Determinism**: Ensures rigorous reproducibility of discovered algorithms.

## Repository Structure

*   `omega_forge_two_stage_feedback.py`: **Main Engine.** The latest version featuring the full Two-Stage pipeline, Feedback Loop, and Curriculum Learning.
*   `archive/`: Contains previous iterations and legacy versions (e.g., `OMEGA_FORGE_V13_STAGED_EVOLVED.py`).

## Quick Start

Run the full discovery pipeline:

```bash
python omega_forge_two_stage_feedback.py full
```

This will:
1. Run **Stage 1** to collect structural candidates.
2. Run **Stage 2** to evolve them against SUM/MAX/DOUBLE tasks.
3. Save the discovered algorithms and feedback data.
