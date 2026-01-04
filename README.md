# Autonomous Algorithm Discovery Engine

> [!IMPORTANT]
> **Experimental Research Project**
> This repository contains code for research into evolutionary computing and program synthesis. It is a Proof of Concept (PoC) and is not intended for production usage. The algorithms discovered here are for educational and analytical purposes.

This project implements a **Two-Stage Evolutionary Algorithm** that synthesizes simple programs (in a custom assembly language) to solve computational tasks without human intervention.

## Technical Components

1.  **Custom Virtual Machine (VM)**: A register-based VM (8 registers, 64-slot memory) that executes the evolved code.
2.  **Two-Stage Evolution**:
    *   **Stage 1 (Structure)**: Evolves code structures using Control Flow Graph (CFG) analysis to find loops and branching, maximizing structural diversity.
    *   **Stage 2 (Task)**: Optimizes the Stage 1 candidates to solve specific tasks (SUM, MAX, DOUBLE).
3.  **Strict Verification**: Evolved programs must pass a "Strict Structural Detector" that rejects linear code and requires valid control flow (SCCs) to be considered a solution.
4.  **Meta-Feedback**: The system uses a feedback loop where successful instruction patterns from Stage 2 bias the random generation in Stage 1.

## Code Structure

*   `omega_forge_two_stage_feedback.py`: The main Python script containing the engine logic, VM, and evolutionary pipeline.
*   `archive/`: Contains legacy versions and experimental logs.

## Usage

To run the discovery process:

```bash
python omega_forge_two_stage_feedback.py full
```
