# AIDAN: Neuro-Symbolic Mathematical Reasoning Engine

An autonomous, neuro-symbolic AI reasoning system engineered to solve complex mathematical problems by unifying deep learning representations with symbolic search and formal verification algorithms.

AIDAN integrates a custom Nano Transformer architecture with Monte Carlo Tree Search (MCTS) and Hierarchical Reinforcement Learning (HRL), delivering high-precision multi-step reasoning while maintaining a strict <15 GB RAM footprint for consumer-grade hardware execution.

---
## Core Engineering Highlights
Neuro-Symbolic Dual System: Blends statistical language representations with deterministic symbolic execution routines to drastically reduce hallucination in multi-step proofs.
MCTS-Guided Path Pruning: Employs tree search algorithms to evaluate alternative proof trajectories, pruning suboptimal steps before state transitions.
Resource-Constrained Optimization: Tailored tensor management and execution buffers enable complex MCTS rollouts and inference within a strict 15 GB RAM boundary.
Curriculum-Driven Reinforcement Learning: Dynamically adjusts task difficulty based on convergence metrics, allowing autonomous progression from fundamental logic to complex proofs.
Modular Object-Oriented Framework: Clean separation between environment wrappers, search controllers, neural policies, and state loggers.
Auditable Execution Logs: Granular event logging records every state evaluation, heuristic score, and search decision for deep telemetry and debugging.

---
## Tech Stack
Core Frameworks: Python 3.10+, PyTorch, NumPy
Algorithmics: Monte Carlo Tree Search (MCTS), Hierarchical Reinforcement Learning (HRL)
Symbolic & Environment: Custom Formal Logic Wrappers (environment/math_env.py)
System & Benchmarking: Custom Memory Profilers, Logging Interfaces

---
## Hardware Constraints & Benchmarks
MetricTarget / BenchmarkMax 
RAM Footprint< 15 GB (Enforced via buffer caps)
Search EngineMonte Carlo Tree Search (MCTS)
Verification EngineSymbolic Environment Transition Checks
Target RuntimeConsumer Hardware / Standard Cloud VMs

---
## Architecture Overview

AIDAN bridges the gap between probabilistic neural generation and deterministic symbolic verification to bound hallucination rates during complex deduction steps.

```text
                                  ┌────────────────────────┐
                                  │   Input Math Problem   │
                                  └───────────┬────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ AIDAN Core Dual-System Loop                                                             │
│                                                                                         │
│   ┌──────────────────────────┐    Guided Search    ┌────────────────────────────────┐   │
│   │  Neural Policy / Value   │ ──────────────────> │    Monte Carlo Tree Search     │   │
│   │   (Nano Transformer)     │ <────────────────── │        (MCTS Engine)           │   │
│   └──────────────────────────┘    Value Feedback   └───────────────┬────────────────┘   │
│                                                                    │                    │
└────────────────────────────────────────────────────────────────────┼────────────────────┘
                                                                     │ Step Proposal
                                                                     ▼
                                                     ┌────────────────────────────────┐
                                                     │     Symbolic Environment       │
                                                     │     (environment/math_env)     │
                                                     └───────────────┬────────────────┘
                                                                     │
                                                                     ▼
                                                     ┌────────────────────────────────┐
                                                     │   Verified Deductive Output    │
                                                     └────────────────────────────────┘
