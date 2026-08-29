# Changelog

## 0.5.2 - Calibration Engine

- added resumable `calibrate` commands with minimum, mid and maximum depth
- added stable hardware/runtime fingerprints and stale stage-cache detection
- added adaptive FIT, context/KV, CPU and placement searches using objective llama.cpp benchmarks
- recorded prompt-processing, token-generation, RAM and VRAM measurements
- generated compact, normal, extended and reasonable-maximum runtime profiles
- protected higher-depth results from shallower replacement
- required a finite Zero-Idle check before calibration promotion

## 0.5.1 - Model Lifecycle

- added Hugging Face GGUF search and resumable download support
- added local GGUF import with disk-space and SHA-256 preflight checks
- added content-addressed artifact storage with deduplication
- added integrity, llama.cpp compatibility and calibration-aware readiness checks
- recorded source, revision, license and artifact metadata in the Model Registry
- added safe removal with assigned-Role and shared-artifact protection
- added a versioned backend compatibility manifest

## 0.5.0 - Runtime Configuration

- added RuntimeConfigGenerator from Role, Model Registry and Calibration state
- added read-only runtime plan, render and status commands
- added guarded runtime apply with atomic config replacement, health checks and rollback
- added transactional Role/model/profile switching with Role Registry rollback on failure
- added `model assign`, `model unassign`, `model swap` and `role profile` management commands
- added user-service `start`, `stop` and `logs` commands
- preserved the pre-apply service state during validation and rollback
- supported valid generated configuration when no local Role is bound
- enforced Zero-Idle llama-swap policy invariants in generated configs
- added `tars help` as a top-level help alias
- documented the future Temporary session persistence contract

## 0.4.3 - Orchestration Core

- added capability-based AUTO Role routing without role-name special cases
- added structured DelegationRequest/DelegationResult persistence
- added child delegation tasks that preserve parent ownership
- added checkpoint-gated transactional task handoff
- added durable routing, delegation and handoff history
- added CLI surfaces for routing, delegation and handoff inspection

This repository starts its public Git history at the v0.4.2 development baseline. Earlier work is summarized here rather than reconstructed as artificial commits.

## 0.4.2 — 2026-08-29

Public development baseline.

- background task-run foundation
- SSE final/reasoning streaming
- durable task runs and safe-boundary controls
- SQLite state schema v3
- role-aware ContextManager and tokenizer-aware budgeting
- pinned terminal HUD and theme system
- dynamic Role Registry
- Model Registry and calibration-store foundations
- Zero-Idle llama-swap reference runtime

## Pre-public development

v0.1 through v0.4.1 established the local runtime, model calibration, Role Registry, terminal UI, canonical task state, immutable checkpoints and ContextManager. Detailed benchmark history is intentionally kept out of the public repository's normal source tree.
