# RoadJEPA

RoadJEPA is a Rust-first framework for compact JEPA-style latent predictive modeling, with a current focus on temporal prediction (vision-based synthetic tasks) rather than broad framework abstraction.

## Current Scope (from `VISION.md`)

- frozen-encoder `VisionJepa` and `ProjectedVisionJepa` training paths
- synthetic temporal batch generation and temporal training examples
- deterministic regression test coverage for step/trajectory behavior and loss contracts

## Core Verification Commands

```bash
cargo test --test temporal_vision_support
cargo test --test projected_temporal_support
cargo clippy --all-targets --all-features
```
