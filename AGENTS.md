# Agent Quickstart: SICNav-Diffusion Reproduction

`CLAUDE.md` is the authoritative project-context document. Read it before
planning, implementing, or revising work in this repository. It contains the
paper notes, educational context, staged roadmap, and working conventions.

## Mission

Support a first-principles, staged reproduction of SICNav-Diffusion while
helping the user understand and own each component.

## Critical Distinction

- ETH/UCY is an open-loop human trajectory forecasting benchmark.
- CrowdSim/CrowdSimPlus is a closed-loop robot navigation benchmark.

Do not conflate prediction quality with navigation quality.

## Current Focus

Work in `toy_diffusion/` first: a small conditional DDPM on synthetic,
multimodal 2D trajectories. Do not introduce ROS, CrowdSim, ETH/UCY loaders,
or the full MID/JMID architecture in this first stage.

For complete scope, implementation order, and design rationale, see
`CLAUDE.md`.
