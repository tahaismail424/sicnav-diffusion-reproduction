# SICNav-Diffusion Reproduction Context

Last updated: 2026-09-02

## Project

This repository is for a first-principles reproduction path toward the paper:

- SICNav-Diffusion: Safe and Interactive Crowd Navigation with Diffusion Trajectory Predictions
- arXiv: https://arxiv.org/abs/2503.08858
- repo: https://github.com/tahaismail424/sicnav-diffusion-reproduction

The user has just joined Yiwei Lyu's lab and is using this project both as:

- a real research reproduction effort
- an educational scaffold for learning diffusion, trajectory forecasting, planning, and social navigation

## High-Level Framing

Important conceptual split:

1. Open-loop trajectory forecasting
   - Predict human futures from past trajectories.
   - Benchmark on ETH/UCY.
   - Metrics include ADE/FDE for individual forecasts and SADE/SFDE for joint forecasts.

2. Closed-loop robot navigation
   - Use human predictions inside a planner/controller.
   - Evaluate robot behavior in simulation such as CrowdSim/CrowdSimPlus.
   - Metrics include success, collisions, freezing, time-to-goal, etc.

The user's current understanding is correct:

- ETH/UCY experiments tell us how accurately we predict human motion.
- CrowdSim/CrowdSimPlus experiments tell us how well the robot navigates when robot and humans react to each other.

This distinction should be preserved in all future planning and implementation.

## Paper-Specific Notes

From the SICNav-Diffusion paper:

- The method combines a joint human trajectory diffusion model with a bilevel SICNav MPC formulation.
- The lower-level optimization acts like a safety filter on human predictions.
- Importance weights are maintained over joint trajectory samples.
- Human intended positions are estimated by a weighted average over sampled future positions.
- The paper evaluates:
  - open-loop forecasting on ETH/UCY
  - closed-loop navigation in simulation and on robots

Useful method details for future implementation:

- JMID is the joint diffusion predictor for human futures.
- DDPM inference uses 100 denoising steps in the paper.
- DDIM with 2 denoising steps is used for faster inference and real-time deployment.
- The importance weight update is particle-filter-like and depends on agreement between refined positions and sampled predictions.

## Educational Context From Prior Chat

The user is still building intuition for diffusion and wants to understand the mechanics before reproducing MID/JMID.

Current educational priorities:

- Understand conditional diffusion mechanically, not just run code.
- Learn why multimodal future prediction is a generative problem.
- Separate forecasting from planning before integrating them.
- Build transferable ML skills, ideally around learned planning/prediction under safety constraints.

Key diffusion concepts the user has already started to internalize:

- The forward diffusion process corrupts the future trajectory, not the past/context.
- Context such as past trajectory helps denoising because it is informative about the clean future.
- The denoiser can be viewed as a function like `epsilon_theta(x_t, t, context)`.

## Recommended Learning / Implementation Order

Preserve this progression unless there is a strong reason to change it:

1. Toy conditional trajectory DDPM on synthetic 2D multimodal data
2. ETH/UCY evaluation pipeline with trivial baselines
3. Run the simple DDPM on ETH/UCY
4. Reproduce MID / iMID ideas
5. Reproduce JMID and open-loop joint forecasting experiments
6. Learn SICNav / MPC / bilevel planning separately from learned prediction
7. Learn CrowdSim/CrowdSimPlus closed-loop evaluation separately
8. Integrate JMID with SICNav

Reason:

- Forecasting and navigation are two different systems.
- The user should not have to learn diffusion, benchmarking, planning, and simulation all at once.

## Immediate Goal

Immediate task for this repo:

- write a concrete first-pass implementation plan for a toy conditional trajectory DDPM in `toy_diffusion/`

Interpret "DPPM" in current discussion as "DDPM" unless the user explicitly means something else.

## Working Style

When helping on this repo:

- favor educational clarity over premature complexity
- keep the first pass small and inspectable
- prefer synthetic data first
- make open-loop forecasting work before any planner integration
- explicitly call out assumptions when moving from toy diffusion toward MID/JMID

