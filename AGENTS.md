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

The toy DDPM and ETH/UCY single-agent baseline now exist. The next stage is
open-loop forecasting: first benchmark that baseline, then implement iMID with
fidelity and extend it to JMID in `experiments/eth_ucy/`. Do not introduce ROS,
CrowdSimPlus, or SICNav planning until the JMID open-loop work is complete.

## Naming Guardrail

- Call the current MLP model a "single-agent conditional DDPM," not MID.
- The user chose to skip an educational iMID-compatible detour; the next
  predictor implementation should target MID fidelity.
- JMID must emit coupled scene samples; independently sampled individual
  predictions are not JMID.

## Diffusion Convention

Use `toy_diffusion.diffusion.Diffuser` for new custom schedules and DDPM
mathematics. Predictor modules must contain only learned conditioning/
noise-prediction architecture; do not add beta or alpha buffers to custom
iMID/JMID models. The faithful `trajectory_prediction/MID` reference fork is
the intentional exception: preserve its native schedule and diffusion classes.

For complete scope, implementation order, and design rationale, see
`CLAUDE.md` and `experiments/eth_ucy/IMPLEMENTATION_PLAN.md`.

## MID Baseline

- `trajectory_prediction/MID` is the faithful iMID reference submodule; adapt
  it for current PyTorch and MPS/CPU portability instead of rebuilding
  Trajectron++.
- Use `experiments/eth_ucy/02_mid_benchmark.ipynb` for MID smoke tests and
  leave-one-scene-out evaluation.
- Use `experiments/eth_ucy/run_mid_benchmark.py` for non-interactive CUDA
  benchmark runs; its paths and device are command-line arguments.
- Use `trajectory_prediction/MID/environment.yml` for MID dependencies. CUDA
  hosts may require a host-appropriate PyTorch wheel after environment creation.
- Preserve split hygiene: train on `*_train.pkl`, select only with `*_val.pkl`,
  and run `*_test.pkl` once for the final result.
- `models/transformer.py` is unused by the MID pipeline. The active social
  encoder is `models/trajectron.py` plus `models/encoders/mgcvae.py`.
