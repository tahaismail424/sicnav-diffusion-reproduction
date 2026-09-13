# SICNav-Diffusion Reproduction Context

Last updated: 2026-09-12

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

## Completed Foundations

### Toy conditional DDPM

`toy_diffusion/` is complete as an educational baseline:

- `data.py` generates cached, stochastic, multimodal 2D trajectories.
- `diffusion.py` exposes `Diffuser`, a reusable `nn.Module` that owns beta,
  alpha, and cumulative-alpha buffers plus batch-friendly forward noising,
  x0 reconstruction, and ancestral DDPM reverse steps.
- `models.py` contains a conditional MLP noise-prediction network. Its default remains the toy
  50-position-history / 10-position-future task, but its observation and
  prediction widths are configurable for ETH/UCY.
- `train.py` trains epsilon prediction and records direct x0-reconstruction
  ADE/FDE diagnostics.
- `01_toy_diffusion_workflow.ipynb` provides data inspection, overfitting,
  training, and multi-sample reverse-diffusion visualizations.

The toy model trained successfully and produces sensible straight/left/right
future modes. It is intentionally not a research benchmark model.

### Diffusion ownership convention

Keep the learned predictor and diffusion process separate:

- `Diffuser` owns the schedule and implements `add_noise`, `predict_x0`, and
  `denoise_step`.
- A custom predictor such as `Denoiser` or future JMID implements only
  `epsilon_theta(x_t, timestep, conditioning)`.
- Training and sampling create both objects, move both to the selected device,
  and save both state dictionaries in checkpoints.

This makes the DDPM mechanics reusable as the trajectory architecture changes.

Exception: the faithful reference implementation in `trajectory_prediction/MID`
retains its original `VarianceSchedule` and `DiffusionTraj`; do not replace
them with the toy `Diffuser` during the MPS port because that would complicate
algorithmic fidelity comparisons.

### ETH/UCY open-loop pipeline

The downloaded Social GAN-format data is intentionally untracked at
`experiments/eth_ucy/data/`.

- Rows are `frame_id, pedestrian_id, x, y` in meters.
- A repeated frame number is one timestamp containing positions for multiple
  pedestrians; it is not duplicated trajectory data.
- Frame labels advance by 10 source-video frames, yielding the benchmark's
  2.5 Hz / 0.4 s sampling rate.
- The supplied `eth`, `hotel`, `univ`, `zara1`, and `zara2` directories are
  standard leave-one-scene-out folds, each with supplied train/val/test splits.
- `dataset.py` creates valid single-pedestrian sliding windows with 8 observed
  and 12 future coordinates, normalized around the final observed point.
- `01_single_agent_ddpm_benchmark.ipynb` uses the shared Denoiser for a
  single-agent DDPM baseline. It includes a constant-velocity reference,
  validation-only model selection, 100-step ancestral sampling, and
  best-of-20 ADE/FDE test evaluation.

Use all 20-position windows with stride 1, as is conventional for this
benchmark. Never tune hyperparameters after inspecting the corresponding test
partition.

### Environment

Maintain one `environment.yml` at the repository root for the toy and ETH/UCY
stages. The current Conda CLI has an unresolved duplicate-`libomp.dylib` error
when importing PyTorch. Do not suppress it with `KMP_DUPLICATE_LIB_OK`; diagnose
or rebuild the environment if it also affects the intended notebook kernel.

## Current Goal: Complete the Closed-Loop Policy Ladder

The faithful JMID open-loop benchmark is running. The closed-loop reproduction
now has a staged policy ladder:

1. Finish and record the full JMID ETH/UCY benchmark.
2. Compare linear and ORCAPlus robot baselines in CrowdSimPlus.
3. Run DWA to introduce feasible unicycle local control.
4. Run MPC-CVMM to introduce horizon optimization without interaction.
5. Run SICNav-CVG to introduce bilevel ORCA response without diffusion.
6. Load the released simulation JMID checkpoint and run SICNav-JMID.
7. Compare every policy on paired cases across built-in scenarios.

`experiments/eth_ucy/IMPLEMENTATION_PLAN.md` is the detailed plan and acceptance
criteria for this phase.

### Terminology and fidelity

- The current MLP baseline is a conditional DDPM, not MID.
- iMID is SICNav's name for the original MID-style *individual* forecaster:
  target-agent history plus neighboring-agent context, Trajectron++ encoder,
  and Transformer diffusion decoder.
- JMID produces one coupled future sample for every human in a scene; it is not
  equivalent to independently sampling iMID once per pedestrian.
- Do not implement a custom "iMID-compatible" social-DDPM detour unless the
  user explicitly reintroduces it; the chosen next target is fidelity-oriented
  iMID.

### MID Reference Baseline

The original MID repository is included as the `trajectory_prediction/MID`
submodule. It is the chosen faithful iMID reference implementation; do not
rewrite Trajectron++ from scratch at this stage.

- Active path: `models/trajectron.py` and `models/encoders/mgcvae.py` provide
  the Trajectron++ target/social context encoder; `models/diffusion.py`
  provides MID's 100-step velocity DDPM and `TransformerConcatLinear` epsilon
  decoder; `models/autoencoder.py` wraps them.
- `models/transformer.py` is unused experimental code and is not part of the
  MID trajectory pipeline.

### JMID Reference and Benchmark

- `trajectory_prediction/safe-interactive-crowdnav` is the user's fork of the
  official SICNav repository. Its `sicnav_diffusion/JMID/MID` subtree is the
  authoritative JMID implementation; do not recreate Trajectron++ or its joint
  decoder in this repository.
- The portable JMID path resolves `auto` as MPS, then CUDA, then CPU. It keeps
  the official joint padding/masking logic and changes only device portability,
  current-Python compatibility, configurable output paths, and split hygiene.
- Use `experiments/eth_ucy/03_jmid_benchmark.ipynb` for exploration and
  `run_jmid_benchmark.py` / `run_jmid_benchmark.sbatch` for durable local/HPC
  runs. Train on `*_train.pkl`, select only on `*_val.pkl`, then evaluate the
  selected epoch once on `*_test.pkl`.
- JMID reports per-agent best-of-20 ADE/FDE and scene-level best-of-20
  SADE/SFDE. Compare ADE/FDE with iMID and the toy DDPM; SADE/SFDE assess the
  coherence of a whole joint sample.
- `mid.py` has been modernized to resolve one global device (`mps` when
  available, otherwise CUDA, otherwise CPU), remove active CUDA hard-coding,
  select `best.pt` by validation ADE, and evaluate the held-out test pickle
  only once after selection.
- The MPS path has been run successfully for a local held-out fold. For the
  slower full benchmark, use `experiments/eth_ucy/run_mid_benchmark.py` on a
  CUDA host such as Grace with explicit `--data-dir`, `--output-dir`, and
  `--device cuda` arguments.
- `trajectory_prediction/MID/environment.yml` is the pip-first reproducible
  environment for the MID fork. On a CUDA host, verify `torch.cuda.is_available()`
  after creation and replace only the PyTorch wheel with the CUDA index selected
  for that host; keep the remaining pinned Python dependencies unchanged.
- `experiments/eth_ucy/02_mid_benchmark.ipynb` is the canonical MID notebook:
  it smoke-tests a batch, runs one or five leave-one-scene-out folds, persists
  `mid_all_folds.csv`, and compares MID with constant velocity and the toy
  single-agent DDPM.

For fidelity, preserve the original state contract: eight history samples of
`[x, y, vx, vy, ax, ay]`, 12 future velocity samples, a 3 m PEDESTRIAN-to-
PEDESTRIAN attention radius, dynamic social edges, and the original linear
100-step schedule ending at beta `5e-2`. Improvements to the model itself
should be explicit follow-up experiments, not accidental changes in the MPS
port.

### CrowdSimPlus Baseline

`experiments/crowd_navigation/01_crowdsim_baseline.ipynb` is the initial
closed-loop exercise. It directly imports `baseline.py`, runs deterministic
CrowdSimPlus test cases, visualizes executed robot/human trajectories, and
records success, timeout, collision, clearance, navigation-time, path-efficiency,
and policy-runtime metrics.

The first baseline intentionally uses a goal-seeking linear robot and ORCAPlus
humans. It excludes JMID, CasADi, Acados, and MPC so the Gym-style `reset ->
robot.act -> env.step` contract can be understood and debugged alone. SFM
humans remain available as a dependency-light comparison. CrowdSimPlus now
imports Gymnasium when available and falls back to legacy Gym; ORCA/RVO2 and
SB3 policies are lazy-loaded optional dependencies. Use
`experiments/crowd_navigation/environment.yml` for this simulator-first stage.

The maintained RVO2 C++ fork lives at `trajectory_prediction/RVO2`. It now
contains a pybind11 binding compatible with the legacy
`rvo2.PyRVOSimulator` API and a scikit-build-core `pyproject.toml`, so it can be
compiled and installed into an active environment with `python -m pip install
-e trajectory_prediction/RVO2`. `PYTHON_BINDINGS.md` documents the build layers,
wheel portability, ownership/GIL considerations, and a general C++ binding
workflow. Do not install the stale external Python-RVO2 project for this repo.

Generated baseline CSVs live under `experiments/crowd_navigation/outputs/` and
are intentionally untracked. Keep test-case IDs fixed for paired comparisons
when ORCA, CVMM-MPC, and SICNav-Diffusion are added.

### Crowd-Navigation Policy Suite

`experiments/crowd_navigation/navigation.py` is the shared runner for `linear`,
`orca_plus`, `dwa`, `mpc_cvmm`, `sicnav_cvg`, and `sicnav_jmid`. It owns
in-memory config overrides, optional dependency checks, device resolution,
paired deterministic runs, checkpoint inspection, and tidy CSV output. Keep
policy construction here rather than duplicating it in notebooks.

The corresponding notebooks are numbered `02_dwa.ipynb` through
`06_policy_comparison.ipynb`. Expensive results are persisted under
`outputs/policy_benchmark`; the comparison notebook does not rerun missing
experiments unless `RUN_MISSING=True` is explicitly selected.

`07_stress_benchmark.ipynb` is the higher-power follow-up. It runs all built-in
scenarios at three humans and a density sweep in `hallway_static_with_back`,
checkpoints every episode under `outputs/stress_benchmark`, resumes completed
case IDs, and plots Wilson uncertainty intervals. Non-three-human JMID results
are explicitly out-of-distribution stress tests.

- DWA and MPC-CVMM have passed end-to-end local smoke tests.
- MPC-CVMM requires CasADi/IPOPT; HSL/MA57 is optional and the local test used
  IPOPT's default fallback linear solver.
- SICNav-CVG and SICNav-JMID pass end-to-end local smoke tests with acados
  source `71800fb7a`, `acados_template==0.5.1`, and PyTorch 2.14. The first JMID
  generated-solver build takes about five minutes; later runs use the cache.
- The released JMID simulation checkpoint loads and performs inference on CPU.
  Its trusted full-module pickle needs the historical `models` package alias
  and explicit `weights_only=False`; `mid.py` handles both without relying on
  the launch directory.
- Current CasADi `blocksqp` requires proprietary HSL/MA27. The ORCA warm-start
  NLPs use portable `sqpmethod` plus qpOASES; acados remains the main optimizer.
- acados can report recoverable QP/max-iteration statuses during SICNav. Keep
  solver diagnostics visible and evaluate aggregate behavior, not one status.
- The strict pybind11 RVO2 API exposed a legacy float-to-integer conversion;
  the planner ORCA wrapper now reads `max_neighbors` with `getint`.

See `experiments/crowd_navigation/IMPLEMENTATION_NOTES.md` for setup details,
method-to-config mappings, portability rationale, and verification boundaries.

## Working Style

When helping on this repo:

- favor educational clarity over premature complexity
- keep the first pass small and inspectable
- prefer synthetic data first
- make open-loop forecasting work before any planner integration
- explicitly call out assumptions when moving from toy diffusion toward MID/JMID
- preserve the distinction between an educational approximation and a faithful
  paper reproduction
