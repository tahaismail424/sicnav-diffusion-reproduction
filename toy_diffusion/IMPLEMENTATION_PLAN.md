# Toy Diffusion First-Pass Plan

Last updated: 2026-09-02

## Objective

Build a small conditional trajectory DDPM that predicts multiple plausible future 2D trajectories from an observed past trajectory.

This first pass is for understanding and debugging the mechanics of diffusion in trajectory forecasting. It is not yet a MID/JMID reproduction.

## Assumption

This plan interprets "DPPM" as "DDPM" in the usual denoising diffusion probabilistic model sense.

## Why This First

This stage should teach one thing well:

- how conditional diffusion can model multimodal futures

It should not also require learning:

- ETH/UCY preprocessing
- joint multi-human forecasting
- MPC
- SICNav
- CrowdSim/CrowdSimPlus

## Concrete First-Pass Deliverable

At the end of the first pass, we want to be able to:

1. Generate synthetic trajectories where the same past can branch into multiple valid futures.
2. Train a conditional DDPM on that dataset.
3. Sample several futures for the same past trajectory.
4. Show that the model produces different valid modes instead of averaging them.
5. Compare against a deterministic baseline that tends to average the modes.

## Proposed Data Design

Use synthetic 2D trajectories with:

- `obs_len = 8`
- `pred_len = 12`
- 2D positions

Start with one simple multimodal family:

- straight past motion
- future branches into two or three modes
- examples:
  - continue straight
  - turn up
  - turn down

Then add mild variation:

- start position jitter
- speed jitter
- curvature jitter
- Gaussian observation noise

Important property:

- multiple training examples should share very similar pasts but differ in future mode

That is what makes the diffusion model visibly useful.

## Representation Choice

Use future trajectory displacements relative to the last observed point as the diffusion target.

Why:

- easier scale for the network
- easier later transfer to real trajectory datasets
- keeps the conditioning and target more stable

Concretely:

- input context:
  - observed past positions
  - optionally observed past velocities
- target `x0`:
  - flattened future displacements of shape `(pred_len, 2)`

## Model Choice

Keep the first denoiser intentionally simple:

- MLP denoiser over flattened future trajectory
- sinusoidal timestep embedding
- context encoder as a small MLP over flattened past trajectory
- concatenate:
  - noisy future `x_t`
  - timestep embedding
  - context embedding
- predict noise `epsilon`

This is simpler than a U-Net or Transformer and is enough for the synthetic stage.

## Training Objective

Standard epsilon-prediction DDPM objective:

- sample training example `(context, x0)`
- sample timestep `t`
- sample Gaussian noise `epsilon`
- create `x_t`
- predict `epsilon_theta(x_t, t, context)`
- minimize MSE between true and predicted epsilon

## Baseline

Add one deterministic baseline:

- MLP regressor from past trajectory to future trajectory
- train with plain MSE on the future coordinates

Expected result:

- the baseline averages across modes
- the diffusion model samples distinct plausible futures

This comparison will make the motivation for generative prediction much clearer.

## Proposed Folder Layout

Suggested first-pass structure inside `toy_diffusion/`:

```text
toy_diffusion/
  README.md
  IMPLEMENTATION_PLAN.md
  configs/
    default.yaml
  data.py
  models.py
  diffusion.py
  train.py
  sample.py
  visualize.py
  baseline.py
  outputs/
```

## Module Responsibilities

`data.py`

- synthetic dataset generation
- train/val split
- normalization helpers

`models.py`

- timestep embedding
- context encoder
- DDPM denoiser MLP
- deterministic baseline MLP

`diffusion.py`

- beta schedule
- forward noising
- training helper for sampling `t` and `epsilon`
- reverse sampling loop

`train.py`

- config loading
- dataloaders
- training loop
- checkpoint saving
- loss logging

`sample.py`

- load checkpoint
- sample `K` futures for chosen contexts

`visualize.py`

- plot past trajectory
- plot ground truth future
- plot multiple DDPM samples
- optionally compare with deterministic baseline prediction

`baseline.py`

- train deterministic past-to-future regressor

## First Implementation Milestones

### Milestone 1: Dataset and plots

Build the synthetic dataset and verify:

- branch modes look correct
- the same past can map to multiple futures
- scaling and normalization look sane

Success check:

- save a figure with several random dataset samples

### Milestone 2: Deterministic baseline

Train the MSE baseline first.

Success check:

- on ambiguous pasts, predictions tend toward the average path

### Milestone 3: DDPM training

Train the conditional DDPM.

Success check:

- training loss decreases stably
- reverse samples look like trajectories, not random noise clouds

### Milestone 4: Qualitative multimodality demo

For a fixed past trajectory, sample multiple futures.

Success check:

- samples cover distinct valid branches
- samples are not all identical
- samples are not implausible geometric nonsense

### Milestone 5: Basic quantitative checks

Add lightweight metrics for the synthetic setup:

- best-of-K ADE/FDE
- sample diversity
- mode coverage on a held-out branched set

We do not need perfect metrics here, just enough to confirm the model learned something real.

## Hyperparameter Defaults

Reasonable first defaults:

- `obs_len = 8`
- `pred_len = 12`
- `num_diffusion_steps = 100`
- `beta_schedule = linear`
- `batch_size = 128`
- `lr = 1e-3`
- `hidden_dim = 256`
- `epochs = 50`

These are starting points, not commitments.

## Verification Artifacts

The first pass should leave behind:

- training curves
- dataset visualization
- deterministic baseline visualization
- DDPM sample visualization
- one short markdown writeup summarizing:
  - setup
  - what worked
  - what still looks wrong

## What We Intentionally Postpone

Do not pull these into the first pass unless the toy model is already working:

- ETH/UCY dataset loading
- social context from neighboring humans
- joint prediction of multiple humans
- map conditioning
- DDIM acceleration
- SICNav planner integration
- CrowdSim/CrowdSimPlus closed-loop evaluation

## Recommended Next Step After This Plan

Implement in this order:

1. `data.py`
2. `visualize.py`
3. `baseline.py`
4. `diffusion.py`
5. `models.py`
6. `train.py`
7. `sample.py`

This keeps debugging local and makes the generative-vs-deterministic comparison available early.
