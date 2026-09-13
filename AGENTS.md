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

The toy DDPM and faithful ETH/UCY MID/JMID paths exist. Closed-loop work now
progresses through linear/ORCAPlus, DWA, MPC-CVMM, SICNav-CVG, and finally
SICNav-JMID. Use the shared runner and paired scenario cases rather than adding
one-off policy setup to notebooks.

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

## JMID Baseline

- Use the official fork at `trajectory_prediction/safe-interactive-crowdnav`.
  Its `sicnav_diffusion/JMID/MID` implementation is the faithful JMID source.
- Run `experiments/eth_ucy/03_jmid_benchmark.ipynb` interactively or
  `run_jmid_benchmark.py` / `.sbatch` for durable runs. It must select on val
  and test only once after selecting the best epoch.
- JMID predicts pedestrians jointly. A simulator robot may be observed context,
  but it is not an ETH/UCY JMID forecast target.

## Crowd Navigation

- `experiments/crowd_navigation/01_crowdsim_baseline.ipynb` is the first
  closed-loop simulator exercise. It uses a linear robot and SFM humans, with
  no JMID, CasADi, Acados, or MPC.
- `experiments/crowd_navigation/baseline.py` owns reusable episode execution,
  trajectory plotting, and metrics. Reuse the same cases and metrics when
  adding ORCA and SICNav policies.
- `experiments/crowd_navigation/navigation.py` owns policy construction,
  dependency checks, config overrides, checkpoint inspection, batch runs, and
  result loading for all policies.
- Use notebooks `02_dwa.ipynb`, `03_mpc_cvmm.ipynb`, `04_sicnav_cvg.ipynb`,
  `05_sicnav_jmid.ipynb`, and `06_policy_comparison.ipynb` in that order.
- Use `07_stress_benchmark.ipynb` for resumable repeated-case scenario and
  density sweeps. It checkpoints each episode; preserve its paired case IDs
  and out-of-distribution label for non-three-human JMID runs.
- CasADi is enough for MPC-CVMM. SICNav-CVG/JMID also need compiled acados and
  its matching editable `acados_template`; see `IMPLEMENTATION_NOTES.md`.
- The verified local stack uses acados `71800fb7a`, `acados_template==0.5.1`,
  and portable CasADi SQP/qpOASES warm starts instead of HSL-dependent blockSQP.
- CVG and JMID both pass complete three-human bottleneck smoke tests. The first
  JMID solver compilation takes about five minutes and may print recoverable
  acados QP/max-iteration diagnostics.
- The released JMID checkpoint is trusted full-module pickle data. Preserve the
  robust historical-module path setup and explicit `weights_only=False` load.
- CrowdSimPlus accepts modern Gymnasium with legacy Gym as a fallback. Optional
  ORCA and SB3 policies are lazy-loaded so their dependencies are required only
  when selected.
- The maintained RVO2 C++ fork is the `trajectory_prediction/RVO2` submodule.
  Install its modern pybind11 extension with `python -m pip install -e
  trajectory_prediction/RVO2`; do not depend on the stale external
  Python-RVO2 package.
- The baseline defaults to `orca_plus` humans. SFM remains a dependency-light
  alternative, and selecting ORCA must fail clearly if the native extension is
  unavailable.
