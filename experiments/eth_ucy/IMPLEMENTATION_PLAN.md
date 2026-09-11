# ETH/UCY: From Baseline DDPM to iMID and JMID

## Goal

Build the open-loop forecasting side of SICNav-Diffusion in deliberate stages.
The comparison must distinguish our educational MLP DDPM baseline, a faithful
iMID reproduction, and finally JMID. Robot planning is deliberately out of
scope until this work is complete.

## Fixed Benchmark Contract

- Data: supplied ETH/UCY leave-one-scene-out fold directories.
- Window: 8 observed positions / 12 future positions at 2.5 Hz.
- Coordinates: metres, normalized to the target agent's final observed
  position for model input and evaluated in the same relative frame.
- Split discipline: tune on the supplied validation partition; run a fold's
  test partition only for the selected checkpoint and settings.
- Individual metrics: best-of-20 ADE and FDE.
- Joint metrics: best-of-20 SADE and SFDE once joint samples exist.
- Sampling: keep 100-step DDPM as the correctness reference, then add 2-step
  DDIM and compare both quality and wall-clock latency.

The SICNav-Diffusion paper uses this 3.2 s / 4.8 s protocol, 100-step DDPM and
2-step DDIM inference, and best-of-20 individual and scene-level metrics.

## Stage 0: Establish the Existing Baseline

### Deliverables

- Run `01_single_agent_ddpm_benchmark.ipynb` for each of `eth`, `hotel`,
  `univ`, `zara1`, and `zara2`.
- Save configuration, seed, checkpoint, validation curves, and test metrics per
  fold under ignored `outputs/`.
- Add a compact results table comparing constant velocity and the MLP DDPM.

### Acceptance checks

- Dataset examples have shapes `[8, 2]` and `[12, 2]`.
- Best-of-20 DDPM evaluation uses 20 independently sampled full trajectories.
- Test results are generated once, after validation selection.

### Purpose

This is a plumbing and diffusion sanity baseline, not a claim of MID-level
performance. Its conditioning has only the target's own history.

## Stage 1: Add Scene-Centric Data Without Changing the Predictor

### Why

MID needs the target human's history *and neighboring agents*. JMID also needs
all agents at one scene timestamp. The current flattened dataset intentionally
loses this grouping, so it is perfect for Stage 0 but insufficient after it.

### Deliverables

- Add a scene-window dataset alongside, not in place of, `dataset.py`.
- One item represents `(source file, start frame)` and preserves every agent
  visible in the observed horizon, its stable ID, and its per-timestep validity
  mask. Do not discard a neighbour merely because it lacks a full future.
- For iMID, form one target view only when the target has all 20 timestamps;
  its neighbours need only valid observed-history entries. For JMID later,
  select the subset of agents with complete 20-step trajectories.
- Return observed and future tensors, stable agent IDs, source metadata, and
  history/future validity masks after batching.
- Use a padded `collate_fn` with shapes `[B, N_max, T, 2]` plus boolean masks.
- Add tests for no missing timestamps, no duplicate agent/frame pairs,
  translation normalization, and batch masks.
- Make a notebook that plots one scene window, target agent, neighboring agents,
  and its retained agent IDs.

### Design decision

Keep the complete-future-agent filtering only for the first joint
implementation. This avoids inventing futures for agents that enter or leave,
while retaining richer neighbour context for iMID. Track this filtering rate by
scene; later work can add variable-lifetime joint masks.

## Stage 2: Faithful iMID Reproduction

### Scope

Replace the educational social encoder with the Trajectron++ encoder used by
MID, while preserving the same ETH/UCY protocol. Use the official MID code and
configuration as a reference, pin its commit/version, and document every
intentional deviation. Do not silently copy results or call the Stage 2 model
MID.

### Tasks

- Study the official MID preprocessing contract and Trajectron++ node/scene
  representation.
- Decide explicitly whether to vendor a pinned, minimal encoder dependency or
  reimplement its needed pedestrian-only pieces with unit comparisons against
  the reference.
- Match the original Transformer diffusion decoder's dimensions, layers,
  timestep embedding, schedule, loss, optimizer, batch size, and normalization
  as closely as hardware permits.
- Implement DDIM sampling after 100-step DDPM is correct; compare 2-step DDIM
  to DDPM for quality and latency.
- Create a reproducibility notebook/table with all five fold results and exact
  command/config provenance.

### Acceptance checks

- Input preprocessing agrees with the reference on fixed scene windows.
- A fixed checkpoint/sample seed gives matching tensor shapes and comparable
  intermediate ranges to the reference implementation.
- Results are labelled "faithful iMID reproduction" only after the above
  checks and documented deviations are complete.

## Stage 3: JMID

### What changes conceptually

iMID samples each target's future independently. JMID has one noise draw and
one final sample for the entire scene: `[N, 12, 2]`. Therefore one sample keeps
the pedestrians' futures coupled. This is exactly what SADE/SFDE assess.

### Model

1. Reuse the per-agent history encoder in parallel for all valid agents.
2. Diffuse the full padded future tensor with a mask; padded positions must not
   contribute noise loss or attention.
3. Project each agent's history embedding, noisy future representation, and
   timestep embedding through shared fully connected layers.
4. Concatenate valid agent representations and use a Transformer without agent
   positional encoding, matching the paper's permutation-equivariant intent.
5. Split outputs back by agent and predict epsilon with a masked loss.

### Data and metric work

- Define a stable flattening/order for valid agents solely for tensor packing;
  test permutation equivariance by reordering agents and undoing the reorder.
- Sample `K=20` **joint** futures per scene window.
- Compute ADE/FDE per agent as before. For SADE/SFDE, choose one of the 20 joint
  samples for the entire scene before averaging agents; never choose a separate
  best sample per agent.
- Add collision-rate and minimum inter-agent-distance diagnostics for generated
  samples. These are diagnostics, not substitutes for SADE/SFDE.

### Acceptance checks

- Agent reorder test passes after undoing the permutation.
- Independent iMID samples and JMID samples are evaluated with their proper,
  different metric interpretations.
- The JMID notebook visually shows coupled samples for crossing groups and
  reports all four best-of-20 metrics across five folds.

## Stage 4: Comparison and Handoff to Planning

Create one results notebook/table with:

- Constant velocity.
- Current MLP single-agent DDPM.
- Faithful iMID.
- JMID.

For each, record model/data version, trainable parameters, DDPM/DDIM steps,
sample count, inference time, ADE/FDE, and where applicable SADE/SFDE. Only
after that table is trustworthy should we begin a separate SICNav learning
track: ORCA refinement, importance weights, bilevel MPC, then CrowdSimPlus.
