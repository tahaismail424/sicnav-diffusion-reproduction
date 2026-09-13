# Crowd-Navigation Implementation Notes

This experiment layer runs the policies from the maintained
`trajectory_prediction/safe-interactive-crowdnav` fork directly. It does not
reimplement DWA, MPC, ORCA, acados, MID, or JMID.

## Shared Runner

`navigation.py` centralizes the parts that otherwise drift between notebooks:

- loading the official environment and policy config files;
- applying method/scenario overrides in memory;
- constructing the matching robot policy lazily;
- resolving MPS, CUDA, or CPU for JMID inference;
- running deterministic paired CrowdSimPlus cases;
- collecting the same safety, efficiency, and runtime metrics;
- writing tidy result CSVs for the comparison notebook;
- checking native dependencies before an expensive controller initialization;
- validating the released JMID checkpoint on CPU.

The runner temporarily enters the SICNav submodule directory around policy
initialization and action selection. The upstream code writes acados caches and
temporary iterate files using paths relative to that directory. Keeping this
compatibility behavior in one context manager avoids forcing notebooks to be
launched from a particular working directory.

## Policy Configuration

The method names map to official implementations as follows:

| Experiment name | Policy class | Important override |
| --- | --- | --- |
| `linear` | `Linear` | none |
| `orca_plus` | `ORCAPlus` | none |
| `dwa` | `DynamicWindowApproach` | calls `configure_dwa` |
| `mpc_cvmm` | `CollisionAvoidMPC` | `hum_model=cvmm` |
| `sicnav_cvg` | `SICNavAcados` | ORCA-KKT plus `human_goal_cvmm=true` |
| `sicnav_jmid` | `SICNavAcados` | ORCA-KKT plus joint MID prediction |

The environment still simulates humans independently using `orca_plus` or
`sfm`. The controller's internal model is therefore not the simulator's source
of truth.

## Modernization Fixes

### RVO2 integer contract

The C++ RVO2 constructor accepts `maxNeighbors` as an integer. The old Cython
binding silently converted the config value `10.0`; pybind11 rejects that
ambiguous conversion. `orca_c_wrapper.py` now reads `max_neighbors` with
`ConfigParser.getint`.

### Trusted PyTorch checkpoints

The released simulation checkpoints store complete encoder modules, not only
tensor state dictionaries. Their pickle records the historical top-level
module package `models`. Before loading, `mid.py` now adds its own MID directory
to `sys.path`, resolves checkpoint paths relative to the SICNav repository, and
calls:

```python
torch.load(path, map_location=device, weights_only=False)
```

`weights_only=False` is intentional only because these files are trusted,
repository-supplied checkpoints. Do not use it for an untrusted checkpoint.
The notebook first calls `inspect_jmid_checkpoint`, which loads on CPU and
requires both `encoder` and `ddpm` payloads. The released joint checkpoint has
18 encoder modules and 74 DDPM tensors under the current code.

### Optional imports

Solver-dependent policies are imported only after their dependency checks.
This keeps DWA and simulator baselines usable without CasADi or acados.

### Portable ORCA warm starts

The current CasADi wheel's `blocksqp` plugin defaults to the proprietary HSL
MA27 linear solver. The paper code used blockSQP only to generate an initial
guess for the main acados OCP, but failed before simulation when `libhsl` was
absent. The two warm-start NLPs now use CasADi `sqpmethod` with qpOASES. Their
objectives, constraints, iteration limits, and generated function interfaces
are unchanged; acados remains the actual trajectory optimizer.

## Solver Dependencies

CasADi is sufficient for `mpc_cvmm`. The implementation uses CasADi's Opti
stack and IPOPT. If HSL/MA57 is unavailable, the repository catches that case
and falls back to IPOPT's default linear solver; this is slower but worked in
the local smoke test.

SICNav-CVG and SICNav-JMID additionally require the compiled acados runtime and
a compatible `acados_template`. The local verified stack uses acados source
commit `71800fb7a` (`v0.6.0-9-g71800fb7a`) with `acados_template==0.5.1`; the
compatibility layer emits deprecation warnings but generates, compiles, loads,
and solves both policies successfully.

After building acados with shared libraries, install its Python interface into
the active environment and export:

```bash
export ACADOS_SOURCE_DIR=/absolute/path/to/acados
# Linux
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
# macOS, use this instead of LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${DYLD_LIBRARY_PATH:-}"
python -m pip install -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"
```

Run acados's `minimal_example_ocp.py` before a SICNav notebook. A successful
Python import alone does not prove that the shared libraries and generated-code
toolchain work. On macOS, `ACADOS_SOURCE_DIR` is sufficient for the tested
loader, but adding the acados `lib` directory to `DYLD_LIBRARY_PATH` is still
the explicit and portable setup.

## Experiment Hygiene

- Use identical case IDs for every policy in a comparison.
- Keep the JMID fidelity comparison at three humans first. Higher cardinality
  is a separate stress/generalization experiment.
- Do not mix result files produced with ORCA and SFM humans without retaining
  the human-policy column.
- Treat `RUN_MISSING=True` in the comparison notebook as an expensive action.
- `07_stress_benchmark.ipynb` checkpoints each episode and uses identical case
  IDs across policies. Its default 30 cases improve precision but are not a
  substitute for confidence intervals or a larger final run.
- Three humans is the released JMID fidelity condition. Other human counts in
  the stress notebook are labeled out-of-distribution controller stress tests.
- Changing policy dimensions, human count, obstacle count, or solver code can
  require deleting the corresponding stale `acados_cache` entry.
- Report success, collision, wall collision, timeout/freezing, clearance,
  navigation time, path efficiency, smoothness, and policy runtime together.

## Verification Performed

- Every notebook parses as valid notebook JSON.
- `navigation.py`, `baseline.py`, and the modified MID loader compile.
- DWA completed a bottleneck episode with the maintained RVO2 binding.
- MPC-CVMM completed a bottleneck episode through CasADi/IPOPT.
- The official acados minimal OCP generated, compiled, loaded, and converged.
- SICNav-CVG completed a three-human bottleneck episode with zero collisions.
- The complete released JMID forecaster loaded under PyTorch 2.14, performed
  inference on CPU, and drove SICNav-JMID through a three-human bottleneck
  episode with zero collisions.
- acados occasionally reports QP failure or maximum-iteration statuses on this
  nonlinear problem. The policy's existing recovery path kept both smoke tests
  running; retain these diagnostics when interpreting larger comparisons.
