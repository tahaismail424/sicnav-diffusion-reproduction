# Crowd Navigation Experiments

This directory introduces closed-loop CrowdSimPlus evaluation separately from
JMID and SICNav's MPC machinery.

## Setup

From the repository root:

```bash
conda env create -f experiments/crowd_navigation/environment.yml
conda activate sicnav_crowdsim_env
python -m pip install -e trajectory_prediction/RVO2
python -c "import gymnasium, rvo2; print(rvo2.__version__, rvo2.__file__)"
```

The final command should print an extension path ending in `.so` on Linux or
macOS (`.pyd` on Windows). That confirms Python loaded compiled code rather than
a similarly named source file.

The environment includes CasADi for MPC-CVMM. SICNav-CVG and SICNav-JMID also
require a separately compiled acados runtime and its `acados_template` package.
The locally tested setup uses the existing `sicnav_mid_env`, acados source at
`/Users/tahaismail/Desktop/work/lyu_lab/acados`, and the exports in the user's
Bash profile. Select that environment as the notebook kernel. See
`IMPLEMENTATION_NOTES.md` for runtime variables and compatibility changes.

## Notebooks

1. `01_crowdsim_baseline.ipynb`: linear and ORCAPlus simulator baselines.
2. `02_dwa.ipynb`: sampled dynamic-window control with a unicycle robot.
3. `03_mpc_cvmm.ipynb`: CasADi MPC with constant-velocity human forecasts.
4. `04_sicnav_cvg.ipynb`: acados SICNav with constant-velocity human goals.
5. `05_sicnav_jmid.ipynb`: released JMID checkpoint plus SICNav.
6. `06_policy_comparison.ipynb`: paired policy/scenario aggregation and plots.
7. `07_stress_benchmark.ipynb`: resumable repeated-case scenario and crowd-density
   sweeps with uncertainty intervals and matched-condition comparisons.

The policy notebooks write CSVs to `outputs/policy_benchmark/`. The comparison
notebook reads existing CSVs by default and only launches missing runs when
`RUN_MISSING=True`, preventing accidental reruns of expensive controllers.
The first SICNav-JMID run generates and compiles a large acados solver; the
local first build took about five minutes. Later runs reuse the cache.

The stress notebook deliberately avoids a full scenario-by-density Cartesian
product. It evaluates every built-in scenario at three humans, then varies
human count in `hallway_static_with_back`. Its separate output directory is
checkpointed after every episode, so an interrupted multi-hour run can resume.

The RVO2 package build and binding design are documented in
`trajectory_prediction/RVO2/PYTHON_BINDINGS.md`.
