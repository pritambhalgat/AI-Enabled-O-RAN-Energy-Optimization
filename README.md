# AI-Enabled O-RAN Energy Optimization

Research simulation project exploring AI-assisted uplink radio-resource allocation, receiver-aware QoS and energy-efficiency trade-offs.

## Project evolution

### Version 1 — Baseline simulation
`ORAN_Energy_Optimization.ipynb`: fixed/equal power allocation versus a channel-aware heuristic using a basic Rayleigh-fading model.

### Version 2 — ML and PPO resource allocation
`ORAN_Energy_Optimization_Version2.ipynb` and `experiments/AI-Enabled-O-RAN-ML-DRL-QoS-Optimization/`: synthetic traffic prediction, PPO-based power allocation, MRC multi-user interference and traffic-dependent QoS targets.

### Version 3 — Receiver-aware QoS-constrained ML/DRL
`ORAN_Energy_Optimization_Version3.ipynb` and `experiments/AI-Enabled-O-RAN-ML-DRL-QoS-Optimization-Version3/`: MRC/ZF/MMSE post-combining SINR, receiver-specific PPO policies, SLSQP QoS projection attempts and multi-seed evaluation.

### Version 4 — Dual-uncertainty QoS safety-shield research prototype
- Notebook: [`ORAN_Energy_Optimization_Version4.ipynb`](ORAN_Energy_Optimization_Version4.ipynb)
- Complete source archive: [`O-RAN-Version4-Dual-Uncertainty-QoS-Safe-DRL.tar.gz`](source_packages/O-RAN-Version4-Dual-Uncertainty-QoS-Safe-DRL.tar.gz)
- Research rationale: [`docs/V4_RESEARCH_NOVELTY.md`](docs/V4_RESEARCH_NOVELTY.md)
- Validation report: [`docs/V4_VALIDATION_REPORT.md`](docs/V4_VALIDATION_REPORT.md)

V4 investigates conservative demand targets, a conditional Gaussian channel posterior under imperfect CSI, receiver-specific lower-tail rates and an SLSQP nearest-action filter for PPO power commands. The posterior is derived from the simulator's channel prior and CSI-error covariance. Calibration uses one block-maximum residual per independent channel realization across a predeclared candidate-action family, rather than treating dependent actions on the same channel as independent samples.

**V4 is a research candidate, not a demonstrated safety guarantee or a verified first-of-its-kind contribution.** Calibration does not guarantee coverage for arbitrary shield-projected actions, adaptive policy changes or distribution shifts. SLSQP is a local solver and may fail. Its equal-power fallback preserves the power budget only; QoS remains unverified.

## Reproduce Version 4

From the repository root, extract the committed archive under `experiments/`:

```bash
mkdir -p experiments
tar -xzf source_packages/O-RAN-Version4-Dual-Uncertainty-QoS-Safe-DRL.tar.gz -C experiments
cd experiments/AI-Enabled-O-RAN-ML-DRL-QoS-Optimization-Version4
python -m pip install -r requirements.txt
python -m compileall -q config.py src tests run_all.py
python -m pytest -q
```

A deliberately small software-integration run (not suitable for research claims):

```bash
python run_all.py --seeds 2 --blocks 3 --traffic-epochs 2 \
  --calibration-samples 20 --ppo-episodes 2 --ppo-steps 3
```

## Scope and limitations

- Synthetic traffic and a single-cell narrowband uplink are used; no operator traces or live O-RAN interface are present.
- The link budget uses an illustrative 3.5 GHz carrier, 20 MHz bandwidth and thermal-noise-plus-noise-figure assumption. The fixed circuit-power value is illustrative, not measured hardware consumption.
- The demand MLP has not demonstrated reliable improvement over persistence in the reported smoke run.
- No measured RAN energy savings, PPO superiority, production xApp/rApp, live A1/E2 integration, deterministic QoS guarantee or formal all-user safety guarantee is claimed.
- Further work requires larger independent training/calibration/test sets, constrained-RL and optimization baselines, runtime analysis, deployment-oriented power-model validation and a systematic literature review.

The V1–V3 notebooks and experiment packages are retained to show the staged development history. V4 is the most complete implementation, but its current validation remains at software-smoke-test level.
