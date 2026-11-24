# AI-Enabled O-RAN Energy Optimization

This repository contains four staged research versions.

## Version 1 — Baseline
ORAN_Energy_Optimization.ipynb: fixed equal power vs channel-aware heuristic,
Rayleigh fading and basic throughput/energy-efficiency calculations.

## Version 2 — PyTorch ML + PPO
ORAN_Energy_Optimization_Version2.ipynb and the code in
experiments/AI-Enabled-O-RAN-ML-DRL-QoS-Optimization/. The package includes
traffic MLP, PPO power allocation, MRC multi-user interference, traffic-based
QoS targets and conventional allocation comparisons.

## Version 3 — Receiver-aware QoS-constrained ML/DRL
ORAN_Energy_Optimization_Version3.ipynb and package in
experiments/AI-Enabled-O-RAN-ML-DRL-QoS-Optimization-Version3/.
V3 adds receiver-specific MRC/ZF/MMSE SINR, separate PPO policies, numerical
QoS projection attempts, multi-seed comparisons and confidence intervals.

## Version 4 — Dual-calibrated QoS safety shield (research candidate)
ORAN_Energy_Optimization_Version4.ipynb.

V4 investigates a two-sided uncertainty-aware safety layer around PPO:
1. One-sided split-conformal upper bounds on predicted user traffic.
2. Receiver-specific lower-tail rate bounds under imperfect CSI, calibrated
   using an independent sample set and actions from the trained policies.
3. A nearest-action SLSQP projection that attempts to satisfy the conservative
   per-user rate targets and the total transmit-power constraint.
4. Evaluation of raw PPO, an uncalibrated posterior shield and the
   dual-calibrated shield, including nominal and distribution-shift tests.

**This is a candidate research contribution, not a verified first-of-its-kind
claim or a demonstrated safety guarantee.** The local SLSQP problem is
non-convex; it can fail. Conformal marginal coverage depends on calibration
assumptions and may degrade under distribution shift or policy-induced
distribution change. A systematic literature review and research-grade
experiments are required.

### Version 4 package
Download the complete source package from the accompanying ChatGPT deliverable:
`O-RAN-Version4-Dual-Uncertainty-QoS-Safe-DRL.zip`.
The bundle contains the modular Python implementation, tests, methodology and
validation report. Extract it, enter the `O-RAN-Version4` directory and run:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_all.py --seeds 5 --blocks 50
```

The included smoke validation passed 4 unit tests, but the full PPO agents and
multi-seed study still need to be run. The MLP did not beat persistence on the
reported synthetic split. No measured RAN energy savings, PPO superiority,
live A1/E2 interface or production xApp deployment is claimed.
