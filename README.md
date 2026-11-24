# AI-Enabled O-RAN Energy Optimization

This repository contains three staged research versions.

## Version 1 — Baseline
ORAN_Energy_Optimization.ipynb: fixed equal power vs channel-aware heuristic,
Rayleigh fading and basic throughput/energy-efficiency calculations.

## Version 2 — PyTorch ML + PPO
ORAN_Energy_Optimization_Version2.ipynb and the code in
experiments/AI-Enabled-O-RAN-ML-DRL-QoS-Optimization/. The package includes
traffic MLP, PPO power allocation, MRC multi-user interference, traffic-based
QoS targets and comparisons with equal/heuristic/SLSQP methods.

## Version 3 — Receiver-aware QoS-constrained ML/DRL
ORAN_Energy_Optimization_Version3.ipynb and package in
experiments/AI-Enabled-O-RAN-ML-DRL-QoS-Optimization-Version3/.

V3 implements actual post-combining SINR for MRC, ZF and MMSE, separate PPO
policies per receiver, SLSQP QoS projection attempts, multi-seed comparisons,
95% confidence intervals, unit tests and reproducible outputs.

Run V3 from its package directory:
    python -m pip install -r requirements.txt
    python -m src.train_traffic
    python -m src.train_ppo --receiver MRC
    python -m src.train_ppo --receiver ZF
    python -m src.train_ppo --receiver MMSE
    python -m src.evaluate --seeds 5 --blocks 50
    pytest -q

QoS projection is a local solver on non-convex constraints and may fail. PPO
is not guaranteed to outperform conventional optimization. The project remains
a single-cell narrowband simulator with synthetic traffic and assumed power
parameters, not a live O-RAN deployment or calibrated hardware study.
