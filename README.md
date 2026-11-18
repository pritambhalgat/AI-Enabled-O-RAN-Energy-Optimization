# AI-Enabled O-RAN Energy Optimization

Research-oriented Python simulation framework for AI-assisted energy-efficient
O-RAN Massive MIMO resource allocation. The repository is organized into **two
versions**, following the two-notebook structure of the Massive MIMO 64×8 project.

## Version 1 — Baseline simulation

Notebook: [ORAN_Energy_Optimization.ipynb](ORAN_Energy_Optimization.ipynb)

Reproduces the original simulation: 64 antennas, 8 users, Rayleigh fading,
fixed equal power versus a channel-aware heuristic, Shannon sum-throughput and
energy efficiency. This version is the non-AI reference baseline.

Run the equivalent Python implementation:
```bash
pip install -r requirements.txt
python -m src.simulate
```

## Version 2 — Advanced AI/ML + DRL

Notebook: [ORAN_Energy_Optimization_Version2.ipynb](ORAN_Energy_Optimization_Version2.ipynb)

Full Python package:
[`experiments/AI-Enabled-O-RAN-ML-DRL-QoS-Optimization/`](experiments/AI-Enabled-O-RAN-ML-DRL-QoS-Optimization/)

Includes a PyTorch traffic-prediction MLP, PPO power allocation, explicit MRC
multi-user interference, traffic-dependent QoS, throughput, energy efficiency,
fairness, and comparisons against equal allocation, channel heuristic and SLSQP.

Run from the Version 2 package directory:
```bash
pip install -r requirements.txt
python -m src.train_traffic
python -m src.train_ppo
python -m src.evaluate --episodes 5 --blocks 100 --seed 29
pytest -q
```

## Comparison

| Feature | Version 1 | Version 2 |
|---|---|---|
| Channel | Rayleigh fading | Pathloss + shadowing + Rayleigh |
| Power allocation | Equal / heuristic | PPO continuous allocation + baselines |
| AI/ML | No | Traffic MLP |
| DRL | No | PPO |
| Multi-user interference | Simplified effective-gain SINR | Explicit MRC interference |
| QoS objective | No per-user QoS | Traffic-dependent minimum-rate targets |
| Main metrics | Throughput, power, EE | Throughput, EE, QoS, outage, fairness, power |

## O-RAN architecture context

The repository is a simulation study and architecture exploration. It does not
yet implement a production Near-RT RIC, Non-RT RIC, A1/E2 interfaces, xApp/rApp
deployment or vendor O-RU/O-DU interoperability.

## Research integrity

Do not infer that Version 2 outperforms Version 1 from model design alone.
Use multi-seed validation, confidence intervals, ablations and realistic power/
channel assumptions. See the Version 2 validation report for smoke-test results
and limitations.
