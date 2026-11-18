# PyTorch ML + PPO O-RAN Massive MIMO Resource Allocation

An experimental extension to the existing AI-Enabled O-RAN Energy Optimization
repository. This implementation is kept in a separate experiment directory so
the original baseline and its dependencies are not overwritten.

## Research question
Can synthetic traffic prediction combined with PPO continuous power allocation
improve the energy-efficiency / throughput trade-off while maintaining user QoS,
compared with equal allocation, a channel-aware heuristic and numerical SLSQP?

## Mathematical model
For M=64 base-station antennas and K=8 single-antenna uplink users:
`y = H diag(sqrt(p)) x + n`, with `n ~ CN(0, sigma2 I)`,
`p_k >= 0`, `sum_k p_k <= Pmax`.

For MRC, `w_k=h_k`:
```
SINR_k = p_k |h_k^H h_k|^2 /
  [sum_(i != k) p_i |h_k^H h_i|^2 + sigma2 ||h_k||^2]
R_k = B log2(1 + SINR_k)
R_sum = sum_k R_k
P_total = P_static + P_circuit + sum_k p_k / eta_PA
EE = R_sum / P_total  [bit/J]
```

Traffic-dependent minimum-rate target:
`Rmin,k = Rmin,base + (Rmin,max-Rmin,base)*predicted_demand_k`.

PPO reward combines normalized EE and throughput, QoS outage penalty and power
switching penalty. Report every metric independently; reward alone is not a result.

## ML + DRL architecture
* Supervised PyTorch MLP forecasts per-user synthetic demand for the next slot.
* PPO actor-critic observes normalized channel gains, predicted demand and previous
  power allocation.
* Continuous action chooses K relative user shares plus a total-power fraction.
  Softmax and sigmoid transforms guarantee nonnegative powers and respect Pmax.
* Evaluation compares equal power, channel heuristic, SLSQP numerical optimization,
  and PPO on the same channel/traffic realizations.

## Install and run
```bash
pip install -r requirements.txt
python -m src.train_traffic
python -m src.train_ppo
python -m src.evaluate --episodes 5 --blocks 100 --seed 29
pytest -q
```
Run commands from this folder. Outputs go to `artifacts/` and `results/`.

## Scope and limitations
Single-cell narrowband flat-fading uplink, MRC receiver, illustrative pathloss and
shadowing, synthetic demand and assumed power model. This is not a standards-
compliant O-RAN RIC implementation. No A1/E2 interface, live xApp/rApp, OFDM,
mobility, pilot-based CSI estimation, pilot contamination, ZF/MMSE or calibrated
RU/DU/CU/fronthaul energy. PPO is not guaranteed to beat SLSQP or other baselines.
Do not claim deployment or measured energy savings from this simulator.

## Reproducibility
Save the seed, config, package versions and all evaluation CSVs. Use several
independent seeds and report confidence intervals. Tune reward only on validation
seeds; freeze it before final test runs. Include predictor-vs-persistence and
PPO ablations before making research claims.
