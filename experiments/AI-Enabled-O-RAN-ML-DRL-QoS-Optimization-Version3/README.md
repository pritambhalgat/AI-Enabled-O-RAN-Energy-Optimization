# O-RAN Resource Allocation — Version 3

Receiver-aware QoS-constrained ML/DRL extension to Version 2. V1 and V2 remain intact.

## Implemented
- Actual post-combining SINR for MRC, ZF and linear-MMSE combiners.
- Supervised PyTorch traffic MLP measured against persistence.
- One PPO actor-critic trained per receiver.
- SLSQP projection attempts to minimally modify PPO power while satisfying each
  per-user rate target and the total transmit-power budget.
- Evaluates equal power, channel heuristic, SLSQP reward optimization, raw PPO
  and QoS-projected PPO.
- Multi-seed evaluation and 95% Student-t confidence intervals.
- Unit tests and CSV outputs.

## Equations
Received signal: y = H diag(sqrt(p)) x + n.
For W (K x M) and G = W H:
SINR_k = p_k |G_kk|^2 /
         (sum_(j != k) p_j |G_kj|^2 + sigma2 ||w_k||^2).
R_k = B log2(1 + SINR_k).
P_total = P_static + P_circuit + sum(p_k)/eta_PA.
EE = sum(R_k)/P_total.

QoS projection minimizes ||p - p_PPO||^2 subject to each R_k >= Rmin_k,
p_k >= 0, and sum(p_k) <= Pmax. These interference-coupled constraints are
non-convex. SLSQP is a local numerical method; projection can fail. The code
records solver success and reports actual achieved QoS; it does not guarantee feasibility.

## Run
    python -m pip install -r requirements.txt
    python -m src.train_traffic
    python -m src.train_ppo --receiver MRC
    python -m src.train_ppo --receiver ZF
    python -m src.train_ppo --receiver MMSE
    python -m src.evaluate --seeds 5 --blocks 50
    pytest -q

Or run all: python run_all.py --seeds 5 --blocks 50

## Outputs
artifacts/traffic_mlp.pt and traffic_norm.npz
artifacts/ppo_MRC.pt, ppo_ZF.pt, ppo_MMSE.pt
results/per_seed.csv and results/summary_ci95.csv

## Assumptions and limitations
Single-cell, narrowband, flat-fading uplink; illustrative normalized pathloss and
shadowing; synthetic demand; assumed power parameters. No OFDM, mobility, pilot
estimation/contamination, live A1/E2, production xApp/rApp, vendor interfaces or
calibrated O-RU/DU/CU/fronthaul power. PPO is not guaranteed to beat numerical
optimization. Multi-seed validation is required but does not by itself prove deployment gains.