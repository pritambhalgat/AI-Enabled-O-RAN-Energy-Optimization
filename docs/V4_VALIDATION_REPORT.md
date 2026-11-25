# Version 4 validation report — software smoke tests

- Python syntax compilation passed for all 13 Python source files.
- Unit tests: 4 passed.
- Nominal end-to-end pipeline completed: demand MLP, three receiver-specific PPO agents, action-matched channel calibration and evaluation.
- Stress-test evaluation completed with CSI error multiplier 1.5 and traffic-surge probability 0.2.
- Smoke settings: 2 PPO episodes × 5 steps per receiver, 12 channel-calibration samples and 2 test seeds × 2 blocks.
- Separate 20-epoch demand-model run: MLP held-out MAE 0.031085; persistence MAE 0.029443. The MLP did not beat persistence.
- One-sided demand upper-bound empirical coverage ranged from 0.9407 to 0.9620 across users in that split.
- Tiny nominal channel lower-rate-bound check: mean coverage 93.75%, minimum 50%; only four test samples per user/receiver, so this is not a reliable coverage estimate.

PPO agents are severely under-trained in the smoke run. Some projections failed. SLSQP is local on non-convex constraints; its fallback respects the power budget only and is not guaranteed to meet QoS.

Before research claims:
- Train each receiver policy with multiple independent training seeds.
- Use disjoint train, calibration, validation and test data.
- Evaluate at least 10 test seeds and hundreds/thousands of channel blocks.
- Report per-user demand and rate-bound coverage, QoS violation probabilities, shortfall, shield failure, solver runtime, throughput, EE, fairness and power.
- Stress test CSI error, traffic bursts, shadowing, user geometry and heavy-tailed errors.
- Do not claim deterministic or simultaneous all-user safety, PPO superiority, live O-RAN deployment or measured RAN energy savings.
