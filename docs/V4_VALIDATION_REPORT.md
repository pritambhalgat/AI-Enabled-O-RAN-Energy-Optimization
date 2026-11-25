# Version 4 validation report — corrected software integration smoke test

## Code and test checks
- `python -m compileall -q config.py src tests run_all.py`: passed.
- Unit tests: **6 passed**.
- Tests cover revised receiver-rate calculations, conditional Gaussian posterior shape and positive-semidefinite covariance, posterior rate-quantile output, conformal quantile handling, block-maximum calibration score construction and power-budget handling.

## End-to-end integration run
The revised source completed one deliberately small integration run:
- Demand model: 2 training epochs.
- Each MRC, ZF and MMSE PPO agent: 2 episodes × 3 steps.
- Channel calibration: 20 independent channel realizations per receiver.
- Evaluation: 2 test seeds × 3 blocks per seed.
- The pipeline produced per-seed metrics, summary files, traffic-bound coverage and method-specific channel lower-bound coverage.

The demand model MAE in this tiny run was **0.035915**, compared with **0.029443** for persistence. The MLP did not beat persistence. Block-maximum empirical calibration coverage on the 20 calibration units was 100% for the user/receiver pairs in this run. At this sample size, the 95% finite-sample quantile is the sample maximum; this is not evidence of 95% out-of-sample coverage.

The two-seed, six-block evaluation is an integration check only. It is too small to support performance comparisons; its confidence intervals are not research-grade. Some shield projections failed, as expected for a local solver on non-convex constraints. Solver success is not proof of global feasibility or robust QoS satisfaction.

## Corrections included
1. Replaced heuristic CSI perturbations with the conditional complex-Gaussian posterior implied by the channel prior and CSI-error covariance.
2. Replaced mean-normalized path loss with a close-in 1 m path-loss model at 3.5 GHz and log-normal shadowing; updated noise to an approximately -94 dBm thermal-noise-plus-noise-figure assumption over 20 MHz.
3. Changed channel calibration to one component-wise block-maximum score vector per independent channel realization and receiver, over four predeclared action families.
4. Added held-out lower-bound coverage diagnostics separately for evaluated action methods.
5. Clarified that previous-command state matches unshielded PPO training; feeding shielded/applied power back into the state requires retraining.
6. Published the complete source archive and added CI to extract, compile and test that archive.

## Not established
- The demand MLP has not demonstrated reliable improvement over persistence.
- Calibration does not guarantee coverage for arbitrary projected actions, adaptive policies, distribution shifts or simultaneous all-user coverage.
- The shield is not guaranteed feasible; its fallback preserves the power budget only.
- PPO superiority, positive energy savings, measured hardware power reduction, live O-RAN operation, A1/E2 integration and production xApp/rApp deployment are not established.

## Required before research or publication claims
- Use separate independent training, calibration, validation and test trajectories/seeds; quantify variation across multiple training seeds.
- Evaluate at least 10–20 held-out test seeds and hundreds or thousands of channel blocks per scenario.
- Increase calibration units and report per-user/receiver coverage with uncertainty; assess shield-selected actions.
- Compare persistence, equal allocation, channel heuristic, raw PPO, uncalibrated shield, calibrated shield, constrained PPO/Lagrangian approaches and optimization baselines.
- Report QoS violation probability, shortfall, solver failure/runtime, throughput, assumed energy efficiency, fairness, transmit power and robustness under CSI/traffic/path-loss shifts.
- Validate link budget and circuit-power assumptions against a target deployment, or retain simulation-only labels.
- Complete a systematic literature review before claiming novelty.
