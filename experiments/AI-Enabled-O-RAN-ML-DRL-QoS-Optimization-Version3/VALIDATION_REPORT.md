# Version 3 validation report

- Python syntax compilation: passed.
- Unit tests: 3 passed.
- Traffic MLP completed 20 epochs. On one synthetic held-out split, MAE=0.028954
  versus persistence MAE=0.030351. This is a small single-seed gain only.
- PPO smoke training completed for MRC, ZF and MMSE using 2 episodes x 5 steps each.
- The multi-receiver evaluation completed for 2 independent seeds x 2 blocks.
- QoS projection returned success for 91.7% of 12 projected samples in this tiny run.
  This is not a reliable estimate of general feasibility.

The PPO agents in the smoke run are severely under-trained. Do not present these
metrics as evidence of PPO performance or superiority. Run the configured PPO
training schedule and evaluate with more independent training and test seeds.
Report throughput, EE, QoS satisfaction, outage, shortfall, total power and
projection success together.

The QoS constraints are non-convex; local SLSQP does not provide a global
feasibility guarantee. Scope remains single-cell narrowband flat fading with
synthetic traffic and assumed powers. No live O-RAN interface or hardware
validation is implemented.