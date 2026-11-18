# Initial software validation report

- Python syntax compilation: completed successfully for project source files.
- Unit tests: 3 passed.
- Traffic MLP trained successfully.
- On one synthetic held-out split, MLP MAE was 0.028954 and persistence MAE was
  0.030351. This is a small single-seed gain, not evidence of general prediction ability.
- PPO training and evaluation pipeline completed in a deliberately short smoke run:
  2 training episodes x 8 steps, then 1 evaluation episode x 10 blocks.
- Illustrative short-run mean values: PPO throughput 371.99 Mbps, EE 3.25 Mbps/W,
  QoS satisfaction 83.75%; SLSQP 573.50 Mbps, 5.55 Mbps/W, 91.25%; equal allocation
  373.69 Mbps, 2.99 Mbps/W, 85.00%. The short-run PPO policy did not beat SLSQP
  and did not improve QoS over equal allocation.

These are pipeline smoke-test metrics only. Do not cite them as research findings.
Required work: train the full configured PPO run; test 5-10 seeds and hundreds/thousands
of blocks; report confidence intervals; tune reward on validation seeds; freeze weights
before testing; run predictor and reward ablations. Simulator limitations are stated in
README: MRC-only, narrowband flat fading, illustrative pathloss, synthetic demand,
assumed power model, no live O-RAN/A1/E2/xApp integration.
