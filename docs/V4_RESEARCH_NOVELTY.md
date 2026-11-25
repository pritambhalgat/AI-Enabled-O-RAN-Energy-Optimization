# Version 4 research rationale and candidate contribution

## Working title
Dual-Calibrated QoS-Safe Reinforcement Learning for Receiver-Aware O-RAN Uplink Power Control under Traffic and CSI Uncertainty.

## Why this is not just another PPO QoS controller
QoS-aware O-RAN DRL and safe wireless RL already exist. Therefore, PPO plus a QoS reward is not claimed as novel. V4 investigates the narrower combination of two uncertainty calibrations and a receiver-conditioned action shield.

## Candidate contribution
1. **Traffic uncertainty:** train a demand MLP and derive per-user one-sided split-conformal upper residual quantiles on a disjoint calibration set. Use the conservative demand estimate to set the minimum-rate target.
2. **CSI/rate uncertainty:** represent imperfect CSI as H_hat=H+E; draw posterior channel scenarios; calculate receiver-specific lower-tail post-combining rates for MRC, ZF and MMSE; calibrate receiver/user lower-bound over-estimation margins using independent samples that include trained-policy actions.
3. **Action shield:** project the PPO-proposed power vector to the closest vector that satisfies calibrated per-user rate lower bounds and the sum-power budget.
4. **Reliability evaluation:** compare raw PPO, uncalibrated posterior shielding and dual-calibrated shielding using per-user coverage, QoS outage, shortfall, solver failure, power, energy efficiency, throughput, fairness and confidence intervals.
5. **Shift analysis:** stress test traffic surges, increased CSI error, user geometry and channel-error distributions.

## Formulation
For user k:
d_upper,k = clip(d_hat,k + q_d,k, 0, 1)
R_min,k = R_base + R_span*d_upper,k.

For receiver r, let L_k,r(p) be a lower posterior quantile of rates from plausible channels around H_hat. Calibrate q_c,k,r on an independent set using residual scores between this lower bound and realised rate:
LCB_k,r(p)=L_k,r(p)-q_c,k,r.

The shield solves:
minimize_p ||p-p_PPO||^2
subject to sum_k p_k <= P_max, p_k >= 0,
and LCB_k,r(p) >= R_min,k.

The interference-coupled constraints are non-convex; SLSQP is a local solver, not a feasibility proof. On failure the code uses a power-budget-feasible fallback and reports QoS as unverified.

## Research questions
- Does the dual-calibrated shield reduce per-user QoS violations versus raw PPO and an uncalibrated posterior shield?
- Do calibration margins and failure rates differ across MRC, ZF and MMSE?
- How do traffic and rate-bound coverage degrade under distribution shift?
- What throughput and energy-efficiency cost is paid for conservative QoS enforcement?

## Related work to review
- Yan et al., “Near-Real-Time Resource Slicing for QoS Optimization in 5G O-RAN using Deep Reinforcement Learning,” arXiv:2509.14343 (2025), https://arxiv.org/abs/2509.14343
- “RSLAQ—A Robust SLA-Driven 6G O-RAN QoS xApp Using Deep Reinforcement Learning,” IEEE Transactions on Mobile Computing (2026), DOI:10.1109/TMC.2026.3666787
- Peng et al., “Enabling Safety-Critical Wireless Communications via Safe Reinforcement Learning,” arXiv:2604.17032 (2026)
- “Deep Reinforcement Learning Based Subchannel Selection and Power Allocation in Wireless Networks with Imperfect CSI,” IEEE VTC 2023, DOI:10.1109/VTC2023-Spring57618.2023.10199481
- Cohen et al., “Calibrating AI Models for Wireless Communications via Conformal Prediction,” IEEE TMLCN (2023)

This is a candidate research direction, not a verified priority claim. Before publication, expand the literature search, implement constrained-RL/Lagrangian baselines, use separate train/calibration/validation/test seeds, and verify the calibration assumptions under temporal dependence and policy-induced action distribution shift.
