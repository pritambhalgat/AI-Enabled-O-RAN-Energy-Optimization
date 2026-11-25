# Version 4 research rationale and candidate contribution

## Working title
**Dual-Calibrated QoS-Aware Reinforcement Learning for Receiver-Aware O-RAN Uplink Power Control under Traffic and CSI Uncertainty**

## Research question
Can a receiver-aware PPO power-allocation policy, filtered by an uncertainty-calibrated action shield, reduce user-level QoS violations under uncertain traffic demand and imperfect CSI while accounting for throughput, power and fairness costs?

## Implemented components
1. **Traffic uncertainty.** A PyTorch MLP forecasts normalized synthetic per-user demand. One-sided split-conformal residual quantiles form conservative demand upper estimates. Persistence remains an essential baseline.
2. **Conditional CSI posterior.** The channel prior covariance is (C_k=\beta_k R), and the CSI error covariance is \(\Sigma_{e,k}=\beta_k\sigma_{CSI}^2I\). For the proper complex-Gaussian model, the conditional posterior is
   \[
   \mu_{k|\widehat H}=C_k(C_k+\Sigma_{e,k})^{-1}\widehat H_k,
   \qquad
   C_{k|\widehat H}=C_k-C_k(C_k+\Sigma_{e,k})^{-1}C_k.
   \]
   Samples are drawn from this conditional distribution instead of applying an arbitrary perturbation around the estimate.
3. **Receiver-aware lower-tail rates.** MRC, ZF and MMSE combining are evaluated with post-combining multi-user interference and receiver noise enhancement. Lower-tail rate quantiles are estimated over posterior channel samples.
4. **Channel calibration.** For each receiver, one independent channel realization contributes one calibration score vector. The score is the component-wise maximum across four predeclared power-allocation families: equal, channel heuristic, deterministic PPO and random. This avoids counting dependent actions on the same channel realization as independent calibration samples.
5. **Action filter.** An SLSQP nearest-action projection attempts to meet calibrated lower-rate targets and the total-power constraint. The constraints are non-convex; SLSQP is local and does not guarantee feasibility or global optimality.
6. **Evaluation.** Raw PPO, uncalibrated shielding, calibrated shielding, equal allocation and channel-aware heuristic allocation are compared using method-specific lower-bound coverage, QoS satisfaction/outage, rate shortfall, throughput, assumed energy efficiency, Jain fairness, power and solver diagnostics.

## Candidate formulation
For user (k),
\[
d_{upper,k}=\operatorname{clip}(\widehat d_k+q_{d,k},0,1),\quad
R_{min,k}=R_{base}+R_{span}d_{upper,k}.
\]
For receiver (r), let (L_{k,r}(p)) be a lower posterior quantile of the rate. A calibrated margin (q_{c,k,r}) forms
\[
LCB_{k,r}(p)=L_{k,r}(p)-q_{c,k,r}.
\]
The shield attempts to solve
\[
\min_p\;\|p-p_{PPO}\|_2^2
\]
subject to
\[
\sum_kp_k\le P_{max},\quad p_k\ge0,\quad
LCB_{k,r}(p)\ge R_{min,k}.
\]
A failed projection falls back to equal power. This preserves the sum-power budget but does not establish QoS satisfaction.

## Calibration assumptions and limits
The block-maximum calibration score is constructed over a finite, predeclared action family for each independent channel realization and receiver. Its finite-sample interpretation relies on exchangeability of calibration and test units. It is not a guarantee for arbitrary SLSQP-projected actions, adaptive policy shifts, temporally dependent channels, distribution shifts or simultaneous all-user coverage. Shield-selected actions must therefore be assessed empirically on held-out seeds.

The PPO state uses the previous PPO command, matching the unshielded training environment. The shield is applied downstream. Feeding shielded/applied power back into subsequent policy states changes the control loop and requires retraining and revalidation.

## Candidate contribution — not a novelty claim
The candidate direction is the joint study of demand uncertainty, simulator-consistent conditional CSI uncertainty, receiver-conditioned lower-tail rates and a QoS-constrained nearest-action filter for PPO power commands. QoS-aware O-RAN DRL, safe wireless RL and conformal prediction already exist. Novelty cannot be inferred from combining these concepts; a systematic literature review and rigorous baselines are required.

## Required research questions
- Does the calibrated shield reduce per-user QoS violations versus raw PPO and an uncalibrated shield on held-out seeds?
- Is block-maximum calibration informative for shield-selected actions, or are action-conditional/trajectory-level methods needed?
- How do coverage and projection failure vary across MRC, ZF and MMSE?
- What throughput, assumed energy-efficiency and fairness costs accompany conservative QoS enforcement?
- How sensitive are results to CSI error, traffic bursts, path loss, shadowing, user geometry and heavy-tailed errors?
- How does the method compare with constrained PPO/Lagrangian approaches, safe-RL baselines and numerical optimization?

## Initial related work to review
This is an initial reading list, not a completed systematic review:
- Yan et al., “Near-Real-Time Resource Slicing for QoS Optimization in 5G O-RAN using Deep Reinforcement Learning,” arXiv:2509.14343 (2025), https://arxiv.org/abs/2509.14343
- “RSLAQ—A Robust SLA-Driven 6G O-RAN QoS xApp Using Deep Reinforcement Learning,” IEEE Transactions on Mobile Computing (2026), DOI:10.1109/TMC.2026.3666787
- Peng et al., “Enabling Safety-Critical Wireless Communications via Safe Reinforcement Learning,” arXiv:2604.17032 (2026)
- “Deep Reinforcement Learning Based Subchannel Selection and Power Allocation in Wireless Networks with Imperfect CSI,” IEEE VTC 2023, DOI:10.1109/VTC2023-Spring57618.2023.10199481
- Cohen et al., “Calibrating AI Models for Wireless Communications via Conformal Prediction,” IEEE TMLCN (2023)
