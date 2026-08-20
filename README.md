# AI-Enabled O-RAN Energy Optimization

Research-oriented Python simulation framework for **AI-enabled, energy-efficient Open RAN (O-RAN) Massive MIMO**.

## Research objective

> Can an AI-enabled O-RAN RIC dynamically allocate transmit power and radio resources in a Massive MIMO network to maximize energy efficiency while maintaining QoS?

## Architecture

```text
                         ┌──────────────────────┐
                         │         SMO          │
                         │ Policies / Mgmt      │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │      Non-RT RIC      │
                         │ AI / rApp / Policy    │
                         └──────────┬───────────┘
                                    │ A1
                         ┌──────────▼───────────┐
                         │      Near-RT RIC      │
                         │ AI / xApp / Control   │
                         └──────────┬───────────┘
                                    │ E2
                         ┌──────────▼───────────┐
                         │         O-DU         │
                         │ MAC / Scheduling /   │
                         │ Massive MIMO PHY     │
                         └──────────┬───────────┘
                                    │ Open Fronthaul
                         ┌──────────▼───────────┐
                         │         O-RU         │
                         │ RF / Low-PHY / Array │
                         └──────────┬───────────┘
                                    │
                                  UEs
```

## Roadmap

1. Reproducible Massive MIMO baseline.
2. O-RU/O-DU/O-CU abstractions.
3. Near-RT RIC xApp.
4. Non-RT RIC rApp.
5. Fixed vs optimization vs ML vs RL power control.
6. Energy-efficiency, throughput, spectral-efficiency, latency and QoS evaluation.

## Run

```bash
pip install -r requirements.txt
python -m src.simulate
```

Results are written to `results/`.

**Important:** this is a research simulation framework, not a claim of production O-RAN certification or vendor-interface compliance.
