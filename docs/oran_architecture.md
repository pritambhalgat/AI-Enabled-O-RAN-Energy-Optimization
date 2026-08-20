# O-RAN Architecture Mapping

| O-RAN element | Simulation role |
|---|---|
| SMO | High-level policy/configuration |
| Non-RT RIC | Long-timescale policy generation |
| rApp | Policy/analytics logic |
| Near-RT RIC | Near-real-time control |
| xApp | Power/resource optimization |
| O-CU | Higher-layer abstraction |
| O-DU | Scheduling + PHY/MAC abstraction |
| O-RU | Radio/antenna abstraction |
| A1 | Policy path abstraction |
| E2 | Control/telemetry abstraction |
| Open Fronthaul | RU-DU functional separation |

The project models logical O-RAN roles; it does not claim conformance to a particular O-RAN specification release.
