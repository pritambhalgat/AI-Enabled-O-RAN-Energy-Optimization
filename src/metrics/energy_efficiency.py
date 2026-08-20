def energy_efficiency(throughput_bps, total_power_w):
    return throughput_bps / max(total_power_w, 1e-12)
