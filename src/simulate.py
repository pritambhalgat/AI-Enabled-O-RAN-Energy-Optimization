import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.channel.channel_model import rayleigh_channel
from src.metrics.energy_efficiency import energy_efficiency
from src.metrics.throughput import shannon_sum_rate

def run(seed=42):
    rng = np.random.default_rng(seed)
    n_antennas, n_users = 64, 8
    bandwidth = 20e6
    noise_w = 1e-9
    circuit_w = 80.0
    max_power = 10.0
    rows = []

    for snr_db in [0, 5, 10, 15, 20]:
        H = rayleigh_channel(n_users, n_antennas, rng)
        gains = np.linalg.norm(H, axis=1) ** 2 / n_antennas
        noise_scaled = noise_w * 10 ** (-snr_db / 10)

        for mode in ["fixed", "energy_aware"]:
            if mode == "fixed":
                p = np.full(n_users, max_power / n_users)
            else:
                weights = gains / gains.sum()
                p = max_power * (0.05 / n_users + 0.95 * weights)
                p = p * max_power / p.sum()

            sinr = p * gains / noise_scaled
            throughput = shannon_sum_rate(sinr, bandwidth)
            total_power = p.sum() / 0.4 + circuit_w
            ee = energy_efficiency(throughput, total_power)

            rows.append({
                "snr_db": snr_db,
                "mode": mode,
                "throughput_bps": throughput,
                "total_power_w": total_power,
                "energy_efficiency_bit_per_joule": ee,
            })

    df = pd.DataFrame(rows)
    out = Path("results")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tables/baseline_results.csv", index=False)

    plt.figure()
    for mode, group in df.groupby("mode"):
        plt.plot(group["snr_db"], group["energy_efficiency_bit_per_joule"],
                 marker="o", label=mode)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Energy efficiency (bit/J)")
    plt.title("O-RAN Massive MIMO Energy-Efficiency Baseline")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "figures/energy_efficiency_vs_snr.png", dpi=200)
    plt.close()
    return df

if __name__ == "__main__":
    print(run().to_string(index=False))
