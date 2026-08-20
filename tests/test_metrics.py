from src.metrics.energy_efficiency import energy_efficiency

def test_energy_efficiency():
    assert energy_efficiency(100.0, 10.0) == 10.0
