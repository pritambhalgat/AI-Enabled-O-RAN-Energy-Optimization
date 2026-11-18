import numpy as np
from config import Config
from src.radio import channel_realization,project_power,mrc_metrics
def test_power_projection_feasible():
    p=project_power(np.array([1.,2.,3.]),7.,10.)
    assert np.all(p>=0) and np.isclose(p.sum(),7.)
def test_mrc_metrics_finite():
    c=Config(); H=channel_realization(c.n_antennas,c.n_users,np.random.default_rng(4))
    m=mrc_metrics(H,np.ones(c.n_users)*c.p_max_w/c.n_users,c,np.ones(c.n_users)*c.min_rate_base_bps)
    assert np.all(np.isfinite(m["sinr"])) and np.all(m["sinr"]>=0)
    assert 0<=m["qos_satisfaction"]<=1
def test_zero_power_zero_rate():
    c=Config(); m=mrc_metrics(np.ones((c.n_antennas,c.n_users),complex),np.zeros(c.n_users),c,np.ones(c.n_users))
    assert np.allclose(m["rates"],0)
