import numpy as np
from config import Config
from src.radio import generate_channel,make_combiner,user_rates,metrics
from src.optimization import project_to_qos

def test_receiver_sinr_shapes_and_finiteness():
    cfg=Config()
    H=generate_channel(cfg.n_antennas,cfg.n_users,np.random.default_rng(7))
    p=np.ones(cfg.n_users)*cfg.p_max_w/cfg.n_users
    for receiver in ("MRC","ZF","MMSE"):
        W=make_combiner(H,p,cfg.noise_power_w,receiver)
        sinr,rates,interference,_=user_rates(H,p,cfg,receiver)
        assert W.shape==(cfg.n_users,cfg.n_antennas)
        assert sinr.shape==(cfg.n_users,)
        assert np.all(np.isfinite(sinr)) and np.all(sinr>=0)
        assert np.all(rates>=0)

def test_power_budget_and_metrics():
    cfg=Config()
    H=generate_channel(cfg.n_antennas,cfg.n_users,np.random.default_rng(11))
    p=np.ones(cfg.n_users)*cfg.p_max_w/cfg.n_users
    m=metrics(H,p,cfg,np.ones(cfg.n_users)*cfg.min_rate_base_bps,"MRC")
    assert np.sum(p)<=cfg.p_max_w+1e-10
    assert 0<=m["qos_satisfaction"]<=1
    assert m["total_power_w"]>=cfg.fixed_power_w

def test_qos_projection_reports_feasibility_flag():
    cfg=Config()
    H=generate_channel(cfg.n_antennas,cfg.n_users,np.random.default_rng(12))
    p0=np.ones(cfg.n_users)*cfg.p_max_w/cfg.n_users
    rmin=np.ones(cfg.n_users)*cfg.min_rate_base_bps
    p,ok,msg=project_to_qos(H,p0,cfg,rmin,"ZF",maxiter=20)
    assert np.all(p>=0) and np.sum(p)<=cfg.p_max_w+1e-6
    assert isinstance(ok,bool) and isinstance(msg,str)
