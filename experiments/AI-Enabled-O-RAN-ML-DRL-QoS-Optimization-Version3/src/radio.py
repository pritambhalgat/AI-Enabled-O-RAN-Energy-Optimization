import numpy as np
RECEIVERS=("MRC","ZF","MMSE")

def generate_channel(M,K,rng,pathloss_exponent=3.5,shadow_std_db=6.0):
    distance=rng.uniform(35.,250.,K)
    rel_loss=10.0*pathloss_exponent*np.log10(distance/100.0)+rng.normal(0.,shadow_std_db,K)
    beta=10.0**(-rel_loss/10.0)
    beta=beta/np.mean(beta)
    idx=np.arange(M)
    corr=.2**np.abs(idx[:,None]-idx[None,:])
    eigval,eigvec=np.linalg.eigh(corr)
    R_sqrt=eigvec@np.diag(np.sqrt(np.maximum(eigval,1e-12)))@eigvec.conj().T
    z=(rng.normal(size=(M,K))+1j*rng.normal(size=(M,K)))/np.sqrt(2.0)
    return R_sqrt@z*np.sqrt(beta)[None,:]

def make_combiner(H,p,noise_power,receiver):
    M,K=H.shape
    receiver=receiver.upper()
    if receiver=="MRC":
        return H.conj().T
    if receiver=="ZF":
        gram=H.conj().T@H
        loading=1e-10*max(1.0,float(np.real(np.trace(gram))/K))
        return np.linalg.solve(gram+loading*np.eye(K),H.conj().T)
    if receiver=="MMSE":
        Ryy=(H*p[None,:])@H.conj().T+noise_power*np.eye(M)
        X=np.linalg.solve(Ryy,H*np.sqrt(p)[None,:])
        return X.conj().T
    raise ValueError(f"Unknown receiver {receiver}; choose one of {RECEIVERS}")

def user_rates(H,p,cfg,receiver):
    W=make_combiner(H,p,cfg.noise_power_w,receiver)
    G=W@H
    desired=p*np.abs(np.diag(G))**2
    total=np.abs(G)**2@p
    interference=np.maximum(total-desired,0.)
    noise=cfg.noise_power_w*np.sum(np.abs(W)**2,axis=1)
    sinr=desired/np.maximum(interference+noise,1e-30)
    rates=cfg.bandwidth_hz*np.log2(1.0+sinr)
    return sinr,rates,interference,W

def metrics(H,p,cfg,rmin,receiver):
    sinr,rates,interference,W=user_rates(H,p,cfg,receiver)
    total_power=cfg.fixed_power_w+float(np.sum(p))/cfg.pa_efficiency
    sum_rate=float(np.sum(rates))
    ee=sum_rate/max(total_power,1e-12)
    satisfaction=float(np.mean(rates>=rmin))
    fairness=float(sum_rate**2/(cfg.n_users*np.sum(rates**2)+1e-30))
    return {"sinr":sinr,"rates":rates,"sum_rate":sum_rate,
      "total_power_w":total_power,"ee_bit_j":ee,"qos_satisfaction":satisfaction,
      "outage_fraction":1.0-satisfaction,
      "qos_shortfall_bps":float(np.mean(np.maximum(rmin-rates,0.))),
      "interference_sum":float(np.sum(interference)),"jain_fairness":fairness}

def state_vector(H,traffic,p_prev,cfg):
    gains=np.mean(np.abs(H)**2,axis=0)
    log_gain=np.log10(gains+1e-30)
    log_gain=(log_gain-log_gain.mean())/(log_gain.std()+1e-8)
    normalized_traffic=np.asarray(traffic)/(np.mean(traffic)+1e-8)
    normalized_power=np.asarray(p_prev)/(cfg.p_max_w/cfg.n_users)
    return np.concatenate([log_gain,normalized_traffic,normalized_power]).astype(np.float32)

def scalar_reward(m,p,p_prev,cfg):
    ee_scaled=m["ee_bit_j"]/1e6
    rate_scaled=m["sum_rate"]/(cfg.n_users*5e6)
    switching=np.mean(np.abs(p-p_prev))/cfg.p_max_w
    fairness_gap=1.0-m["jain_fairness"]
    return (cfg.alpha_energy*ee_scaled+cfg.beta_throughput*rate_scaled
            -cfg.qos_penalty*m["outage_fraction"]
            -cfg.fairness_penalty*fairness_gap
            -cfg.smoothness_penalty*switching)
