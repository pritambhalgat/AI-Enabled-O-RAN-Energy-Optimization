import numpy as np
def channel_realization(M,K,rng):
    # Illustrative macro path-loss / shadowing and i.i.d. Rayleigh fading.
    distance=rng.uniform(35.,250.,K); d_km=distance/1000.
    pl_db=128.1+37.6*np.log10(d_km) # illustrative 2-GHz model
    shadow=rng.normal(0.,6.,K); beta=10**(-(pl_db+shadow)/10.)
    g=(rng.normal(size=(M,K))+1j*rng.normal(size=(M,K)))/np.sqrt(2)
    return g*np.sqrt(beta)[None,:]
def project_power(weights,total,pmax):
    z=np.maximum(np.asarray(weights,float),1e-12)
    total=float(np.clip(total,0.,pmax))
    return total*z/z.sum()
def mrc_metrics(H,p,cfg,rmin):
    M,K=H.shape; desired=np.zeros(K); interf=np.zeros(K); noise=np.zeros(K)
    for k in range(K):
        hk=H[:,k]; norm=np.vdot(hk,hk).real
        desired[k]=p[k]*norm**2
        interf[k]=sum(p[j]*abs(np.vdot(hk,H[:,j]))**2 for j in range(K) if j!=k)
        noise[k]=cfg.noise_power_w*norm
    sinr=desired/np.maximum(interf+noise,1e-30)
    rates=cfg.bandwidth_hz*np.log2(1+sinr)
    total=cfg.fixed_power_w+float(np.sum(p))/cfg.pa_efficiency
    sr=float(rates.sum()); ee=sr/max(total,1e-12)
    sat=float(np.mean(rates>=rmin)); outage=1.-sat
    jain=float(sr**2/(K*np.sum(rates**2)+1e-30))
    return {"sinr":sinr,"rates":rates,"sum_rate":sr,"total_power":total,"ee":ee,
      "qos_satisfaction":sat,"outage":outage,
      "qos_shortfall":float(np.mean(np.maximum(rmin-rates,0.))),
      "interference_ratio":float(interf.sum()/max(desired.sum(),1e-30)),
      "fairness":jain}
def make_state(H,traffic,p_prev,cfg):
    gain=np.mean(np.abs(H)**2,axis=0); lg=np.log10(gain+1e-30)
    lg=(lg-lg.mean())/(lg.std()+1e-8)
    tr=np.asarray(traffic); tr=tr/(tr.mean()+1e-8)
    pp=p_prev/(cfg.p_max_w/cfg.n_users)
    return np.concatenate([lg,tr,pp]).astype(np.float32)
def reward(m,p,prev,cfg):
    e=m["ee"]/1e6; r=m["sum_rate"]/(cfg.n_users*5e6)
    change=np.mean(np.abs(p-prev))/cfg.p_max_w
    return cfg.alpha_energy*e+cfg.beta_throughput*r-cfg.qos_penalty*m["outage"]-cfg.smoothness_penalty*change
