import numpy as np
from scipy.optimize import minimize
from src.radio import metrics,scalar_reward

def project_to_qos(H,p_proposed,cfg,rmin,receiver,maxiter=150):
    p0=np.clip(np.asarray(p_proposed,dtype=float),0.0,cfg.p_max_w)
    bounds=[(0.0,cfg.p_max_w)]*cfg.n_users
    constraints=[
        {"type":"ineq","fun":lambda p:cfg.p_max_w-float(np.sum(p))},
        {"type":"ineq","fun":lambda p:metrics(H,p,cfg,rmin,receiver)["rates"]-rmin}
    ]
    objective=lambda p:float(np.sum((p-p0)**2))
    result=minimize(objective,p0,method="SLSQP",bounds=bounds,constraints=constraints,
                    options={"maxiter":maxiter,"ftol":1e-9})
    candidate=np.clip(result.x,0.0,cfg.p_max_w)
    achieved=metrics(H,candidate,cfg,rmin,receiver)
    feasible=(bool(result.success) and np.sum(candidate)<=cfg.p_max_w+1e-6
              and np.all(achieved["rates"]>=rmin-1e-3))
    if feasible:
        return candidate,True,str(result.message)
    return p0,False,str(result.message)

def reward_optimize(H,p_initial,p_previous,cfg,rmin,receiver,maxiter=120):
    p0=np.asarray(p_initial,dtype=float)
    def objective(p):
        return -scalar_reward(metrics(H,p,cfg,rmin,receiver),p,p_previous,cfg)
    constraints=[{"type":"ineq","fun":lambda p:cfg.p_max_w-float(np.sum(p))}]
    result=minimize(objective,np.clip(p0,0,cfg.p_max_w),method="SLSQP",
                    bounds=[(0.0,cfg.p_max_w)]*cfg.n_users,constraints=constraints,
                    options={"maxiter":maxiter,"ftol":1e-8})
    if result.success:
        return np.clip(result.x,0,cfg.p_max_w),True
    return np.clip(p0,0,cfg.p_max_w),False
