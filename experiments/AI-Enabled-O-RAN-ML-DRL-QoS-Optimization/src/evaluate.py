import argparse, os, numpy as np, pandas as pd, torch
from scipy.optimize import minimize
from config import Config
from src.data import generate_traffic
from src.model import load_traffic
from src.radio import channel_realization,make_state,mrc_metrics,reward,project_power
from src.ppo import ActorCritic,decode_action
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--episodes",type=int,default=5)
    ap.add_argument("--blocks",type=int,default=100); ap.add_argument("--seed",type=int,default=29)
    a=ap.parse_args(); c=Config(); rng=np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    os.makedirs("results",exist_ok=True)
    pred,norm=load_traffic(c.traffic_history,c.n_users)
    agent=ActorCritic(3*c.n_users,c.n_users+1,c.hidden_dim)
    agent.load_state_dict(torch.load("artifacts/ppo_actor_critic.pt",map_location="cpu",weights_only=True)); agent.eval()
    X,_=generate_traffic(a.episodes*a.blocks,c.traffic_history,c.n_users,a.seed+101)
    xm=float(norm["xmean"]); xs=float(norm["xstd"]); ym=np.asarray(norm["ymean"]); ys=np.asarray(norm["ystd"])
    rows=[]; idx=0
    for ep in range(a.episodes):
        prev=np.ones(c.n_users)*c.p_max_w/c.n_users
        for b in range(a.blocks):
            with torch.no_grad():
                tn=pred(torch.tensor((X[idx:idx+1]-xm)/xs,dtype=torch.float32)).numpy()[0]
                H=channel_realization(c.n_antennas,c.n_users,rng)
                traffic=np.clip(tn*ys+ym,.05,1.)
                rmin=c.min_rate_base_bps+(c.min_rate_max_bps-c.min_rate_base_bps)*traffic
                state=make_state(H,traffic,prev,c)
                raw=agent.mu(torch.tensor(state).unsqueeze(0)) # deterministic PPO action
                pppo=decode_action(raw,c.n_users,c.p_max_w).numpy().ravel()
            eq=np.ones(c.n_users)*c.p_max_w/c.n_users
            gains=np.sum(np.abs(H)**2,axis=0)
            heur=project_power(.05/c.n_users+.95*gains/(gains.sum()+1e-30),c.p_max_w,c.p_max_w)
            def objective(p): return -reward(mrc_metrics(H,p,c,rmin),p,prev,c)
            opt=minimize(objective,eq,bounds=[(0,c.p_max_w)]*c.n_users,
              constraints=[{"type":"ineq","fun":lambda p:c.p_max_w-np.sum(p)}],
              method="SLSQP",options={"maxiter":100,"ftol":1e-8})
            ps=np.maximum(opt.x,0.) if opt.success else eq
            for name,p in {"equal":eq,"channel_heuristic":heur,"SLSQP":ps,"PPO_ML":pppo}.items():
                m=mrc_metrics(H,p,c,rmin)
                rows.append({"episode":ep,"block":b,"method":name,"sum_rate_mbps":m["sum_rate"]/1e6,
                  "energy_efficiency_mbps_per_w":m["ee"]/1e6,"qos_satisfaction":m["qos_satisfaction"],
                  "qos_outage_fraction":m["outage"],"mean_qos_shortfall_mbps":m["qos_shortfall"]/1e6,
                  "total_power_w":m["total_power"],"interference_ratio":m["interference_ratio"],
                  "jain_fairness":m["fairness"],"power_budget_used":float(np.sum(p)/c.p_max_w)})
            prev=pppo; idx+=1
    df=pd.DataFrame(rows); df.to_csv("results/evaluation.csv",index=False)
    cols=["sum_rate_mbps","energy_efficiency_mbps_per_w","qos_satisfaction","qos_outage_fraction",
      "mean_qos_shortfall_mbps","total_power_w","interference_ratio","jain_fairness","power_budget_used"]
    df.groupby("method")[cols].agg(["mean","std"]).to_csv("results/summary.csv")
    print(df.groupby("method")[cols].mean().to_string())
if __name__=="__main__": main()
