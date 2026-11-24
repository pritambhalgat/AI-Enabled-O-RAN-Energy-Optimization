import argparse, os
import numpy as np
import pandas as pd
import torch
from scipy.stats import t
from config import Config
from src.traffic import generate_traffic,load_traffic,predict_traffic
from src.radio import generate_channel,state_vector,metrics,RECEIVERS
from src.ppo import ActorCritic,decode_action
from src.optimization import project_to_qos,reward_optimize

def confidence_summary(df):
    metrics_to_report=["sum_rate_mbps","ee_mbps_per_w","qos_satisfaction","outage_fraction",
                       "total_power_w","jain_fairness","mean_qos_shortfall_mbps"]
    rows=[]
    for (receiver,method),group in df.groupby(["receiver","method"]):
        row={"receiver":receiver,"method":method}
        for metric in metrics_to_report:
            values=group.groupby("seed")[metric].mean().to_numpy()
            n=len(values); mean=float(values.mean())
            std=float(values.std(ddof=1)) if n>1 else float("nan")
            half=float(t.ppf(.975,n-1)*std/np.sqrt(n)) if n>1 else float("nan")
            row[metric+"_mean"]=mean; row[metric+"_std_seed"]=std
            row[metric+"_ci95_low"]=mean-half if n>1 else float("nan")
            row[metric+"_ci95_high"]=mean+half if n>1 else float("nan")
        row["n_seeds"]=len(group["seed"].unique())
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,default=5)
    parser.add_argument("--blocks",type=int,default=50)
    parser.add_argument("--seed-start",type=int,default=4100)
    parser.add_argument("--output",default="results")
    args=parser.parse_args()
    if args.seeds<2 or args.blocks<1:
        raise ValueError("Use at least two independent seeds and one block.")
    cfg=Config()
    predictor,norm=load_traffic(cfg.traffic_history,cfg.n_users)
    agents={}
    for receiver in RECEIVERS:
        agent=ActorCritic(3*cfg.n_users,cfg.n_users+1,cfg.hidden_dim)
        agent.load_state_dict(torch.load(f"artifacts/ppo_{receiver}.pt",map_location="cpu",weights_only=True))
        agent.eval(); agents[receiver]=agent
    os.makedirs(args.output,exist_ok=True)
    rows=[]
    for seed in range(args.seed_start,args.seed_start+args.seeds):
        rng=np.random.default_rng(seed)
        X,_=generate_traffic(args.blocks,cfg.traffic_history,cfg.n_users,seed+23)
        previous={r:np.ones(cfg.n_users)*cfg.p_max_w/cfg.n_users for r in RECEIVERS}
        for block in range(args.blocks):
            traffic=predict_traffic(predictor,norm,X[block].reshape(cfg.traffic_history,cfg.n_users))
            H=generate_channel(cfg.n_antennas,cfg.n_users,rng,cfg.pathloss_exponent,cfg.shadow_std_db)
            rmin=cfg.min_rate_base_bps+(cfg.min_rate_max_bps-cfg.min_rate_base_bps)*traffic
            equal=np.ones(cfg.n_users)*cfg.p_max_w/cfg.n_users
            gains=np.sum(np.abs(H)**2,axis=0)
            heuristic=.05/cfg.n_users+.95*gains/(gains.sum()+1e-30)
            heuristic=heuristic/heuristic.sum()*cfg.p_max_w
            for receiver in RECEIVERS:
                pprev=previous[receiver]
                state=state_vector(H,traffic,pprev,cfg)
                with torch.no_grad():
                    mu,_,_=agents[receiver](torch.tensor(state,dtype=torch.float32).unsqueeze(0))
                    p_raw=decode_action(mu,cfg.n_users,cfg.p_max_w).numpy().ravel()
                p_projected,projection_ok,_=project_to_qos(H,p_raw,cfg,rmin,receiver)
                p_slsqp,slsqp_ok=reward_optimize(H,equal,pprev,cfg,rmin,receiver)
                methods={"equal":(equal,True),"channel_heuristic":(heuristic,True),
                         "SLSQP_reward":(p_slsqp,slsqp_ok),"PPO_raw":(p_raw,True),
                         "PPO_QoS_projected":(p_projected,projection_ok)}
                for method,(power,solver_ok) in methods.items():
                    m=metrics(H,power,cfg,rmin,receiver)
                    rows.append({"seed":seed,"block":block,"receiver":receiver,"method":method,
                        "sum_rate_mbps":m["sum_rate"]/1e6,
                        "ee_mbps_per_w":m["ee_bit_j"]/1e6,
                        "qos_satisfaction":m["qos_satisfaction"],
                        "outage_fraction":m["outage_fraction"],
                        "total_power_w":m["total_power_w"],
                        "jain_fairness":m["jain_fairness"],
                        "mean_qos_shortfall_mbps":m["qos_shortfall_bps"]/1e6,
                        "power_budget_used":float(np.sum(power)/cfg.p_max_w),
                        "solver_success":bool(solver_ok)})
                previous[receiver]=p_raw
        print(f"Completed evaluation seed {seed}")
    df=pd.DataFrame(rows)
    summary=confidence_summary(df)
    df.to_csv(os.path.join(args.output,"per_seed.csv"),index=False)
    summary.to_csv(os.path.join(args.output,"summary_ci95.csv"),index=False)
    print(summary.to_string(index=False))
    projected=df[df["method"]=="PPO_QoS_projected"]
    print("QoS projection success fraction:",float(projected["solver_success"].mean()))
if __name__=="__main__": main()
