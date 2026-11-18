import os, numpy as np, torch
from config import Config
from src.data import generate_traffic
from src.model import load_traffic
from src.radio import channel_realization,make_state,mrc_metrics,reward
from src.ppo import ActorCritic,decode_action,compute_gae
def main():
    c=Config(); np.random.seed(c.seed); torch.manual_seed(c.seed)
    os.makedirs("artifacts",exist_ok=True)
    predictor,norm=load_traffic(c.traffic_history,c.n_users)
    X,_=generate_traffic(c.ppo_episodes*c.steps_per_episode,c.traffic_history,c.n_users,c.seed+101)
    xm=float(norm["xmean"]); xs=float(norm["xstd"]); ym=np.asarray(norm["ymean"]); ys=np.asarray(norm["ystd"])
    rng=np.random.default_rng(c.seed); agent=ActorCritic(3*c.n_users,c.n_users+1,c.hidden_dim)
    opt=torch.optim.Adam(agent.parameters(),lr=c.learning_rate); idx=0
    for ep in range(c.ppo_episodes):
        S=[]; A=[]; LP=[]; V=[]; RR=[]; DD=[]
        prev=np.ones(c.n_users)*c.p_max_w/c.n_users
        for _ in range(c.steps_per_episode):
            with torch.no_grad():
                tn=predictor(torch.tensor((X[idx:idx+1]-xm)/xs,dtype=torch.float32)).numpy()[0]
            traffic=np.clip(tn*ys+ym,.05,1.)
            H=channel_realization(c.n_antennas,c.n_users,rng)
            rmin=c.min_rate_base_bps+(c.min_rate_max_bps-c.min_rate_base_bps)*traffic
            state=make_state(H,traffic,prev,c); st=torch.tensor(state).unsqueeze(0)
            with torch.no_grad():
                raw,lp,val=agent.act(st); p=decode_action(raw,c.n_users,c.p_max_w).squeeze(0).numpy()
            met=mrc_metrics(H,p,c,rmin); rew=reward(met,p,prev,c)
            S.append(state); A.append(raw.squeeze(0).numpy()); LP.append(lp.item()); V.append(val.item())
            RR.append(rew); DD.append(0.); prev=p; idx+=1
        S=torch.tensor(np.asarray(S),dtype=torch.float32); A=torch.tensor(np.asarray(A),dtype=torch.float32)
        oldlp=torch.tensor(LP,dtype=torch.float32); values=torch.tensor(V,dtype=torch.float32)
        R=torch.tensor(RR,dtype=torch.float32); D=torch.tensor(DD,dtype=torch.float32)
        adv,ret=compute_gae(R,values,D,c.gamma,c.gae_lambda); adv=(adv-adv.mean())/(adv.std()+1e-8)
        for _ in range(c.ppo_epochs):
            order=np.random.permutation(len(S))
            for start in range(0,len(S),c.minibatch_size):
                ix=order[start:start+c.minibatch_size]
                lp,ent,v=agent.evaluate(S[ix],A[ix]); ratio=torch.exp(lp-oldlp[ix])
                actor=-torch.min(ratio*adv[ix],torch.clamp(ratio,1-c.clip_epsilon,1+c.clip_epsilon)*adv[ix]).mean()
                critic=.5*((v-ret[ix])**2).mean()
                loss=actor+c.value_coef*critic-c.entropy_coef*ent.mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(agent.parameters(),.5); opt.step()
        if (ep+1)%25==0: print(f"episode {ep+1}/{c.ppo_episodes}, reward={np.mean(RR):.4f}")
    torch.save(agent.state_dict(),"artifacts/ppo_actor_critic.pt")
    print("Saved artifacts/ppo_actor_critic.pt")
if __name__=="__main__": main()
