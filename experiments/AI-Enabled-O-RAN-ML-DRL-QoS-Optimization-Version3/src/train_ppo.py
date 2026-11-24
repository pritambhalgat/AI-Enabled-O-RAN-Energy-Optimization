import argparse, os
import numpy as np
import torch
from config import Config
from src.traffic import generate_traffic,load_traffic,predict_traffic
from src.radio import generate_channel,state_vector,metrics,scalar_reward,RECEIVERS
from src.ppo import ActorCritic,decode_action,generalized_advantage

def train_for_receiver(receiver,cfg,episodes=None,steps=None,seed=None):
    receiver=receiver.upper()
    if receiver not in RECEIVERS: raise ValueError(f"receiver must be one of {RECEIVERS}")
    episodes=int(episodes or cfg.ppo_episodes)
    steps=int(steps or cfg.steps_per_episode)
    seed=int(cfg.seed if seed is None else seed)
    np.random.seed(seed); torch.manual_seed(seed)
    predictor,norm=load_traffic(cfg.traffic_history,cfg.n_users)
    traffic_X,_=generate_traffic(episodes*steps,cfg.traffic_history,cfg.n_users,seed+101)
    actor=ActorCritic(3*cfg.n_users,cfg.n_users+1,cfg.hidden_dim)
    optimizer=torch.optim.Adam(actor.parameters(),lr=cfg.learning_rate)
    rng=np.random.default_rng(seed)
    idx=0
    for episode in range(episodes):
        states=[]; actions=[]; old_logp=[]; values=[]; rewards=[]; dones=[]
        p_previous=np.ones(cfg.n_users)*cfg.p_max_w/cfg.n_users
        for _ in range(steps):
            traffic=predict_traffic(predictor,norm,traffic_X[idx].reshape(cfg.traffic_history,cfg.n_users))
            H=generate_channel(cfg.n_antennas,cfg.n_users,rng,cfg.pathloss_exponent,cfg.shadow_std_db)
            rmin=cfg.min_rate_base_bps+(cfg.min_rate_max_bps-cfg.min_rate_base_bps)*traffic
            state=state_vector(H,traffic,p_previous,cfg)
            s=torch.tensor(state,dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                raw,logp,value=actor.sample(s)
                p=decode_action(raw,cfg.n_users,cfg.p_max_w).squeeze(0).numpy()
            m=metrics(H,p,cfg,rmin,receiver)
            rew=scalar_reward(m,p,p_previous,cfg)
            states.append(state); actions.append(raw.squeeze(0).numpy())
            old_logp.append(float(logp.item())); values.append(float(value.item()))
            rewards.append(float(rew)); dones.append(0.0)
            p_previous=p; idx+=1
        dones[-1]=1.0
        S=torch.tensor(np.asarray(states),dtype=torch.float32)
        A=torch.tensor(np.asarray(actions),dtype=torch.float32)
        oldlp=torch.tensor(old_logp,dtype=torch.float32)
        V=torch.tensor(values,dtype=torch.float32)
        R=torch.tensor(rewards,dtype=torch.float32)
        D=torch.tensor(dones,dtype=torch.float32)
        advantage,returns=generalized_advantage(R,V,D,cfg.gamma,cfg.gae_lambda)
        advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
        for _ in range(cfg.ppo_epochs):
            order=np.random.permutation(len(S))
            for start in range(0,len(S),cfg.minibatch_size):
                ix=order[start:start+cfg.minibatch_size]
                logp,entropy,value=actor.evaluate(S[ix],A[ix])
                ratio=torch.exp(logp-oldlp[ix])
                obj1=ratio*advantage[ix]
                obj2=torch.clamp(ratio,1-cfg.clip_epsilon,1+cfg.clip_epsilon)*advantage[ix]
                actor_loss=-torch.min(obj1,obj2).mean()
                critic_loss=.5*((value-returns[ix])**2).mean()
                loss=actor_loss+cfg.value_coef*critic_loss-cfg.entropy_coef*entropy.mean()
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(),.5)
                optimizer.step()
        if (episode+1)%25==0:
            print(f"{receiver}: episode {episode+1}/{episodes}, mean reward={np.mean(rewards):.4f}")
    os.makedirs("artifacts",exist_ok=True)
    path=f"artifacts/ppo_{receiver}.pt"
    torch.save(actor.state_dict(),path)
    return actor,path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--receiver",choices=RECEIVERS,required=True)
    parser.add_argument("--episodes",type=int,default=None)
    parser.add_argument("--steps",type=int,default=None)
    parser.add_argument("--seed",type=int,default=None)
    args=parser.parse_args()
    _,path=train_for_receiver(args.receiver,Config(),args.episodes,args.steps,args.seed)
    print("Saved",path)
if __name__=="__main__": main()
