import torch
from torch import nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self,state_dim,action_dim,hidden=128):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(state_dim,hidden),nn.Tanh(),
                                nn.Linear(hidden,hidden),nn.Tanh())
        self.mu=nn.Linear(hidden,action_dim)
        self.log_std=nn.Parameter(torch.full((action_dim,),-.6))
        self.value=nn.Linear(hidden,1)
    def forward(self,s):
        z=self.body(s)
        return self.mu(z),self.log_std.expand(s.shape[0],-1),self.value(z).squeeze(-1)
    def sample(self,s):
        mu,log_std,value=self(s)
        dist=Normal(mu,log_std.exp())
        action=dist.sample()
        return action,dist.log_prob(action).sum(-1),value
    def evaluate(self,s,action):
        mu,log_std,value=self(s)
        dist=Normal(mu,log_std.exp())
        return dist.log_prob(action).sum(-1),dist.entropy().sum(-1),value

def decode_action(raw,n_users,pmax):
    shares=torch.softmax(raw[...,:n_users],dim=-1)
    fraction=torch.sigmoid(raw[...,n_users:n_users+1])
    return shares*fraction*pmax

def generalized_advantage(rewards,values,dones,gamma,lam):
    advantages=torch.zeros_like(rewards)
    last=0.0
    for t in reversed(range(len(rewards))):
        next_value=0.0 if (t==len(rewards)-1 or dones[t]) else values[t+1]
        alive=0.0 if dones[t] else 1.0
        delta=rewards[t]+gamma*next_value-values[t]
        last=delta+gamma*lam*alive*last
        advantages[t]=last
    return advantages,advantages+values
