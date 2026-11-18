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
        z=self.body(s); mu=self.mu(z)
        return mu,self.log_std.expand_as(mu),self.value(z).squeeze(-1)
    def act(self,s):
        mu,ls,v=self(s); dist=Normal(mu,ls.exp()); a=dist.sample()
        return a,dist.log_prob(a).sum(-1),v
    def evaluate(self,s,a):
        mu,ls,v=self(s); d=Normal(mu,ls.exp())
        return d.log_prob(a).sum(-1),d.entropy().sum(-1),v
def decode_action(raw,K,pmax):
    shares=torch.softmax(raw[...,:K],dim=-1)
    frac=torch.sigmoid(raw[...,K:K+1])
    return shares*frac*pmax
def compute_gae(rewards,values,dones,gamma,lam):
    adv=torch.zeros_like(rewards); last=0.
    for t in reversed(range(len(rewards))):
        nv=0. if t==len(rewards)-1 or dones[t] else values[t+1]
        nt=0. if dones[t] else 1.
        delta=rewards[t]+gamma*nv-values[t]
        last=delta+gamma*lam*nt*last; adv[t]=last
    return adv,adv+values
