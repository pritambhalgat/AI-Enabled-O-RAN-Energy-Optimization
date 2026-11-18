import os, numpy as np, torch
from torch import nn
class TrafficMLP(nn.Module):
    def __init__(self,input_dim,n_users,hidden=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.ReLU(),
            nn.Linear(hidden,64),nn.ReLU(),nn.Linear(64,n_users))
    def forward(self,x): return self.net(x)

def fit_traffic_model(history=8,n_users=8,seed=17,epochs=20):
    from torch.utils.data import TensorDataset,DataLoader
    from src.data import generate_traffic
    torch.manual_seed(seed); np.random.seed(seed)
    X,Y=generate_traffic(16000,history,n_users,seed); split=int(.8*len(X))
    xm=float(X[:split].mean()); xs=float(X[:split].std()+1e-8)
    ym=Y[:split].mean(axis=0); ys=Y[:split].std(axis=0)+1e-8
    Xn=(X-xm)/xs; Yn=(Y-ym)/ys
    model=TrafficMLP(history*n_users,n_users)
    dl=DataLoader(TensorDataset(torch.tensor(Xn[:split]),torch.tensor(Yn[:split])),
                  batch_size=128,shuffle=True)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=nn.MSELoss()
    for ep in range(epochs):
        model.train(); losses=[]
        for xb,yb in dl:
            opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
            losses.append(loss.item())
        if (ep+1)%5==0: print(f"traffic epoch {ep+1}/{epochs}, MSE={np.mean(losses):.5f}")
    model.eval()
    with torch.no_grad(): pred=model(torch.tensor(Xn[split:])).numpy()*ys+ym
    mae=float(np.mean(np.abs(pred-Y[split:])))
    persistence=float(np.mean(np.abs(X[split:].reshape(-1,history,n_users)[:,-1,:]-Y[split:])))
    norm={"xmean":xm,"xstd":xs,"ymean":ym,"ystd":ys}
    return model,norm,mae,persistence

def save_traffic(model,norm,path="artifacts"):
    os.makedirs(path,exist_ok=True)
    torch.save(model.state_dict(),os.path.join(path,"traffic_mlp.pt"))
    np.savez(os.path.join(path,"traffic_norm.npz"),**norm)

def load_traffic(history,n_users,path="artifacts"):
    model=TrafficMLP(history*n_users,n_users)
    model.load_state_dict(torch.load(os.path.join(path,"traffic_mlp.pt"),
                                     map_location="cpu",weights_only=True))
    model.eval(); norm=dict(np.load(os.path.join(path,"traffic_norm.npz")))
    return model,norm
