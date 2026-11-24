import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def generate_traffic(n_samples, history, n_users, seed=0):
    rng=np.random.default_rng(seed)
    t=np.arange(n_samples+history+1)
    phase=rng.uniform(0,2*np.pi,n_users)
    base=rng.uniform(.35,.75,n_users)
    state=base.copy()
    series=np.zeros((len(t),n_users),dtype=np.float32)
    for i in range(len(t)):
        seasonal=.18*np.sin(2*np.pi*i/48+phase)
        state=np.clip(.88*state+.12*(base+seasonal)+rng.normal(0,.035,n_users),.05,1.0)
        series[i]=state
    X=[]; Y=[]
    for i in range(n_samples):
        X.append(series[i:i+history].reshape(-1))
        Y.append(series[i+history])
    return np.asarray(X,np.float32),np.asarray(Y,np.float32)

class TrafficMLP(nn.Module):
    def __init__(self,input_dim,n_users,hidden=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.ReLU(),
                               nn.Linear(hidden,64),nn.ReLU(),nn.Linear(64,n_users))
    def forward(self,x): return self.net(x)

def train_traffic(history=8,n_users=8,seed=17,epochs=20):
    torch.manual_seed(seed); np.random.seed(seed)
    X,Y=generate_traffic(16000,history,n_users,seed)
    split=int(.8*len(X))
    xm=float(X[:split].mean()); xs=float(X[:split].std()+1e-8)
    ym=Y[:split].mean(axis=0); ys=Y[:split].std(axis=0)+1e-8
    Xn=(X-xm)/xs; Yn=(Y-ym)/ys
    model=TrafficMLP(history*n_users,n_users)
    loader=DataLoader(TensorDataset(torch.tensor(Xn[:split]),torch.tensor(Yn[:split])),
                      batch_size=128,shuffle=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    criterion=nn.MSELoss()
    for epoch in range(epochs):
        losses=[]; model.train()
        for xb,yb in loader:
            optimizer.zero_grad()
            loss=criterion(model(xb),yb)
            loss.backward(); optimizer.step()
            losses.append(float(loss.item()))
        if (epoch+1)%5==0:
            print(f"Traffic MLP epoch {epoch+1}/{epochs}: MSE={np.mean(losses):.6f}")
    model.eval()
    with torch.no_grad():
        prediction=model(torch.tensor(Xn[split:],dtype=torch.float32)).numpy()*ys+ym
    mae=float(np.mean(np.abs(prediction-Y[split:])))
    persistence=float(np.mean(np.abs(X[split:].reshape(-1,history,n_users)[:,-1,:]-Y[split:])))
    norm={"xmean":xm,"xstd":xs,"ymean":ym,"ystd":ys}
    return model,norm,mae,persistence

def save_traffic(model,norm,out_dir="artifacts"):
    import os
    os.makedirs(out_dir,exist_ok=True)
    torch.save(model.state_dict(),os.path.join(out_dir,"traffic_mlp.pt"))
    np.savez(os.path.join(out_dir,"traffic_norm.npz"),**norm)

def load_traffic(history,n_users,out_dir="artifacts"):
    import os
    model=TrafficMLP(history*n_users,n_users)
    model.load_state_dict(torch.load(os.path.join(out_dir,"traffic_mlp.pt"),
                                     map_location="cpu",weights_only=True))
    model.eval()
    return model,dict(np.load(os.path.join(out_dir,"traffic_norm.npz")))

def predict_traffic(model,norm,history_window):
    xm=float(norm["xmean"]); xs=float(norm["xstd"])
    ym=np.asarray(norm["ymean"]); ys=np.asarray(norm["ystd"])
    x=np.asarray(history_window,dtype=np.float32).reshape(1,-1)
    with torch.no_grad():
        prediction=model(torch.tensor((x-xm)/xs,dtype=torch.float32)).numpy()[0]
    return np.clip(prediction*ys+ym,.05,1.0)
