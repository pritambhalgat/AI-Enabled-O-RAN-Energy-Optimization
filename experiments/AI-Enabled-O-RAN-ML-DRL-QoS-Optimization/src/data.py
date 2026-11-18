import numpy as np
def generate_traffic(n_samples, history, n_users, seed=0):
    rng=np.random.default_rng(seed)
    t=np.arange(n_samples+history+1)
    phase=rng.uniform(0,2*np.pi,n_users); base=rng.uniform(.35,.75,n_users)
    state=base.copy(); series=np.zeros((len(t),n_users),np.float32)
    for i in range(len(t)):
        seasonal=.18*np.sin(2*np.pi*i/48+phase)
        state=np.clip(.88*state+.12*(base+seasonal)+rng.normal(0,.035,n_users),.05,1.)
        series[i]=state
    X=[]; Y=[]
    for i in range(n_samples):
        X.append(series[i:i+history].reshape(-1)); Y.append(series[i+history])
    return np.asarray(X,np.float32),np.asarray(Y,np.float32)
