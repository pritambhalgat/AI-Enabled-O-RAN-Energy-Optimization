from config import Config
from src.model import fit_traffic_model,save_traffic
def main():
    c=Config(); m,n,mae,base=fit_traffic_model(c.traffic_history,c.n_users,c.seed)
    save_traffic(m,n); print(f"test MLP MAE={mae:.6f}; persistence MAE={base:.6f}")
    print("MLP beats persistence on this split." if mae<base else "No predictor gain: do not claim improvement.")
if __name__=="__main__": main()
