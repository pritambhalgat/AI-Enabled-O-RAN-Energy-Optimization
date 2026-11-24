import argparse
from config import Config
from src.traffic import train_traffic,save_traffic

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=20)
    args=parser.parse_args()
    cfg=Config()
    model,norm,mae,persistence=train_traffic(cfg.traffic_history,cfg.n_users,cfg.seed,args.epochs)
    save_traffic(model,norm)
    print(f"MLP test MAE={mae:.6f}; persistence MAE={persistence:.6f}")
    print("MLP beats persistence on this synthetic split." if mae<persistence
          else "MLP does not beat persistence on this synthetic split.")
if __name__=="__main__": main()
