import argparse, subprocess, sys
parser=argparse.ArgumentParser()
parser.add_argument("--seeds",type=int,default=5)
parser.add_argument("--blocks",type=int,default=50)
args=parser.parse_args()
subprocess.run([sys.executable,"-m","src.train_traffic"],check=True)
for receiver in ("MRC","ZF","MMSE"):
    subprocess.run([sys.executable,"-m","src.train_ppo","--receiver",receiver],check=True)
subprocess.run([sys.executable,"-m","src.evaluate","--seeds",str(args.seeds),
                "--blocks",str(args.blocks)],check=True)
