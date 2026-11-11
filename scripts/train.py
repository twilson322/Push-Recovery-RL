#!/usr/bin/env python3
"""Train PPO and SAC, each as a full policy and as a residual on the capture-point controller.

    python3 scripts/train.py --seeds 3            # ~25 min on 2 CPU cores
"""
import argparse
import pathlib
import pickle
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from prl.agents import train_ppo, train_sac  # noqa: E402
from prl.env import EnvConfig  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
torch.set_num_threads(2)


def log(msg):
    print(msg, flush=True)
    with open(RES / "train_log.txt", "a") as f:
        f.write(msg + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--algos", default="ppo_full,ppo_residual,sac_full,sac_residual")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    cfg = EnvConfig()  # training distribution: pushes up to 1.4 m/s, omega +/-5 %, noise on
    rows = []
    for algo in a.algos.split(","):
        for seed in range(a.seeds):
            t0 = time.time()
            kind, mode = algo.split("_")
            residual = mode == "residual"
            if kind == "ppo":
                pol, hist = train_ppo(cfg, residual, seed=seed, log=log)
            else:
                pol, hist = train_sac(cfg, residual, seed=seed, log=log)
            with open(RES / f"policy_{algo}_s{seed}.pkl", "wb") as f:
                pickle.dump(pol, f)
            for h in hist:
                rows.append({"algo": algo, "seed": seed, **h})
            pd.DataFrame(rows).to_csv(RES / "training_curves.csv", index=False)
            log(f"{algo} seed {seed} trained in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
