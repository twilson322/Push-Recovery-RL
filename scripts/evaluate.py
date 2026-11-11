#!/usr/bin/env python3
"""Evaluate every trained policy and the capture-point baseline.

  * success rate vs push magnitude (0 to 2 m/s) on the training dynamics distribution
  * out-of-distribution: omega +/-10 % (twice the training range) and 3 cm foot noise
  * what the policies learned: step time and ankle use vs push magnitude
"""
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from prl.agents import Policy, evaluate  # noqa: E402
from prl.env import EnvConfig  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
PUSHES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def load_policies():
    pols = {"capture_point": [Policy("baseline")]}
    for f in sorted(RES.glob("policy_*_s*.pkl")):
        algo = f.stem.split("policy_")[1].rsplit("_s", 1)[0]
        with open(f, "rb") as fh:
            pols.setdefault(algo, []).append(pickle.load(fh))
    return pols


def main():
    pols = load_policies()
    rows, beh = [], []
    for name, plist in pols.items():
        for si, pol in enumerate(plist):
            for m in PUSHES:
                for dist, cfg in [("train_dist", EnvConfig(push_min=m, push_max=m + 1e-9)),
                                  ("ood_omega10_foot3cm", EnvConfig(push_min=m, push_max=m + 1e-9,
                                                                    omega_scale=(0.90, 1.10), foot_noise=0.03))]:
                    r = evaluate(pol, cfg, n=2000, seed=777)
                    rows.append({"policy": name, "seed": si, "push": m, "dist": dist, "success": r["success"], "fall": r["fall"]})
                    if dist == "train_dist":
                        beh.append({"policy": name, "seed": si, "push": m, "T_first_step": r["T_mean"][0],
                                    "T_second_step": r["T_mean"][1], "cop_first_step": r["cop_mean"][0]})
            print(f"evaluated {name} seed {si}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "eval.csv", index=False)
    pd.DataFrame(beh).to_csv(RES / "behavior.csv", index=False)
    # Area under the success-vs-push curve over the training range, as one number per policy.
    tr = df[(df.dist == "train_dist") & (df.push <= 1.4)]
    auc = tr.groupby(["policy", "seed"]).success.mean().groupby("policy").agg(["mean", "std", "count"])
    print(auc)


if __name__ == "__main__":
    main()
