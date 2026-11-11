#!/usr/bin/env python3
"""Which push directions fail? Success by push direction at 0.6 m/s for the baseline and the
first PPO seed. Directions are in the stance-foot frame: +x forward, +y toward the swing leg.

    python3 scripts/direction.py
"""
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from prl.agents import Policy, policy_obs  # noqa: E402
from prl.env import N_STEPS, EnvConfig, StepEnv  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"


def run(policy, n=8000, push=0.6, seed=4242):
    env = StepEnv(n, EnvConfig(push_min=push, push_max=push + 1e-9), seed=seed)
    # the nominal touchdown velocity is (0, -0.2) m/s; what is left is the push (plus 3 cm/s noise)
    d = env.com_vel - np.array([0.0, -0.20])
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    done_any = np.zeros(n, bool)
    succ = np.zeros(n, bool)
    for _ in range(N_STEPS):
        a = policy(policy_obs(env))
        _, _, done, info = env.step(a)
        newly = done & ~done_any
        succ |= info["success"] & newly
        done_any |= done
    return ang, succ


def main():
    with open(RES / "policy_ppo_full_s0.pkl", "rb") as f:
        ppo = pickle.load(f)
    bins = np.arange(-180, 181, 45)
    names = {0: "forward", 90: "toward swing leg", 180: "backward", -90: "toward stance leg (crossover side)"}
    rows = []
    for name, pol in [("capture_point", Policy("baseline")), ("ppo_full_s0", ppo)]:
        ang, succ = run(pol)
        centers = ((ang + 180 + 22.5) // 45 * 45 - 180).astype(int)
        centers[centers == -180] = 180
        for c in sorted(set(centers)):
            m = centers == c
            rows.append({"policy": name, "direction_deg": c, "label": names.get(int(c), ""), "episodes": int(m.sum()),
                         "success": float(succ[m].mean())})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "direction.csv", index=False)
    print(df.pivot(index=["direction_deg", "label"], columns="policy", values="success").round(3))


if __name__ == "__main__":
    main()
