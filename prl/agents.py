"""PPO and SAC written from scratch in PyTorch, plus the residual wrapper.

Residual RL (Johannink et al. 2019; Silver et al. 2018): the executed action is
    a = clip(a_base(s) + scale * a_policy(s), -1, 1)
where a_base is the capture-point controller. The policy only has to learn a correction, it
starts out as the baseline, and exploration happens around a controller that already works.
"""
import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .env import N_STEPS, StepEnv, capture_point_action, to_physical


def mlp(i, o, h=64, act=nn.Tanh, out_gain=1.0):
    net = nn.Sequential(nn.Linear(i, h), act(), nn.Linear(h, h), act(), nn.Linear(h, o))
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(net[-1].weight, out_gain)
    return net


class RunningNorm:
    """Welford running mean/variance for observation normalization."""

    def __init__(self, dim):
        self.mean, self.var, self.count = np.zeros(dim), np.ones(dim), 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d**2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)


def policy_obs(env):
    """Observation = state + phase (step index), which makes the 8-step horizon Markov."""
    return np.column_stack([env.obs(), env.step_index / N_STEPS]).astype(np.float32)


class Policy:
    """Deterministic evaluation wrapper shared by PPO, SAC and the baseline."""

    def __init__(self, kind, net=None, norm=None, residual=False, scale=0.3):
        self.kind, self.net, self.norm, self.residual, self.scale = kind, net, norm, residual, scale

    @torch.no_grad()
    def __call__(self, obs_full):
        base = capture_point_action(obs_full[:, :4])
        if self.kind == "baseline":
            return base
        z = torch.as_tensor(self.norm(obs_full))
        a = self.net(z)
        if self.kind == "sac":
            a = torch.tanh(a[:, : a.shape[1] // 2])
        a = a.numpy()
        return np.clip(base + self.scale * a, -1, 1) if self.residual else np.clip(a, -1, 1)


# ============================================================================ PPO
class ActorCritic(nn.Module):
    def __init__(self, od, ad, residual):
        super().__init__()
        self.pi = mlp(od, ad, out_gain=0.01)
        self.v = mlp(od, 1)
        self.log_std = nn.Parameter(torch.full((ad,), -1.0 if residual else -0.5))

    def dist(self, z):
        return torch.distributions.Normal(self.pi(z), torch.exp(self.log_std))


@dataclass
class PPOConfig:
    iters: int = 250
    n_envs: int = 1024
    horizon: int = 16             # env steps per environment per iteration (two episodes)
    lr: float = 3e-4              # decays linearly to zero
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 8
    minibatches: int = 8
    value_coef: float = 0.5
    entropy_coef: float = 1e-3
    max_grad_norm: float = 0.5


def train_ppo(cfg, residual, hp: PPOConfig | None = None, seed=0, scale=0.3, log=print):
    hp = hp if hp is not None else PPOConfig()
    torch.manual_seed(seed)
    iters, n_envs, horizon = hp.iters, hp.n_envs, hp.horizon
    env = StepEnv(n_envs, cfg, seed=seed)
    od, ad = env.obs_dim + 1, env.act_dim
    ac = ActorCritic(od, ad, residual)
    opt = torch.optim.Adam(ac.parameters(), lr=hp.lr, eps=1e-5)
    norm = RunningNorm(od)
    gamma, lam, clip = hp.gamma, hp.gae_lambda, hp.clip
    hist = []
    t0 = time.time()
    for it in range(iters):
        for g in opt.param_groups:
            g["lr"] = hp.lr * (1 - it / iters)
        O = np.zeros((horizon, n_envs, od), np.float32)
        A = np.zeros((horizon, n_envs, ad), np.float32)
        LP = np.zeros((horizon, n_envs), np.float32)
        R = np.zeros((horizon, n_envs), np.float32)
        D = np.zeros((horizon, n_envs), np.float32)
        V = np.zeros((horizon + 1, n_envs), np.float32)
        succ, fell, eps = 0, 0, 0
        for t in range(horizon):
            raw = policy_obs(env)
            norm.update(raw)
            z = torch.as_tensor(norm(raw))
            with torch.no_grad():
                d = ac.dist(z)
                a = d.sample()
                V[t] = ac.v(z).squeeze(-1).numpy()
                LP[t] = d.log_prob(a).sum(-1).numpy()
            a = a.numpy()
            exec_a = np.clip(capture_point_action(raw[:, :4]) + scale * a, -1, 1) if residual else np.clip(a, -1, 1)
            _, r, done, info = env.step(exec_a)
            O[t], A[t], R[t], D[t] = z.numpy(), a, r, done
            succ += info["success"].sum()
            fell += (info["fell"] & done).sum()
            eps += done.sum()
        with torch.no_grad():
            V[horizon] = ac.v(torch.as_tensor(norm(policy_obs(env)))).squeeze(-1).numpy()
        # GAE
        adv = np.zeros_like(R)
        last = 0.0
        for t in reversed(range(horizon)):
            nonterm = 1.0 - D[t]
            delta = R[t] + gamma * V[t + 1] * nonterm - V[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        ret = adv + V[:-1]
        b_o, b_a = torch.as_tensor(O.reshape(-1, od)), torch.as_tensor(A.reshape(-1, ad))
        b_lp, b_adv, b_ret = (torch.as_tensor(x.reshape(-1)) for x in (LP, adv, ret))
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        N = b_o.shape[0]
        for _ in range(hp.epochs):
            perm = torch.randperm(N)
            for mb in perm.chunk(hp.minibatches):
                d = ac.dist(b_o[mb])
                ratio = torch.exp(d.log_prob(b_a[mb]).sum(-1) - b_lp[mb])
                pg = -torch.min(ratio * b_adv[mb], torch.clamp(ratio, 1 - clip, 1 + clip) * b_adv[mb]).mean()
                vl = 0.5 * ((ac.v(b_o[mb]).squeeze(-1) - b_ret[mb]) ** 2).mean()
                loss = pg + hp.value_coef * vl - hp.entropy_coef * d.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), hp.max_grad_norm)
                opt.step()
        rate = succ / max(1, eps)
        hist.append({"iter": it, "env_steps": (it + 1) * horizon * n_envs, "train_success": rate,
                     "train_fall": fell / max(1, eps), "wall_s": time.time() - t0})
        if it % 25 == 0 or it == iters - 1:
            log(f"  ppo{'-res' if residual else ''} s{seed} it {it:4d} success {rate:.3f} fall {fell / max(1, eps):.3f} "
                f"std {torch.exp(ac.log_std).mean():.3f} ({time.time() - t0:.0f} s)")
    pol = Policy("ppo", ac.pi, norm, residual, scale)
    return pol, hist


# ============================================================================ SAC
class SquashedActor(nn.Module):
    def __init__(self, od, ad, h=128):
        super().__init__()
        self.net = mlp(od, 2 * ad, h=h, act=nn.ReLU, out_gain=0.01)
        self.ad = ad

    def forward(self, z):
        return self.net(z)

    def sample(self, z):
        out = self.net(z)
        mu, log_std = out[:, : self.ad], torch.clamp(out[:, self.ad :], -5, 1)
        std = torch.exp(log_std)
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        # log-prob with the tanh change of variables
        logp = (torch.distributions.Normal(mu, std).log_prob(u) - torch.log(1 - a * a + 1e-6)).sum(-1)
        return a, logp


@dataclass
class SACConfig:
    iters: int = 2500             # env steps per environment
    n_envs: int = 128
    updates: int = 16             # gradient steps per env step
    batch: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005            # target network averaging
    initial_alpha: float = 0.1
    warmup_env_steps: int = 20    # per environment, before the first update
    hidden: int = 128


def train_sac(cfg, residual, hp: SACConfig | None = None, seed=0, scale=0.3, log=print):
    hp = hp if hp is not None else SACConfig()
    torch.manual_seed(seed)
    # Minibatch indices come from a seeded generator. They came from NumPy's unseeded global RNG
    # before, so a SAC run could not be reproduced from its seed.
    sample_rng = np.random.default_rng(seed)
    iters, n_envs, batch, h = hp.iters, hp.n_envs, hp.batch, hp.hidden
    env = StepEnv(n_envs, cfg, seed=seed)
    od, ad = env.obs_dim + 1, env.act_dim
    actor = SquashedActor(od, ad, h=h)
    q1, q2 = mlp(od + ad, 1, h=h, act=nn.ReLU), mlp(od + ad, 1, h=h, act=nn.ReLU)
    q1t, q2t = mlp(od + ad, 1, h=h, act=nn.ReLU), mlp(od + ad, 1, h=h, act=nn.ReLU)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    log_alpha = torch.tensor(np.log(hp.initial_alpha), requires_grad=True)
    opt_a = torch.optim.Adam(actor.parameters(), lr=hp.lr)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=hp.lr)
    opt_al = torch.optim.Adam([log_alpha], lr=hp.lr)
    target_ent = -float(ad)  # the usual heuristic: -dim(action)
    gamma, tau = hp.gamma, hp.tau
    cap = iters * n_envs
    B = {"o": np.zeros((cap, od), np.float32), "a": np.zeros((cap, ad), np.float32), "r": np.zeros(cap, np.float32),
         "o2": np.zeros((cap, od), np.float32), "d": np.zeros(cap, np.float32)}
    ptr = 0
    norm = RunningNorm(od)
    # Normalizer statistics are frozen after a warm-up so stored and new observations stay comparable.
    warm = StepEnv(4096, cfg, seed=seed + 999)
    samples = []
    for _ in range(N_STEPS):
        samples.append(policy_obs(warm))
        warm.step(capture_point_action(warm.obs()))
    norm.update(np.concatenate(samples))
    hist, t0 = [], time.time()
    succ = fell = eps = 0
    for it in range(iters):
        raw = policy_obs(env)
        z = torch.as_tensor(norm(raw))
        with torch.no_grad():
            a, _ = actor.sample(z)
        a = a.numpy()
        exec_a = np.clip(capture_point_action(raw[:, :4]) + scale * a, -1, 1) if residual else a
        _, r, done, info = env.step(exec_a)
        # terminal observation for finished episodes (the env has already auto-reset them)
        o2 = policy_obs(env)
        term_raw = np.column_stack([info["terminal_obs"], np.full(n_envs, 1.0)]).astype(np.float32)
        o2[done] = term_raw[done]
        n = n_envs
        B["o"][ptr : ptr + n], B["a"][ptr : ptr + n], B["r"][ptr : ptr + n] = z.numpy(), a, r
        B["o2"][ptr : ptr + n], B["d"][ptr : ptr + n] = norm(o2), done.astype(np.float32)
        ptr += n
        succ += info["success"].sum()
        fell += (info["fell"] & done).sum()
        eps += done.sum()
        if ptr >= hp.warmup_env_steps * n_envs:
            for _ in range(hp.updates):
                idx = sample_rng.integers(0, ptr, batch)
                o, ac_, rr, o2b, dd = (torch.as_tensor(B[k][idx]) for k in ("o", "a", "r", "o2", "d"))
                with torch.no_grad():
                    a2, lp2 = actor.sample(o2b)
                    qt = torch.min(q1t(torch.cat([o2b, a2], 1)), q2t(torch.cat([o2b, a2], 1))).squeeze(-1)
                    y = rr + gamma * (1 - dd) * (qt - torch.exp(log_alpha) * lp2)
                qa, qb = q1(torch.cat([o, ac_], 1)).squeeze(-1), q2(torch.cat([o, ac_], 1)).squeeze(-1)
                lq = ((qa - y) ** 2).mean() + ((qb - y) ** 2).mean()
                opt_q.zero_grad()
                lq.backward()
                opt_q.step()
                an, lp = actor.sample(o)
                qn = torch.min(q1(torch.cat([o, an], 1)), q2(torch.cat([o, an], 1))).squeeze(-1)
                la = (torch.exp(log_alpha).detach() * lp - qn).mean()
                opt_a.zero_grad()
                la.backward()
                opt_a.step()
                lal = -(log_alpha * (lp.detach() + target_ent)).mean()
                opt_al.zero_grad()
                lal.backward()
                opt_al.step()
                with torch.no_grad():
                    for net, tgt in ((q1, q1t), (q2, q2t)):
                        for p, pt in zip(net.parameters(), tgt.parameters()):
                            pt.mul_(1 - tau).add_(tau * p)
        if (it + 1) % 50 == 0:
            rate = succ / max(1, eps)
            hist.append({"iter": it, "env_steps": (it + 1) * n_envs, "train_success": rate, "train_fall": fell / max(1, eps),
                         "wall_s": time.time() - t0, "alpha": float(torch.exp(log_alpha))})
            if (it + 1) % 250 == 0:
                log(f"  sac{'-res' if residual else ''} s{seed} it {it + 1:5d} success {rate:.3f} fall {fell / max(1, eps):.3f} "
                    f"alpha {float(torch.exp(log_alpha)):.4f} ({time.time() - t0:.0f} s)")
            succ = fell = eps = 0
    pol = Policy("sac", actor.net, norm, residual, scale)
    return pol, hist


@torch.no_grad()
def evaluate(policy, cfg, n=4000, seed=12345):
    """Success / fall rate over n episodes with fixed seeds. Returns dict + per-episode arrays."""
    env = StepEnv(n, cfg, seed=seed)
    done_any = np.zeros(n, bool)
    succ = np.zeros(n, bool)
    fell = np.zeros(n, bool)
    Ts, cops, steps = [], [], np.full(n, N_STEPS)
    push = env.push.copy()
    for k in range(N_STEPS):
        o = policy_obs(env)
        a = policy(o)
        ph = to_physical(a)
        active = ~done_any
        Ts.append(np.where(active, ph[:, 2], np.nan))
        cops.append(np.where(active, np.abs(ph[:, 3]), np.nan))
        _, _, done, info = env.step(a)
        newly = done & ~done_any
        succ |= info["success"] & newly
        fell |= info["fell"] & newly
        steps[newly & info["fell"]] = k + 1
        done_any |= done
    with warnings.catch_warnings():  # steps where every episode has already ended give NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        T_mean, cop_mean = np.nanmean(np.array(Ts), axis=1), np.nanmean(np.array(cops), axis=1)
    return {"success": succ.mean(), "fall": fell.mean(), "push": push, "succ_arr": succ, "fell_arr": fell,
            # per step k: mean over the episodes still running at step k
            "T_mean": T_mean, "cop_mean": cop_mean}
