import numpy as np
import torch

from prl.agents import PPOConfig, Policy, RunningNorm, SACConfig, SquashedActor, evaluate, policy_obs, train_ppo, train_sac
from prl.env import LO, HI, OMEGA_NOM, REACH, EnvConfig, StepEnv, capture_point_action, to_normalized

QUIET = EnvConfig(push_min=0.0, push_max=1e-9, omega_scale=(1.0, 1.0), foot_noise=0.0, delay_max=0.0, cop_noise=0.0)


def test_lipm_closed_form_matches_numerical_integration():
    env = StepEnv(1, QUIET, seed=0)
    c0, v0 = env.com_pos.copy(), env.com_vel.copy()
    a = to_normalized(np.array([[0.1, 0.2, 0.37, 0.03, -0.01]]))
    env.step(a)
    # integrate c'' = w^2 (c - p) with RK4 at 0.1 ms, then apply the same frame change
    p, T, w = np.array([0.03, -0.01]), 0.37, OMEGA_NOM
    x = np.concatenate([c0[0], v0[0]])
    f = lambda s: np.concatenate([s[2:], w**2 * (s[:2] - p)])
    h = 1e-4
    for _ in range(int(round(T / h))):
        k1 = f(x); k2 = f(x + 0.5 * h * k1); k3 = f(x + 0.5 * h * k2); k4 = f(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    c_new = x[:2] - np.array([0.1, 0.2])
    c_new[1] *= -1
    v_new = x[2:] * np.array([1, -1])
    assert np.allclose(env.com_pos[0], c_new, atol=1e-8)
    assert np.allclose(env.com_vel[0], v_new, atol=1e-8)


def test_baseline_walks_in_place_without_disturbances():
    env = StepEnv(100, QUIET, seed=1)
    for _ in range(8):
        _, _, done, info = env.step(capture_point_action(env.obs()))
    assert info["success"].all()


def test_kinematic_projection_respects_reach():
    env = StepEnv(5000, EnvConfig(push_min=1.5, push_max=2.0), seed=2)
    a = np.random.default_rng(0).uniform(-1, 1, (5000, 5))
    env.step(a)
    # Survivors: c is relative to the new stance foot, which was projected to within REACH of
    # the CoM (foot noise is applied before the projection, so the bound is exact). Episodes that
    # fell were auto-reset to a CoM about 0.1 m from the foot, which also satisfies it.
    assert np.all(np.linalg.norm(env.com_pos, axis=1) <= REACH + 1e-9)


def test_running_norm_matches_numpy():
    x = np.random.default_rng(0).normal(3, 2, (10000, 3))
    n = RunningNorm(3)
    for chunk in np.array_split(x, 7):
        n.update(chunk)
    assert np.allclose(n.mean, x.mean(0), atol=1e-6)
    assert np.allclose(n.var, x.var(0), rtol=1e-3)


def test_squashed_gaussian_logprob():
    torch.manual_seed(0)
    actor = SquashedActor(3, 2)
    z = torch.randn(5, 3)
    torch.manual_seed(1)
    a, logp = actor.sample(z)
    out = actor(z)
    mu, ls = out[:, :2], torch.clamp(out[:, 2:], -5, 1)
    u = torch.atanh(torch.clamp(a, -1 + 1e-6, 1 - 1e-6))
    manual = (torch.distributions.Normal(mu, torch.exp(ls)).log_prob(u) - torch.log(1 - a * a + 1e-6)).sum(-1)
    assert torch.allclose(logp, manual, atol=1e-3)


def test_residual_policy_with_zero_correction_is_the_baseline():
    class Zero(torch.nn.Module):
        def forward(self, z):
            return torch.zeros(z.shape[0], 5)

    env = StepEnv(64, EnvConfig(), seed=3)
    norm = RunningNorm(5)
    pol = Policy("ppo", Zero(), norm, residual=True)
    o = policy_obs(env)
    assert np.allclose(pol(o), capture_point_action(o[:, :4]))


def test_ppo_smoke_and_evaluate():
    pol, hist = train_ppo(EnvConfig(), residual=True, hp=PPOConfig(iters=3, n_envs=64, horizon=8), log=lambda *_: None)
    assert len(hist) == 3
    r = evaluate(pol, EnvConfig(), n=200)
    assert 0.0 <= r["success"] <= 1.0 and 0.0 <= r["fall"] <= 1.0
    assert np.all(r["T_mean"] >= LO[2] - 1e-9) and np.all(r["T_mean"] <= HI[2] + 1e-9)


def test_sac_is_reproducible_from_its_seed():
    # Minibatches used NumPy's unseeded global RNG, so two runs with one seed diverged.
    hp = SACConfig(iters=30, n_envs=16, updates=2, batch=32, warmup_env_steps=5, hidden=32)
    policies = [train_sac(EnvConfig(), residual=True, hp=hp, seed=4, log=lambda *_: None)[0] for _ in range(2)]
    obs = policy_obs(StepEnv(32, EnvConfig(), seed=9))
    assert np.array_equal(policies[0](obs), policies[1](obs))


def test_falls_when_the_stance_leg_overextends():
    env = StepEnv(1, QUIET, seed=0)
    env.com_vel[:] = [[3.0, 0.0]]  # far faster than any step can catch
    _, reward, done, info = env.step(capture_point_action(env.obs()))
    assert done[0] and info["fell"][0] and not info["success"][0]
    assert reward[0] == EnvConfig().reward.fall
