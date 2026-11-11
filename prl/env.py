"""Step-to-step push-recovery environment for a reduced-order humanoid.

Model: 3D linear inverted pendulum (LIPM), CoM at constant height, point foot with an
adjustable centre of pressure (ankle strategy). One decision per step:

    action = [foot x, foot y (outward), step duration, CoP x, CoP y]   (all normalized to [-1, 1])

Within a step the CoM follows the closed-form LIPM solution about the CoP. At touchdown the
swing foot becomes the new stance foot and coordinates are re-expressed relative to it. The
lateral axis is mirrored by stance side, so one policy serves both legs. Positions are in m,
velocities in m/s, times in s, all in the stance-foot frame.

What makes it more than a textbook exercise (the controller does not see any of this):
  * the pendulum frequency omega is randomized per episode (+/-5 % around sqrt(g/z0), i.e. about
    +/-10 % CoM height), standing in for CoM height variation and unmodelled whole-body dynamics,
  * foot placement noise (1.5 cm), touchdown timing delay (0-30 ms), CoP tracking noise,
  * kinematic limits: step length/width box, horizontal leg reach of 0.55 m from the CoM at
    touchdown, minimum swing time 0.3 s,
  * the push arrives as a random-direction velocity impulse at the start of the episode.
An episode is 8 steps. It fails if the horizontal distance from the stance foot to the CoM
exceeds 0.75 m during a step (the stance leg would have to be longer than it is). It succeeds
if it survives all 8 steps and ends with its DCM within 12 cm of where steady in-place
stepping would put it (it is back to a periodic gait, not just still upright).
"""
from dataclasses import dataclass, field

import numpy as np

GRAVITY_M_S2 = 9.81
COM_HEIGHT_M = 0.9
OMEGA_NOM = np.sqrt(GRAVITY_M_S2 / COM_HEIGHT_M)  # LIPM natural frequency, 1/s

# Physical action ranges (stance-relative, lateral mirrored so +y is toward the swing side):
# foot x, foot y, step duration, CoP x, CoP y
LO = np.array([-0.45, 0.10, 0.30, -0.05, -0.03])
HI = np.array([0.60, 0.45, 0.70, 0.10, 0.03])
MID, HALF = 0.5 * (LO + HI), 0.5 * (HI - LO)
N_STEPS = 8
REACH = 0.55                  # max horizontal swing-foot distance from the CoM at touchdown, m
MAX_STANCE_EXTENSION = 0.75   # max horizontal stance-foot-to-CoM distance before the leg gives out, m
DIVERGED_SPEED = 4.0          # CoM speed treated as a fall even if the leg has not overextended, m/s
NOMINAL_STEP_TIME = 0.4       # s
DCM_OFFSET = 0.04             # lateral DCM offset of steady in-place stepping (step width 0.2 m, T = 0.4 s), m
SUCCESS_DCM_TOLERANCE = 0.12  # m


@dataclass
class RewardWeights:
    alive: float = 1.0
    speed: float = 0.5            # per m/s of CoM speed after touchdown
    dcm_error: float = 0.5        # per m of DCM distance from the periodic-gait offset
    step_time: float = 0.2        # per s^2 of deviation from the nominal step time
    fall: float = -10.0           # replaces the step reward
    terminal_bonus: float = 5.0   # times exp(-dcm_error^2 / terminal_width^2) at the end of the episode
    terminal_width: float = 0.1   # m


@dataclass
class EnvConfig:
    push_max: float = 1.4             # training range of the initial velocity impulse, m/s
    push_min: float = 0.0
    omega_scale: tuple = (0.95, 1.05)
    foot_noise: float = 0.015         # m, standard deviation
    delay_max: float = 0.03           # s, uniform touchdown delay
    cop_noise: float = 0.005          # m, standard deviation
    reward: RewardWeights = field(default_factory=RewardWeights)


def to_physical(a):
    return MID + HALF * np.clip(a, -1.0, 1.0)


def to_normalized(p):
    return (p - MID) / HALF


class StepEnv:
    """`n` independent episodes stepped together. Finished episodes reset automatically."""

    obs_dim, act_dim = 4, 5

    def __init__(self, n, cfg: EnvConfig | None = None, seed=0):
        self.n, self.cfg = n, cfg if cfg is not None else EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.com_pos = np.zeros((n, 2))
        self.com_vel = np.zeros((n, 2))
        self.omega = np.full(n, OMEGA_NOM)
        self.step_index = np.zeros(n, dtype=int)
        self.push = np.zeros(n)
        self.reset_idx(np.arange(n))

    def reset_idx(self, idx):
        m = len(idx)
        r = self.rng
        lo, hi = self.cfg.omega_scale
        self.omega[idx] = OMEGA_NOM * r.uniform(lo, hi, m)
        # touchdown during in-place stepping: CoM between the feet (on the swing side, +y in the
        # mirrored frame) moving toward the new stance foot, DCM about 4 cm inside it
        self.com_pos[idx] = np.column_stack([r.normal(0, 0.01, m), 0.10 + r.normal(0, 0.01, m)])
        self.com_vel[idx] = np.column_stack([r.normal(0, 0.03, m), -0.20 + r.normal(0, 0.03, m)])
        magnitude = r.uniform(self.cfg.push_min, self.cfg.push_max, m)
        angle = r.uniform(-np.pi, np.pi, m)
        self.com_vel[idx] += np.column_stack([magnitude * np.cos(angle), magnitude * np.sin(angle)])
        self.push[idx] = magnitude
        self.step_index[idx] = 0

    def obs(self):
        return np.column_stack([self.com_pos, self.com_vel]).astype(np.float32)

    def dcm(self, omega=None):
        w = self.omega if omega is None else omega
        return self.com_pos + self.com_vel / w[:, None]

    def step(self, a_norm):
        """Returns obs, reward, done, info. Resets finished environments automatically."""
        a = to_physical(a_norm)
        r = self.rng
        weights = self.cfg.reward
        foot = a[:, 0:2].copy()
        step_time = a[:, 2] + r.uniform(0, self.cfg.delay_max, self.n)  # what actually happens
        cop = a[:, 3:5] + r.normal(0, self.cfg.cop_noise, (self.n, 2))
        w = self.omega[:, None]
        c0, v0 = self.com_pos, self.com_vel
        # LIPM about the CoP: c'' = w^2 (c - p), solved in closed form at mid-step and touchdown
        half_step = 0.5 * step_time[:, None]
        com_mid = cop + (c0 - cop) * np.cosh(w * half_step) + v0 / w * np.sinh(w * half_step)
        cosh_t, sinh_t = np.cosh(w * step_time[:, None]), np.sinh(w * step_time[:, None])
        com_touchdown = cop + (c0 - cop) * cosh_t + v0 / w * sinh_t
        vel_touchdown = (c0 - cop) * w * sinh_t + v0 * cosh_t
        # kinematic projection: the swing foot cannot land further than REACH from the CoM
        foot = foot + r.normal(0, self.cfg.foot_noise, (self.n, 2))
        to_foot = foot - com_touchdown
        distance = np.linalg.norm(to_foot, axis=1)
        too_far = distance > REACH
        foot[too_far] = com_touchdown[too_far] + to_foot[too_far] * (REACH / distance[too_far])[:, None]
        # Stance-leg overextension. Each coordinate of c(t) - p is a sum of exp(+wt) and exp(-wt),
        # whose square is convex in t, so with the CoP within 10 cm of the foot the distance peaks
        # at an end of the step in all but marginal cases. The mid-step check covers those.
        extension = np.maximum(np.linalg.norm(com_mid, axis=1), np.linalg.norm(com_touchdown, axis=1))
        # touchdown: new stance frame, mirror lateral axis
        mirror = np.array([1.0, -1.0])
        self.com_pos = (com_touchdown - foot) * mirror
        self.com_vel = vel_touchdown * mirror
        self.step_index += 1

        speed = np.linalg.norm(self.com_vel, axis=1)
        fell = (extension > MAX_STANCE_EXTENSION) | ~np.isfinite(self.com_pos).all(1) | (speed > DIVERGED_SPEED)
        dcm_error = np.linalg.norm(self.dcm() - np.array([0.0, DCM_OFFSET]), axis=1)
        reward = (weights.alive - weights.speed * speed - weights.dcm_error * dcm_error
                  - weights.step_time * (step_time - NOMINAL_STEP_TIME) ** 2)
        done_time = self.step_index >= N_STEPS
        success = done_time & ~fell & (dcm_error < SUCCESS_DCM_TOLERANCE)
        reward = np.where(fell, weights.fall, reward)
        terminal_bonus = weights.terminal_bonus * np.exp(-(dcm_error**2) / weights.terminal_width**2)
        reward = np.where(done_time & ~fell, reward + terminal_bonus, reward)
        done = fell | done_time
        info = {"fell": fell.copy(), "success": success.copy(), "push": self.push.copy(), "T": step_time.copy(),
                "terminal_obs": self.obs()}
        if done.any():
            self.reset_idx(np.where(done)[0])
        return self.obs(), reward.astype(np.float32), done, info


def capture_point_action(obs, omega=OMEGA_NOM, step_time=0.3, step_width=0.2):
    """Classic DCM stepping with the nominal model: fixed step time, CoP at the foot centre,
    place the foot so the DCM would come to rest over it (plus the lateral offset of periodic
    in-place stepping). Returns a normalized action.

    A step time of 0.3 s (the minimum swing time) was the best of {0.30, 0.35, 0.40}: the DCM
    grows by exp(omega T) per step, so a shorter step amplifies model and timing errors less. At
    0.4 s this controller fell in 22 % of episodes with no push at all."""
    com_pos, com_vel = obs[:, 0:2], obs[:, 2:4]
    dcm = com_pos + com_vel / omega
    growth = np.exp(omega * step_time)
    dcm_touchdown = dcm * growth  # with the CoP at the stance foot
    lateral_offset = step_width / (growth + 1.0)
    foot = np.column_stack([dcm_touchdown[:, 0], dcm_touchdown[:, 1] + lateral_offset])
    phys = np.column_stack([foot, np.full(len(obs), step_time), np.zeros((len(obs), 2))])
    phys = np.clip(phys, LO, HI)
    return to_normalized(phys).astype(np.float32)
