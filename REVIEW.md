# Code Review: push-recovery-rl

Reviewed as a staff engineer on a learning-for-locomotion team would review a pull request that adds a vectorized stepping environment, PPO and SAC written from scratch, residual variants and an evaluation protocol. Scope: `prl/` (environment, agents), the scripts, tests and build files.

## 1. Overall assessment

A well-designed small RL study. The environment is the closed-form LIPM solution, tested against an RK4 integration, with deliberate model mismatch, noise and kinematic limits. Every policy and the baseline are evaluated on the same seeded pushes, with a shifted-dynamics test set, and the README separates what the policies learned from what they scored.

The review found one reproducibility defect. SAC drew its minibatch indices from NumPy's unseeded global random generator, so no SAC result could be reproduced from its seed. It is fixed, a test catches it, and all twelve policies were retrained. PPO's numbers did not change, which confirms the environment refactor preserved behavior. SAC's moved by at most 0.6 percentage points.

The rest of the changes are about self-documentation. Hyperparameters were literals inside the training loops, and the fall, success and reward constants were bare numbers. Short math names leaked into the public environment interface (`env.c`, `env.v`, `env.k`), and a shared mutable default argument was hiding in the constructor.

## 2. Scorecard

| Area | Score | Basis |
|---|---|---|
| Correctness | 8 | Closed-form dynamics tested against RK4, the frame mirroring tested, the tanh-squashed log-probability tested, GAE and PPO clipping standard. |
| Architecture | 7 | Environment, agents and evaluation are separate. PPO and SAC are single long functions. |
| Readability | 7 | After this pass the environment reads in physical terms. The training loops still use one-letter buffer names (`O`, `A`, `R`, `B`). |
| Self-documentation | 8 | Constants, reward weights and hyperparameters are named with units. The module docstring states every source of mismatch. |
| Maintainability | 7 | Policies are pickled objects, so a class change can break loading old results. |
| Testing | 8 | Nine tests, including a reproducibility test that fails on the old sampler and a fall test. No test pins a trained policy's performance. |
| Performance | Not assessed | CPU research code. Training cost is reported. |
| Reliability | 7 | Non-finite states count as falls. Every run is now seeded end to end. |
| API design | 7 | `PPOConfig` and `SACConfig` replace keyword soup. `StepEnv` auto-resets inside `step`, which is efficient but makes terminal observations a side channel (`info["terminal_obs"]`). |
| Domain-specific correctness | 8 | DCM-based baseline derived correctly, reach projection exact, overextension checked where the distance can peak, phase in the observation makes the 8-step horizon Markov so treating time-outs as terminal is valid. |
| Safety and robustness | Not assessed | A reduced-order model. Nothing here bears on a robot's safety. |

## 3. Problems

### Fixed in this review

| Severity | Location | What | Why | Fix and evidence |
|---|---|---|---|---|
| High | `train_sac` | Minibatch indices came from `np.random.randint`, NumPy's unseeded global generator. | Two SAC runs with the same seed diverged, so the published SAC numbers could not be reproduced. | A generator seeded from the run seed. `test_sac_is_reproducible_from_its_seed` fails on the old code (mutation-checked). All policies retrained and the README updated. |
| Medium | `train_ppo`, `train_sac` | Learning rates, discount, GAE lambda, clip range, epochs, minibatches, loss coefficients, gradient clip, target smoothing, initial temperature and warm-up were literals in the loops. | Tuning knobs without names, and no single place to read the configuration of a run. | `PPOConfig` and `SACConfig` with the same defaults. PPO results bit-identical after the change. |
| Medium | `prl/env.py` | Fall extension (0.75 m), divergence speed (4 m/s), success tolerance (12 cm), nominal step time and every reward weight were bare numbers. | The reward and termination are the task specification. | Named constants and a `RewardWeights` dataclass in `EnvConfig`. `test_falls_when_the_stance_leg_overextends`. |
| Low | `StepEnv` | Public state as `c`, `v`, `k`. | Callers (tests, scripts, the agents) had to know the math notation. | `com_pos`, `com_vel`, `step_index`. |
| Low | `StepEnv.__init__` | `cfg: EnvConfig = EnvConfig()`, one instance shared by every environment created with the default. | Harmless while nothing mutates it, a trap once something does. | `cfg=None` and a fresh default. |
| Low | `evaluate` | An import inside the per-step loop. | Noise in a hot loop. | Moved to the module header. |

### Open

| Severity | Location | What | Why | Recommended fix |
|---|---|---|---|---|
| Low | `StepEnv.step` | The step-time penalty uses the realized step time, which includes the 0 to 30 ms touchdown delay. | The agent is penalized for noise it cannot control. That adds variance and shifts the optimum about 15 ms short. | Penalize the commanded step time. Changing it means retraining, so it is left as a documented choice. |
| Low | `train_ppo`, `train_sac` | 90 to 140-line functions mixing rollout, advantage computation and updates. | Hard to test the pieces separately. | Split into `collect`, `advantages`, `update`. |
| Low | `results/*.pkl` | Policies are pickled `Policy` objects. | A class change can make them unloadable. | Save `state_dict`s, normalizer statistics and the config. |
| Low | evaluation | Three seeds per variant. | Enough to show the seeds agree within 1.5 points, not enough to rank variants that close. | More seeds before ranking SAC-full against PPO-full. |

## 4. Readability and self-documentation

```python
# before
fell = (ext > 0.75) | ~np.isfinite(self.c).all(1) | (np.linalg.norm(self.v, axis=1) > 4.0)
rew = 1.0 - 0.5 * speed - 0.5 * np.linalg.norm(xi - np.array([0.0, DCM_OFFSET]), axis=1) - 0.2 * (T - 0.4) ** 2
...
idx = np.random.randint(0, ptr, batch)

# after
fell = (extension > MAX_STANCE_EXTENSION) | ~np.isfinite(self.com_pos).all(1) | (speed > DIVERGED_SPEED)
reward = (weights.alive - weights.speed * speed - weights.dcm_error * dcm_error
          - weights.step_time * (step_time - NOMINAL_STEP_TIME) ** 2)
...
idx = sample_rng.integers(0, ptr, batch)  # seeded from the run's seed
```

The new comment on the overextension check records why checking the middle and end of the step is enough: each coordinate of $c(t) - p$ is a sum of $e^{\omega t}$ and $e^{-\omega t}$, whose square is convex in $t$.

## 5. Architecture and design

Running thousands of environments as NumPy arrays with a closed-form step is the right choice for a 4-state model: it makes 4 million PPO steps cost about two minutes. The `Policy` wrapper shared by PPO, SAC and the baseline means every controller is evaluated by the same code. The residual formulation, $a = \mathrm{clip}(a_{base} + 0.3\,a_\pi)$, is implemented once and reused by both algorithms.

## 6. Testing and verification

| Claim | Verified by |
|---|---|
| The closed-form step equals RK4 integration, including the frame change | Test |
| The baseline walks in place without disturbances | Test |
| Swing feet never land beyond reach | Test |
| Overextension ends the episode with the fall reward | Test |
| Running normalization matches NumPy | Test |
| The squashed-Gaussian log-probability is correct | Test against a manual computation |
| A zero residual reproduces the baseline | Test |
| SAC is reproducible from its seed | Test, mutation-checked |
| The refactor preserved the environment | PPO results identical after retraining |
| Any of it on a robot | Not applicable, the model is a LIPM |

## 7. Prioritized action plan

Must fix before commit: none outstanding.

Before sharing: save policies as state dicts, and split the training functions.

Recommended: penalize the commanded rather than the realized step time on the next retraining, and add seeds before ranking close variants.

Optional: a regression test that a short PPO run reaches a minimum success rate.

## 8. Final verdict

| Context | Verdict |
|---|---|
| Personal or portfolio project | STRONG |
| Shared codebase | ACCEPTABLE |
| Simulation study | STRONG |
| Hardware | Not applicable. A LIPM stepping policy would need a whole-body controller and sim-to-real work before it meant anything on a robot. |
