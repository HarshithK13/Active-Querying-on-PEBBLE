import os
import math
import random
from collections import deque

import numpy as np

# NumPy 1.24+ compat for old aliases
for _name, _alias in [("int", int), ("bool", bool), ("float", float), ("complex", complex)]:
    if not hasattr(np, _name):
        setattr(np, _name, _alias)

import torch
import torch.nn.functional as F
from torch import nn
from torch import distributions as pyd

# Gym (classic) & wrappers
import gym
from gym.wrappers.time_limit import TimeLimit

# DMControl through dmc2gym (required for walker_* etc.)
import dmc2gym

# rlkit wrappers (your code uses this)
from rlkit.envs.wrappers import NormalizedBoxEnv

# --- Optional Meta-World support (don’t block DMControl runs) ---
# Some experiments import Meta-World; others (dm_control*) don't.
# Make this optional so DMControl jobs run without Meta-World installed.
_env_dict = None
try:
    import metaworld  # may not be installed

    try:
        # Old layout
        import metaworld.envs.mujoco.env_dict as _env_dict
    except ModuleNotFoundError:
        try:
            # Newer layout
            from metaworld.envs import env_dict as _env_dict
        except Exception:
            _env_dict = None
except Exception:
    _env_dict = None
# ---------------------------------------------------------------


def make_env(cfg):
    """Helper to create dm_control environment."""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=cfg.seed,
        visualize_reward=False
    )
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    return env


def ppo_make_env(env_id, seed):
    """Helper to create dm_control environment for PPO eval."""
    if env_id == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = env_id.split('_')[0]
        task_name = '_'.join(env_id.split('_')[1:])

    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=True
    )
    env.seed(seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    return env


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def make_metaworld_env(cfg):
    if _env_dict is None:
        raise RuntimeError(
            "Meta-World not available. Install a compatible version or run a dm_control task."
        )
    env_name = cfg.env.replace('metaworld_', '')
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls()
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(cfg.seed)
    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)


def ppo_make_metaworld_env(env_id, seed):
    if _env_dict is None:
        raise RuntimeError(
            "Meta-World not available. Install a compatible version or run a dm_control task."
        )
    env_name = env_id.replace('metaworld_', '')
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls()
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)
    return TimeLimit(env, env.max_path_length)


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def weight_init(m):
    """Custom weight init for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # do not clamp; use cache_size=1 instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        base_dist = pyd.Normal(loc, scale)
        super().__init__(base_dist, [TanhTransform()])

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0, unbiased=False)
            batch_count = x.shape[0]
            self.mean, self.var, self.count = update_mean_var_count_from_moments(
                self.mean, self.var, self.count, batch_mean, batch_var, batch_count
            )

    @property
    def std(self):
        return torch.sqrt(self.var)


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    # Standard parallel update (Welford-style)
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * (batch_count / tot_count)
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + (delta ** 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    return nn.Sequential(*mods)


def to_np(t):
    if t is None:
        return None
    if t.nelement() == 0:
        return np.array([])
    return t.detach().cpu().numpy()
