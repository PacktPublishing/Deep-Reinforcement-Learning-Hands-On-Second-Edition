import gym
import magent
import numpy as np
from typing import Callable

from gym import spaces
from gym.vector.vector_env import VectorEnv


class MAgentEnv(VectorEnv):
    def __init__(self, env: magent.GridWorld, handle,
                 reset_env_func: Callable[[], None]):
        reset_env_func()
        action_space = spaces.Discrete(env.get_action_space(handle)[0])

        # view shape
        v = env.get_view_space(handle)
        # extra features
        r = env.get_feature_space(handle)

        # rearrange planes to pytorch convention
        view_shape = (v[-1],) + v[:2]
        view_space = spaces.Box(low=0.0, high=1.0, shape=view_shape)
        extra_space = spaces.Box(low=0.0, high=1.0, shape=r)
        observation_space = spaces.Tuple((view_space, extra_space))

        count = env.get_num(handle)

        super(MAgentEnv, self).__init__(count, observation_space, action_space)
        self.action_space = self.single_action_space
        self._env = env
        self._handle = handle
        self._reset_env_func = reset_env_func

    def reset_wait(self):
        self._reset_env_func()
        return self._build_observation()

    def _build_observation(self):
        view_obs, feats_obs = self._env.get_observation(self._handle)
        entries = view_obs.shape[0]
        if entries == 0:
            return []
        # copy data
        view_obs = np.array(view_obs)
        feats_obs = np.array(feats_obs)
        view_obs = np.moveaxis(view_obs, 3, 1)

        res = []
        for o_view, o_feats in zip(np.vsplit(view_obs, entries),
                                   np.vsplit(feats_obs, entries)):
            res.append((o_view[0], o_feats[0]))
        return res

    def step_async(self, actions):
        self._env.set_action(self._handle, np.array(actions, dtype=np.int32))

    def step_wait(self):
        done = self._env.step()
        self._env.clear_dead()

        obs = self._build_observation()
        r = self._env.get_reward(self._handle).tolist()
        dones = [done] * len(r)
        if done:
            obs = self.reset()
            dones = [done] * self.num_envs
            r = [0.0] * self.num_envs
        return obs, r, dones, {}


def config_double_attack(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"embedding_size": 10})

    deer = cfg.register_agent_type(
        "deer",
        {'width': 1, 'length': 1, 'hp': 5, 'speed': 1,
         'view_range': gw.CircleRange(1), 'attack_range': gw.CircleRange(0),
         'step_recover': 0.2,
         'kill_supply': 8,
         })

    tiger = cfg.register_agent_type(
        "tiger",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 1,
         'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(1),
         'damage': 1, 'step_recover': -0.2,
         # added to standard 'double_attack' setup in MAgent. Needed to get reward for longer episodes
         'step_reward': 0.1,
         })

    deer_group  = cfg.add_group(deer)
    tiger_group = cfg.add_group(tiger)

    a = gw.AgentSymbol(tiger_group, index='any')
    b = gw.AgentSymbol(tiger_group, index='any')
    c = gw.AgentSymbol(deer_group,  index='any')

    # tigers get reward when they attack a deer simultaneously
    e1 = gw.Event(a, 'attack', c)
    e2 = gw.Event(b, 'attack', c)
    cfg.add_reward_rule(e1 & e2, receiver=[a, b], value=[1, 1])

    return cfg
