import gym
import magent
import numpy as np
from typing import Callable, List, Any, Tuple, Optional

from gym import spaces
from gym.vector.vector_env import VectorEnv


class MAgentEnv(VectorEnv):
    def __init__(self, env: magent.GridWorld, handle,
                 reset_env_func: Callable[[], None],
                 is_slave: bool = False,
                 steps_limit: Optional[int] = None):
        reset_env_func()
        action_space = self.handle_action_space(env, handle)
        observation_space = self.handle_obs_space(env, handle)

        count = env.get_num(handle)

        super(MAgentEnv, self).__init__(count, observation_space,
                                        action_space)
        self.action_space = self.single_action_space
        self._env = env
        self._handle = handle
        self._reset_env_func = reset_env_func
        self._is_slave = is_slave
        self._steps_limit = steps_limit
        self._steps_done = 0

    @classmethod
    def handle_action_space(cls, env: magent.GridWorld,
                            handle) -> gym.Space:
        return spaces.Discrete(env.get_action_space(handle)[0])

    @classmethod
    def handle_obs_space(cls, env: magent.GridWorld, handle) -> gym.Space:
        # view shape
        v = env.get_view_space(handle)
        # extra features
        r = env.get_feature_space(handle)

        # rearrange planes to pytorch convention
        view_shape = (v[-1],) + v[:2]
        view_space = spaces.Box(low=0.0, high=1.0,
                                shape=view_shape)
        extra_space = spaces.Box(low=0.0, high=1.0, shape=r)
        return spaces.Tuple((view_space, extra_space))

    def reset_wait(self):
        self._steps_done = 0
        if not self._is_slave:
            self._reset_env_func()
        return self.handle_observations(self._env, self._handle)

    @classmethod
    def handle_observations(cls, env: magent.GridWorld,
                            handle) -> List[Tuple[np.ndarray,
                                                  np.ndarray]]:
        view_obs, feats_obs = env.get_observation(handle)
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
        act = np.array(actions, dtype=np.int32)
        self._env.set_action(self._handle, act)

    def step_wait(self):
        self._steps_done += 1
        if not self._is_slave:
            done = self._env.step()
            self._env.clear_dead()
            if self._steps_limit is not None and self._steps_limit <= self._steps_done:
                done = True
        else:
            done = False

        obs = self.handle_observations(self._env, self._handle)
        r = self._env.get_reward(self._handle).tolist()
        dones = [done] * len(r)
        if done:
            obs = self.reset()
            dones = [done] * self.num_envs
            r = [0.0] * self.num_envs
        return obs, r, dones, {}


def config_forest(map_size):
    gw = magent.gridworld

    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"embedding_size": 10})

    deer = cfg.register_agent_type(
        "deer",
        {'width': 1, 'length': 1, 'hp': 5, 'speed': 1,
         'view_range': gw.CircleRange(1), 'attack_range': gw.CircleRange(0),
         'damage': 0, 'step_recover': 0.2,
         'food_supply': 0, 'kill_supply': 8,
         # added to standard 'forest' setup to motivate deers to live longer :)
         'step_reward': 1,
         })

    tiger = cfg.register_agent_type(
        "tiger",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 1,
         'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(1),
         'damage': 3, 'step_recover': -0.5,
         'food_supply': 0, 'kill_supply': 0,
         'step_reward': 1, 'attack_penalty': -0.1,
         })

    deer_group  = cfg.add_group(deer)
    tiger_group = cfg.add_group(tiger)

    return cfg


def config_double_attack(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"embedding_size": 10})

    deer = cfg.register_agent_type("deer", {
        'width': 1, 'length': 1, 'hp': 5, 'speed': 1,
        'view_range': gw.CircleRange(1),
        'attack_range': gw.CircleRange(0),
        'step_recover': 0.2,
        'kill_supply': 8,
        # added to standard 'double_attack' setup in MAgent.
        # Needed to get reward for longer episodes
        'step_reward': 0.1,
    })

    tiger = cfg.register_agent_type("tiger", {
        'width': 1, 'length': 1, 'hp': 10, 'speed': 1,
        'view_range': gw.CircleRange(4),
        'attack_range': gw.CircleRange(1),
        'damage': 1, 'step_recover': -0.2,
        # added to standard 'double_attack' setup in MAgent.
        # Needed to get reward for longer episodes
        # but this breaks the tigers' incentive for double
        # attack :(. Better exploration is needed, as
        # double attack is more profitable
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
