from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
import pygame as pg

import numpy as np


class Canvas(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=1, maximum=200, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(9,), dtype=np.int32, name='observation')
        self._state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.status = np.array(self._state).reshape(3, 3)
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # TODO: Set the _state
        self._state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        pass
    
    def render(self):
        pass

    def metrics(self, policy, num_episodes=10):
        pass
    
    def _calc_dir_length(self, action):
        dir_count = 1
        for i in range(1, 201):
            if action == dir_count:
                return dir_count, action // dir_count
            
            dir_count += 1
            if dir_count >= 5:
                dir_count = 1
        
        raise ValueError('Invalid action: {}'.format(action))
