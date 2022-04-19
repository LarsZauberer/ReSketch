from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
import matplotlib.pyplot as plt

import numpy as np


class Canvas(py_environment.PyEnvironment):
    def __init__(self, img: np.ndarray, max_strokes: int=200):
        input_shape = (28*28)+(28*28)+2 # 28x28 image, 28x28 image, 2D Location
        
        self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=4*4-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(input_shape,), dtype=np.int32, name='observation')
        
        # States
        self.originalImage = self.preprocess_image(img)
        self.currentCanvas = np.zeros(28*28, dtype=np.int32)
        self.pen_location = np.array([5, 5], dtype=np.int32)
        
        self.max_strokes = max_strokes
        self.strokes = 0
        
        self._episode_ended = False
        
        self._state = self.create_state()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.currentCanvas = np.zeros(28*28, dtype=np.int32)
        self.pen_location = np.array([5, 5], dtype=np.int32)
        self.strokes = 0
        
        self._episode_ended = False
        
        self._state = self.create_state()
        
        return ts.restart(np.array(self.create_state(), dtype=np.int32))

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        
        reward = 0.0
        
        direction, length, drawing = self._calc_dir_length(action)
        
        canvasre = self.currentCanvas.reshape(28, 28)
        canvBefore = canvasre.copy().reshape(28*28,)
        
        # Draw the line
        for i in range(length):
            if drawing:
                canvasre[self.pen_location[1]][self.pen_location[0]] = 255
            if direction == 1:
                self.pen_location[0] += 1
            elif direction == 2:
                self.pen_location[1] += 1
            elif direction == 3:
                self.pen_location[0] -= 1
            elif direction == 4:
                self.pen_location[1] -= 1
            
            # Bordering
            if self.pen_location[0] >= 28:
                self.pen_location[0] = 27
            if self.pen_location[0] < 0:
                self.pen_location[0] = 0
            if self.pen_location[1] >= 28:
                self.pen_location[1] = 27
            if self.pen_location[1] < 0:
                self.pen_location[1] = 0
                
        self.currentCanvas = canvasre.reshape(28*28,)


        # Calculate the current reward
        for i, e, j in zip(self.currentCanvas, self.originalImage, canvBefore):
            if i == e and i == 255:
                if i != j:
                    reward += 1
        
        self.strokes += 1
        
        self._state = self.create_state()
        
        if self.strokes >= self.max_strokes:
            self._episode_ended = True
            return ts.termination(np.array(self.create_state(), dtype=np.int32), reward)
        else:
            return ts.transition(np.array(self.create_state(), dtype=np.int32), reward=reward, discount=0.5)
        
    def render(self, mode='None'):
        plt.imshow(self.currentCanvas.reshape(28, 28), cmap='gray')
        plt.show()

    def metrics(self, policy, num_episodes=10):
        total_return = 0.0
        for _ in range(num_episodes):

            time_step = self.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = self.step(action_step.action)
                episode_return += time_step.reward
                total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]
    
    def _calc_dir_length(self, action):
        dir_count = 1
        length = 1
        for i in range(200):
            if abs(action) == i:
                return dir_count, length, True

            dir_count += 1
            if dir_count >= 5:
                dir_count = 1
                length += 1

        raise ValueError(f'Invalid action: {action}')

    def create_state(self):
        return list(self.originalImage) + list(self.currentCanvas) + list(self.pen_location)
    
    def preprocess_image(self, img):
        a = img.reshape(28*28,)
        for index, item in enumerate(a):
            if item > 0:
                a[index] = 255
        return a
