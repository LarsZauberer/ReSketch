from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import math

import time


class Physic_Engine:
    def __init__(self, friction: float, action_scale: float=1.0):
        self.friction = friction  # The friction koefficient of the pen on the paper
        self.action_scale = action_scale
        # State data
        self.velocity = [.0, .0]
    

    def calc_new_pos(self, pos: list, force: list, update_velocity : bool = True):
        """
        calc_position_step Calculate the next position on the canvas with the velocity and acceleration. Input is the force

        :param pos: The position where the agent currently is
        :type pos: list
        :param force: The force of the movement
        :type force: list
        :return: a new position (already rounded to integers)
        :rtype: list
        """

        pos = list(pos)

        vel = self.calc_velocity(force)
        if update_velocity: self.velocity = vel

        pos = [round(p+v) for p, v in zip(pos, vel)]
    
        return pos



    def calc_velocity(self, force: list):
        #t, m = 1 -> force == added velocity
        force = [f * self.action_scale for f in force]

        new_vel = [sv+f for sv, f in zip(self.velocity, force)]
        new_vel_size = self.vector_size(new_vel)

        #add friction (constant velocity in opposite direction, as long as it does not accelerate in opposite direction)
        if new_vel_size > self.friction:
            new_vel = [v-(nv*self.friction) for v, nv in zip(new_vel, self.normalize_vector(new_vel))]
        else:
            new_vel = [0 for _ in new_vel]

        return new_vel


    def normalize_vector(self, vector: list):
        scalar = self.vector_size(vector)
        return [v/scalar for v in vector]
    def vector_size(self, vector: list):
        return  math.sqrt(sum(v**2 for v in vector))
    
    
    
        
