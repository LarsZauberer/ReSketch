import math


class Physic_Engine:
    def __init__(self, mass: float, friction: float, time_scale: float=1.0, g: float = 9.81, action_scale: float=1.0):
        self.mass = mass  # The mass of the pen
        self.g = g  # The gravity constant
        self.friction = friction  # The friction koefficient of the pen on the paper
        self.time_scale = time_scale  # How fast the t variable is moving forward
        self.action_scale = action_scale
        
        # State data
        self.velocity = [.0, .0]
    
    def calc_position_step(self, pos: list, force: list):
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
        force[0] *= self.action_scale
        force[1] *= self.action_scale
        a = self.calc_acceleration(force)
        self.velocity[0] += a[0]*self.time_scale
        self.velocity[1] += a[1]*self.time_scale
        
        f = self.calc_friction()
        self.velocity[0] -= f[0]*self.time_scale
        self.velocity[1] -= f[1]*self.time_scale
        
        pos[0] += self.velocity[0]*self.time_scale
        pos[1] += self.velocity[1]*self.time_scale
        
        # Round the position to a whole number
        pos[0] = round(pos[0])
        pos[1] = round(pos[1])
        
        return pos

    def calc_acceleration(self, force: list):
        """
        calc_acceleration Calculate the acceleration of the force

        :param force: The force of the movement
        :type force: list
        :return: The acceleration of the move in the direction of the force
        :rtype: _type_
        """
        # Pure acceleration
        vel = [None, None]
        vel[0] = force[0]/self.mass
        vel[1] = force[1]/self.mass
        
        return vel
    
    def calc_friction(self):
        """
        calc_friction Calculate the friction of the move. If the pen is already in motion. It will get slower

        :return: The amount of friction for the velocity
        :rtype: list
        """
        vel = self.velocity.copy()
        # Calc friction and subtract it
        n = self.mass * self.g
        fric = n*self.friction
        
        fricVel = [None, None]
        
        # Make the friction go in the correct direction
        fricVel[0] = math.copysign(fric, vel[0])
        fricVel[1] = math.copysign(fric, vel[1])
        
        # The friction cannot go over 0 barrier
        fricVel[0] = vel[0] if self.check_too_much_friction(vel[0], fricVel[0]) else fricVel[0]
        fricVel[1] = vel[0] if self.check_too_much_friction(vel[1], fricVel[1]) else fricVel[1]
        
        return fricVel
    
    def check_too_much_friction(self, vel: float, fricVel: float):
        """
        check_too_much_friction Check if the friction wouldn't accelerate in the reversed direction

        :param vel: Current velocity on only one dimension
        :type vel: float
        :param fricVel: The calculated friction
        :type fricVel: float
        :return: True if the friction would accelerate in the wrong direction. Then the velocity should be 0
        :rtype: bool
        """
        positiv_and_negative = vel >= 0 and vel-fricVel <= 0
        negative_and_positive = vel <= 0 and vel-fricVel >= 0
        
        return positiv_and_negative or negative_and_positive
        
