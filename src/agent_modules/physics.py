import math


class Physic_Engine:
    def __init__(self, mass: float, friction: float, time_scale: float=1.0, g: float = 9.81):
        self.mass = mass  # The mass of the pen
        self.g = g  # The gravity constant
        self.friction = friction  # The friction koefficient of the pen on the paper
        self.time_scale = time_scale  # How fast the t variable is moving forward
        
        # State data
        self.velocity = [.0, .0]
    
    def calc_position_step(self, pos: list, force: list):
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
        # Pure acceleration
        vel = [None, None]
        vel[0] = force[0]/self.mass
        vel[1] = force[1]/self.mass
        
        return vel
    
    def calc_friction(self):
        vel = self.velocity.copy()
        # Calc friction and subtract it
        n = self.mass * self.g
        fric = n*self.friction
        
        fricVel = [None, None]
        
        fricVel[0] = math.copysign(fric, vel[0])
        fricVel[1] = math.copysign(fric, vel[1])
        
        print(vel)
        print(fricVel)
        
        # The friction cannot go over 0
        fricVel[0] = 0 if self.check_too_much_friction(vel[0], fricVel[0]) else fricVel[0]
        fricVel[1] = 0 if self.check_too_much_friction(vel[1], fricVel[1]) else fricVel[1]
        
        return fricVel
    
    def check_too_much_friction(self, vel: float, fricVel: float):
        positiv_and_negative = vel >= 0 and vel-fricVel <= 0
        negative_and_positive = vel <= 0 and vel-fricVel >= 0
        
        return positiv_and_negative or negative_and_positive
        
