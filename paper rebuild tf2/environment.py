import os
import random
from types import NoneType
import numpy as np
import math as ma
import time
import matplotlib.pyplot as plt


class ShapeDraw(object):
    def __init__(self, sidelength, patchsize, referenceData):
        self.s = sidelength
        self.p = patchsize #sidelength of patch (local Input). must be odd
        
        #Input gloabal stream
        self.referenceData = referenceData
        self.canvas = np.zeros((self.s, self.s))
        self.distmap = np.zeros((self.s, self.s))
        self.colmap = np.zeros((self.s, self.s))
        
        #Input local stream
        self.ref_patch = np.zeros((self.p, self.p))
        self.canvas_patch = np.zeros((self.p, self.p))

        #possible outputs
        self.n_actions= self.p*self.p*2 # For each pixel, is an action option (location of that pixel)

        #initializes rest
        self.lastSim = 0 # Last similarity between reference and canvas
        self.reference = referenceData[0] # Pick just the first image of the reference data in the first initialization
        self.curRef = 0
        self.agentPos = [0,0]
        self.isDrawing = 0 #! 0 = not Drawing, 1 = Drawing (not bool because NN)
        self.set_agentPos([random.randint(1, self.s-2), random.randrange(1, self.s-2)]) # Set a random start location for the agent (but with one pixel margin)

        #rendering / visualization
        self.fig = plt.figure(figsize=(10, 7))

    def step(self, agent_action):
        action = [0, 0]
        self.isDrawing = 1
        
        x = agent_action % self.p
        y = agent_action // self.p
        if y >= self.p:
            y -= self.p
            self.isDrawing = 0
        
        ownpos = (self.p-1)/2
        action = [int(self.agentPos[0]+x-ownpos) , int(self.agentPos[1]+y-ownpos)]

        penalty = 0
        if abs(x) < (self.p-1)/2 or abs(y) < (self.p-1)/2:
            penalty = -0.05

        if self.move_isLegal(action):
            self.set_agentPos(action)
        else:
            self.isDrawing = 0
            penalty = -0.01
        
        reward = self.reward() if self.isDrawing else 0.0
        reward += penalty

        return np.array([self.reference, self.canvas, self.distmap, self.colmap]), np.array([self.ref_patch, self.canvas_patch]), reward

    def set_agentPos(self, pos):
        if self.isDrawing:
            #print(type(self.canvas), type(drawline(self.agentPos, pos, self.canvas)))
            self.canvas = drawline(self.agentPos, pos, self.canvas)
        self.agentPos = pos
        self.update_distmap()
        self.update_patch()
        self.update_colmap()
        
    def update_distmap(self):
        x0 = self.agentPos[0]
        y0 = self.agentPos[1]
        for y in range(self.s):
            for x in range(self.s):
                dist =  ma.sqrt((x-x0)**2 + (y-y0)**2)/self.s
                self.distmap[y][x] = dist

    def update_colmap(self):
        for y in range(self.s):
            for x in range(self.s):
                self.colmap[y][x] = self.isDrawing
    
    def update_patch(self):
        patchX = int(self.agentPos[0]-(self.p-1)/2)
        patchY = int(self.agentPos[1]-(self.p-1)/2)
        for y in range(self.p):
            for x in range(self.p):
                yInd = 0 if patchY+y >= len(self.reference) or patchY+y < 0 else patchY+y
                xInd = 0 if patchX+x >= len(self.reference[0]) or patchX+x < 0 else patchX+x

                self.ref_patch[y][x] = self.reference[yInd][xInd]
                self.canvas_patch[y][x] = self.canvas[yInd][xInd]

    def reward(self):
        #calculates reward of action based on gained similarity and length of step
        reward = 0
        similarity = 0
        for i in range(self.s):
            for j in range(self.s):
                similarity += (self.canvas[i][j] - self.reference[i][j])**2
        similarity = similarity/(self.s**2)
        
        reward = self.lastSim - similarity
        self.lastSim = similarity

        return reward

    def move_isLegal(self, action):
        if action[0] > len(self.canvas[0])-2 or action[0] < 1:
            return False
        if action[1] > len(self.canvas)-2 or action[1] < 1:
            return False
        return True

    def reset(self):
        self.curRef += 1
        self.reference = self.referenceData[self.curRef]
        self.canvas = np.zeros((self.s, self.s))
        self.set_agentPos((random.randint(1, self.s-2), random.randint(1, self.s-2)))
        self.reward()
        return np.array([self.reference, self.canvas, self.distmap, self.colmap]), np.array([self.ref_patch, self.canvas_patch])
    
    def render(self, mode="None"):
        if mode == "Compare":
            rendCanv = self.canvas.copy().reshape(self.s**2,)
            rendRef = self.reference.copy().reshape(self.s**2,)
            for index, item in enumerate(zip(rendCanv, rendRef)):
                i, e = item
                if i == e and i == 1:
                    rendCanv[index] = 255
                    rendRef[index] = 255
                elif i == 1:
                    rendCanv[index] = 50
                elif e == 1:
                    rendRef[index] = 50
            
            # Original image
            self.fig.add_subplot(2, 2, 1)
            plt.imshow(rendRef.reshape(28, 28), cmap='gray', label='Original', vmin=0, vmax=255)
            plt.axis("off")
            plt.title("Original")
            
            # AI Generated Image
            self.fig.add_subplot(2, 2, 2) 
            plt.imshow(rendCanv.reshape(28, 28), cmap='gray', label='AI Canvas', vmin=0, vmax=255)
            plt.axis("off")
            plt.title("AI Canvas")
            
            plt.pause(0.01)

        else:
            plt.imshow(self.canvas, interpolation='none', cmap='gray')
            plt.pause(0.01)
    

#draws Line directly on bitmap to save convert
def drawline(setpos, pos, canvas):
    weight = 1
    dx = pos[0] - setpos[0] 
    dy = pos[1] - setpos[1] 
    linePix = []

    if dx == 0 and dy == 0:
        return canvas

    if abs(dx) > abs(dy):
        inc = int(ma.copysign(1,dx))
        for i in range(0, dx+inc, inc):
            linePix.append([setpos[0]+i, 0])
        
        step = dy/(abs(dx)+1)
        sign = int(ma.copysign(1,dy))
        move = 0
        res = 0
        for i in range((abs(dx)+1)):
            res += step
            if sign*res >= 0.5:
                move += sign
                res -= sign
            linePix[i][1] = setpos[1]+move

        for pix in linePix:
            canvas[pix[1]-weight][pix[0]] = 1
            canvas[pix[1]+weight][pix[0]] = 1
            canvas[pix[1]][pix[0]] = 1
    else:
        inc = int(ma.copysign(1,dy))
        for i in range(0, dy+inc, inc):
            linePix.append([0, setpos[1]+i])
        
        step = dx/(abs(dy)+1)
        sign = int(ma.copysign(1,dx))
        move = 0
        res = 0
        for i in range((abs(dy)+1)):
            res += step
            if sign*res >= 0.5:
                move += sign
                res -= sign
            linePix[i][0] = setpos[0]+move

        for pix in linePix:
            canvas[pix[1]][pix[0]+weight] = 1
            canvas[pix[1]][pix[0]-weight] = 1
            canvas[pix[1]][pix[0]] = 1

    return canvas