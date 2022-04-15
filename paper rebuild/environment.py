import os
import random
import numpy as np
import math as ma


class ShapeDraw(object):
    def __init__(self, sidelength, patchsize, referenceData):
        self.s = sidelength
        self.p = patchsize #sidelength of patch (local Input). must be odd
        
        #Input gloabal stream
        self.referenceData = referenceData
        self.canvas = np.full((self.s, self.s), 1)
        self.distmap = np.zeros((self.s, self.s))
        self.colmap = np.zeros((self.s, self.s))
        
        #Input local stream
        self.ref_patch = np.zeros((self.p, self.p))
        self.canvas_patch = np.zeros((self.p, self.p))

        #possible outputs
        self.actionspace = range(self.p*self.p*2)

        #initializes rest
        self.reference = referenceData[0]
        self.curRef = 0
        self.agentPos
        self.set_agentPos(random.randrange(0, self.s), random.randrange(0, self.s))
        

        self.lastReward
        self.reward
        
    def set_agentPos(self, x, y):
        self.agentPos = (x, y)
        self.update_distmap()

    def update_distmap(self):
        x0 = self.agentPos[1]
        y0 = self.agentPos[2]
        for y in range(self.s):
            for x in range(self.s):
                dist =  ma.sqrt((x-x0)**2 + (y-y0)**2)/self.s
                self.distmap[y][x] = dist

    def update_colmap(self, state):
        for y in range(self.s):
            for x in range(self.s):
                self.colmap[y][x] = state

    def update_patch(self, reference):
        patchX = self.agentPos[0]-(self.p-1)/2
        patchY = self.agentPos[1]-(self.p-1)/2
        for y in range(self.p):
            for x in range(self.p):
                self.ref_patch[y][x] = reference[patchY+y][patchX+x]
                self.canvas_patch[y][x] = self.canvas[patchY+y][patchX+x]

    def reset(self):
        self.curRef += 1
        self.reference = self.referenceData[self.curRef]
        self.canvas = np.full((self.s, self.s), 1)
        self.update_colmap(0)
        self.set_agentPos(random.randrange(0, self.s), random.randrange(0, self.s))

    


    

        






   

arr = np.zeros(5, dtype=1)

print(arr)
    

if __name__ == "__main__":

    """
    ToDo:
        - load training data
        - convert to bitmap?
        - shuffle training data
        - 
    """


        
    




