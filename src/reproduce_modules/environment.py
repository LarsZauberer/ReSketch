import random
import numpy as np
import math as ma
import matplotlib.pyplot as plt
import logging

from models.Predictor import Predictor

from extras.logger import critical


class Environment(object):
    def __init__(self, sidelength: int, patchsize: int, referenceData: np.array, with_stopAction : bool = False, with_liftpen : bool = False,  with_overdraw : bool = False, generative : bool = False, do_render : bool = True):
        self.s = sidelength
        self.p = patchsize  # sidelength of patch (local Input). must be odd

        # Input gloabal stream
        self.referenceData = referenceData
        self.canvas = np.zeros((self.s, self.s))
        self.distmap = np.zeros((self.s, self.s))
        self.colmap = np.zeros((self.s, self.s))
        self.stepmap = np.zeros((self.s, self.s))
        self.curStep = 0

        # Input local stream
        self.ref_patch = np.zeros((self.p, self.p))
        self.canvas_patch = np.zeros((self.p, self.p))

        # possible outputs
        # For each pixel, is an action option (location of that pixel)
        self.n_actions = 2*patchsize*patchsize + 1  # +1 für die Stop action

        # initializes rest
        self.lastSim = 0  # Last similarity between reference and canvas
        self.lastDirection = [0, 0]
        self.maxScore = 1 # Maximum Reward (changes with reference Image) = base Similarity between empty canvas and reference  
        
        self.label = 0
        if len(self.referenceData[0]) == 2:
            self.label, self.reference = referenceData[0] # Pick just the first image of the reference data in the first initialization
        else:
            self.reference = referenceData[0]

        self.curRef = -1 #current reference image, counter that increments with every episode
        self.isDrawing = 0 # 0 = not Drawing, 1 = Drawing (not bool because NN)
        self.agentPos = [0, 0] # initialize agent position to top left corner of the image

        
        """ if generative:
            self.set_agentPos([4,4])
        else:
            self.set_agentPos([np.random.choice(range(1, 27)), np.random.choice(range(1, 27))]) """
        
        self.set_agentPos([4,4])
            

        # variations
        self.with_stopAction = with_stopAction
        self.with_overdraw = with_overdraw
        self.with_liftpen = with_liftpen
        self.generative = generative

        self.overdrawn_perepisode = 0
        

        self.show_Reference = True
        # Mnist Model -> Recognition of symbol
        self.rec_model = Predictor(mnist=True)

        # rendering / visualization
        self.renderCanvas = np.zeros((self.s, self.s))
        if do_render: self.fig, self.axs = plt.subplots(1, 2, figsize=[10,7])

        
    def step(self, score: float, agent_action):
        """
        step execute a timestep. Creates a new canvas state in account of the action
        index input

        :param agent_action: the index of the action to be executed
        :type agent_action: int
        :param n_step: number of the current step
        :type n_step: int
        :return: the data of the environment after the timestep (observation for the agent). It is containing an np array with
            `reference`, `canvas`, `distmap` and `colmap` another np array with
            `ref_patch` and `canvas_patch` and an int with the `reward`
        :rtype: tuple
        """
        self.isDrawing = 1
        action = self.translate_action(agent_action)
        self.set_agentPos(action)

        # Calculate the reward for the action in this turn. The reward can be 0 because it is gaining the reward only for new pixels
        liftpen = -0.005 if self.with_liftpen else 0
        reward = self.reward(score=score, action=action) if self.isDrawing else liftpen
        # Ending the timestep

        #show the reference or not (important for generative)
        if self.show_Reference:
            shown_ref = self.reference
            shown_patch = self.ref_patch
        else:
            shown_ref = np.zeros((28,28))
            shown_patch = np.zeros((7,7))

        return np.array([shown_ref, self.canvas, self.distmap, self.colmap]), np.array([shown_patch, self.canvas_patch]), reward

    def illegal_actions(self, illegal_list : np.array):
        for action in range(self.n_actions):
            if not self.move_isLegal(self.translate_action(action)):
                illegal_list[action] = 1 # 1 == illegal, 0 == legal
        return illegal_list

    def move_isLegal(self, action):
        """
        move_isLegal Check if an action is legel.

        :param action: The action to validate
        :type action: list
        :return: Wether it is legal or not
        :rtype: bool
        """
        if action == True:
            #llegalize stopAction
            if self.with_stopAction: return True
            else: return False
        if action[0] > len(self.canvas[0])-1 or action[0] < 0:
            return False
        if action[1] > len(self.canvas)-1 or action[1] < 0:
            return False
        return True

    def translate_action(self, agent_action: int):
        if agent_action - 2*self.p**2 == 0:
            return True
        
        action = [0, 0]
        x = agent_action % self.p
        y = agent_action // self.p
        if y >= self.p:
            y -= self.p
            self.isDrawing = 0
        # Calculate the global aim location of the action
        ownpos = (self.p-1)/2
        action = [int(self.agentPos[0]+x-ownpos),
                  int(self.agentPos[1]+y-ownpos)]

        return action

    def reward(self, score: float, action):
        """
        reward Calculate the reward based on gained similarity and length of step

        :return: The reward value
        :rtype: float
        """
        reward = 0
        similarity = 0
        
        overdrawn: int = 0      
        for i in range(self.s):
            for j in range(self.s):
                # Check for overdrawn pixel
                if self.canvas[i][j] > 1:
                    overdrawn += 1
                    self.overdrawn_perepisode += 1
                    self.canvas[i][j] = 1  # Reset to normalized color view
                
                # When the there is a difference -> It is 1 or -1 (squared to become 1). If it's similar -> 0
                similarity += (self.canvas[i][j] - self.reference[i][j])**2
        similarity /= self.maxScore
            
        # Only use the newly found similar pixels for the reward
        reward = self.lastSim - similarity
        if self.maxScore == 1:
            self.maxScore = similarity
            self.lastSim = 1
        else:
            self.lastSim = similarity

        if self.with_overdraw:
            # Penality for the overdrawn pixel
            free_overdraw = 3
            if overdrawn - free_overdraw > 0:  # Agent can overdraw 3 pixel for free
                max_penalty_per_pixel = 0.02
                penalty_per_pixel = (max_penalty_per_pixel / 1) * score
                # log.debug(f"Overdrawn penalty: {penalty_per_pixel * overdrawn}")
                reward -= penalty_per_pixel * (overdrawn - free_overdraw)
            
            """ # Angle between direction vectors
            new_direction = [0, 0]
            new_direction[0] = action[0] - self.agentPos[0]
            new_direction[1] = action[1] - self.agentPos[1]
            length_new_direction = ma.sqrt(new_direction[0]**2 + new_direction[1]**2)
            length_last_direction = ma.sqrt(self.lastDirection[0]**2 + self.lastDirection[1]**2)
            if length_last_direction == 0 or length_new_direction == 0:
                phi = 0
            else:
                phi = ma.acos((new_direction[0]*self.lastDirection[0] + new_direction[1] * self.lastDirection[1])/(length_last_direction * length_new_direction))
            fac = (1/ma.pi) * phi
            reward -= fac * 0.05 """

        return reward

    @critical
    def generative_stop_reward(self: float, step: int) -> float:
        
        """ stop_reward The stop action reward for the agent

        :param score: the accumulated reward of this episode
        :type score: float
        :param step: which step in the action is it (has to be less than 64)
        :type step: int
        :return: The reward for the agent for choosing the stop action
        :rtype: float """
       
        assert step < 64, f"step ({step}) is greater than 64"  # Assert that the step count is less than 64

        
        if self.show_Reference:
            prediction = self.rec_model.mnist(self.canvas)
            if prediction == self.label:
                return 0.1
            else:
                return -0.01
        else:
            prediction = self.rec_model.mnist(self.canvas)
            if prediction == self.label:
                return 0.3
            else:
                return -0.005


    def stop_reward(self, score: float, step: int):
        ACC_THRESHOLD = 0.85
        SPEED = 2.5
        WEIGHT = 0.5

        #accuracy_factor = score-ACC_THRESHOLD


        if score < ACC_THRESHOLD:
            accuracy_factor = -0.01
        else:
            accuracy_factor = 0.1

        if SPEED == 0:
            speed_factor = 1
        else:
            speed_factor = 1 - (step/64)**SPEED

        return accuracy_factor * speed_factor * WEIGHT

        


    def set_agentPos(self, pos: list):
        """
        set_agentPos Sets the agent to a new position.

        :param pos: Koordinates of the new position
        :type pos: list
        """
        self.lastDirection[0] = pos[0] - self.agentPos[0]
        self.lastDirection[1] = pos[1] - self.agentPos[1]
        if self.isDrawing:
            self.canvas = drawline(self.agentPos, pos, self.canvas, with_overdrawn=True)
            self.renderCanvas = drawline(self.agentPos, pos, self.renderCanvas, color=0.25+0.75*self.curStep/64)
        self.agentPos = pos
        self.update_distmap()
        self.update_patch()
        self.update_colmap()

    def update_distmap(self):
        """
        update_distmap Calculate a new distmap
        The distmap tells the agent as a heatmap where he is.
        """
        x0 = self.agentPos[0]
        y0 = self.agentPos[1]
        for y in range(self.s):
            for x in range(self.s):
                dist = ma.sqrt((x-x0)**2 + (y-y0)**2)/self.s  # Calculate the distance to that pixel
                self.distmap[y][x] = dist  # Save the distance to the distmap

    def update_colmap(self):
        """
        update_colmap Calculate a new colmap
        The colmap tells the agent if he is drawing or not
        """
        for y in range(self.s):
            for x in range(self.s):
                self.colmap[y][x] = self.isDrawing

    def update_patch(self):
        """
        update_patch Calculate a local input patch of the agent
        The local patch is a concentrated smaller part of the canvas
        """
        # Get start locations of the patch
        patchX = int(self.agentPos[0]-(self.p-1)/2)
        patchY = int(self.agentPos[1]-(self.p-1)/2)
        for y in range(self.p):
            for x in range(self.p):
                # Check for bounds
                yInd = 0 if patchY + \
                    y >= len(self.reference) or patchY+y < 0 else patchY+y
                xInd = 0 if patchX + \
                    x >= len(self.reference[0]) or patchX+x < 0 else patchX+x

                self.ref_patch[y][x] = self.reference[yInd][xInd]
                self.canvas_patch[y][x] = self.canvas[yInd][xInd]


    def reset(self):
        """
        reset Reset the canvas to the initial state.

        :return: Returns an array with the inital state maps
        :rtype: np.array
        """
        # Get another reference image
        self.curRef += 1
        self.curRef = self.curRef % len(self.referenceData)

        if len(self.referenceData[0]) == 2:
            
            self.label, self.reference = self.referenceData[self.curRef] # Pick just the first image of the reference data in the first initialization
        else:
            self.reference = self.referenceData[self.curRef]
        

        self.overdrawn_perepisode = 0
        self.isDrawing = 0
        
        # Reset canvas and agent position
        self.canvas = np.zeros((self.s, self.s))
        self.renderCanvas = np.zeros((self.s, self.s))
        
        """ if self.generative:
            self.set_agentPos([4,4])
        else:
            self.set_agentPos([np.random.choice(range(1, 27)), np.random.choice(range(1, 27))]) """
        self.set_agentPos([4,4])

        self.lastDirection = [0, 0]

        
        # Reset the reward by rerunning it on an empty canvas
        # This should clear the last similarity variable
        self.maxScore = 1
        self.reward(score=0, action=[0, 0])


        if self.show_Reference:
            shown_ref = self.reference
            shown_patch = self.ref_patch
        else:
            shown_ref = np.zeros((28,28))
            shown_patch = np.zeros((7,7))

        return np.array([shown_ref, self.canvas, self.distmap, self.colmap]), np.array([shown_patch, self.canvas_patch])



    def compare_render(self):
        if self.show_Reference:
            rendRef = self.reference.copy().reshape(self.s**2,)
        else:
            rendRef = np.zeros((28, 28))
        rendCanv = self.canvas.copy().reshape(self.s**2,)
        for index, item in enumerate(zip(rendCanv, rendRef)):
            i, e = item
            if i == e and i == 1:
                rendCanv[index] = 255
                rendRef[index] = 255
            elif i == 1:
                rendCanv[index] = 50
            elif e == 1:
                rendRef[index] = 50
        
        return rendRef, rendCanv

    def gradient_render(self):
        if self.show_Reference:
            rendRef = self.reference.copy().reshape(self.s**2,)
        else:
            rendRef = np.zeros(28*28)
        rendCanv = self.renderCanvas.copy().reshape(self.s**2,)
        
        for i in range(rendRef.shape[0]):
            rendCanv[i] *= 255
            rendRef[i] *= 255

        return rendRef, rendCanv

    def render(self, mode="Gradient"):
        if mode == "Compare":
            rendRef, rendCanv = self.compare_render()
        else:
            rendRef, rendCanv = self.gradient_render()

        self.axs[0].imshow(rendRef.reshape(28, 28), cmap='gray',
                    interpolation = 'none', label='Original', vmin=0, vmax=255)
        self.axs[0].set_title('Original')
        self.axs[0].axis("off")

        self.axs[1].imshow(rendCanv.reshape(28,28), cmap='gray', interpolation='none',
                    label='AI Canvas', vmin=0, vmax=255)
        self.axs[1].set_title('AI Canvas')
        self.axs[1].axis('off')

        plt.pause(0.005)
        plt.pause(0.005)

       
       

            

###############################################################################

# Utility Functions

###############################################################################


# draws Line directly on bitmap to save convert
def drawline(setpos, pos, canvas, with_overdrawn: bool = False, color : float = 1):
    """
    drawline Draw a line on a specified canvas.

    :param setpos: The starting position of the line
    :type setpos: list
    :param pos: The ending position of the line
    :type pos: list
    :param canvas: The canvas on which the line should be drawn
    :type canvas: np.array
    :return: The canvas with the line drawn
    :rtype: np.array
    """
    weight = 1  # The weight of a stroke
    dx = pos[0] - setpos[0]  # delta x
    dy = pos[1] - setpos[1]  # delta y
    linePix = []  # Pixels of the line

    #do not draw if pos == setpos (avoids division by 0)
    if dx == 0 and dy == 0:
        return canvas

    # It is going more on the x-axis
    # Difference important for the diagonal pixel painting.
    if abs(dx) > abs(dy):
        # Get the counting direction for the x-coordinate
        inc = int(ma.copysign(1, dx))
        for i in range(0, dx+inc, inc):
            # Append the x-coordinates
            linePix.append([setpos[0]+i, 0])

        # After how many steps go one up/down
        step = dy/(abs(dx)+1)
        sign = int(ma.copysign(1, dy))
        move = 0
        res = 0
        
        # Move amount to the right
        for i in range((abs(dx)+1)):
            res += step
            
            # Should move upwards
            if sign*res >= 0.5:
                move += sign
                res -= sign
            
            # change the y-position on the drawing pixel
            linePix[i][1] = setpos[1]+move

        # Paint onto the real canvas
        for pix in linePix:
            # Add a weight to the y-position
            # Iterate through the y-axis and add weight
            for y in range(weight*2+1):
                # Check out of bounds
                if pix[1]+weight-y >= len(canvas) or pix[1]+weight-y < 0:
                    continue
                if with_overdrawn:
                    canvas[pix[1]+weight-y][pix[0]] += color
                else:
                    canvas[pix[1]+weight-y][pix[0]] = color
                
                    
    
    # The same procedure for the y-position
    else:
        inc = int(ma.copysign(1, dy))
        for i in range(0, dy+inc, inc):
            linePix.append([0, setpos[1]+i])

        step = dx/(abs(dy)+1)
        sign = int(ma.copysign(1, dx))
        move = 0
        res = 0
        for i in range((abs(dy)+1)):
            res += step
            if sign*res >= 0.5:
                move += sign
                res -= sign
            linePix[i][0] = setpos[0]+move

        for pix in linePix:
            # Iterate through the y-axis and add weight
            for x in range(weight*2+1):
                # Check out of bounds
                if pix[0]+weight-x >= len(canvas) or pix[0]+weight-x < 0:
                    continue
                if with_overdrawn:
                    canvas[pix[1]][pix[0]+weight-x] += color
                else:
                    canvas[pix[1]][pix[0]+weight-x] = color

    return canvas
