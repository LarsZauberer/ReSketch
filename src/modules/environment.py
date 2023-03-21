import numpy as np
import math as ma
import matplotlib.pyplot as plt
from models.Predictor import Predictor



class Environment(object):
    def __init__(self, sidelength: int, patchsize: int, referenceData: np.array, with_stopAction : int = 0,  with_liftpen : bool = False, with_overdraw : bool = False, with_noisy : bool = False, generative : bool = False, dataset="mnist", do_render : bool = True):
        self.s = sidelength
        self.p = patchsize

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
        # For each pixel is an action option (location of that pixel)
        self.n_actions = 2*patchsize*patchsize + 1  # +1 for stop action

        # initializes rest
        self.lastSim = 0  # Last similarity between reference and canvas
        self.maxScore = 1 # Maximum Similarity Reward (changes with reference Image) = base Similarity between empty canvas and reference  
        
        self.label = 0
        if len(self.referenceData[0]) == 2:
            self.label, self.reference = referenceData[0] # Pick just the first image of the reference data in the first initialization
        else:
            self.reference = referenceData[0]
        self.curRef = -1 #current reference image, counter that increments with every episode
        self.show_Reference = True

        self.isDrawing = 0 # 0 = not Drawing, 1 = Drawing (not bool because NN)
        self.agentPos = [0, 0] # initialize agent position to top left corner of the image

        # set start position
        if generative:
            self.set_agentPos([4,4])
        else:
            self.set_agentPos([np.random.choice(range(1, 27)), np.random.choice(range(1, 27))])
        
        # variations
        self.with_stopAction = int(with_stopAction)
        self.with_overdraw = with_overdraw
        self.with_liftpen = with_liftpen
        self.with_noisy = with_noisy
        self.generative = generative

        

        # variation variables
        self.overdrawn_perepisode = 0
        self.score = 0

        self.dataset = dataset
        self.rec_model = Predictor(mnist=True, emnist=True, quickdraw=True)

        # rendering / visualization
        self.renderCanvas = np.zeros((self.s, self.s))
        if do_render: self.fig, self.axs = plt.subplots(1, 2, figsize=[10,7])


    # Convert Action of Agent into observation    
    def step(self, agent_action):
        self.isDrawing = 1
        action = self.translate_action(agent_action)
        self.set_agentPos(action)

        # Calculate the reward for the action in this step. The reward can be 0 because it is gaining the reward only for new pixels
        liftpen = -0.01 if self.with_liftpen else 0
        reward = self.reward() if self.isDrawing else liftpen

        #show the reference or not (important for generative)
        if self.show_Reference:
            shown_ref = self.reference
            shown_patch = self.ref_patch
        else:
            if self.with_noisy:
                # random Pixels
                shown_ref = np.random.random((self.s,self.s))*0.5
                shown_patch = np.random.random((self.p,self.p))*0.5
            else:
                # black image
                shown_ref = np.zeros((self.s,self.s))
                shown_patch = np.zeros((self.p,self.p))

        #return observation
        return np.array([shown_ref, self.canvas, self.distmap, self.colmap]), np.array([shown_patch, self.canvas_patch]), reward


    # Return List of all illegal actions
    def illegal_actions(self, illegal_list : np.array):
        for action in range(self.n_actions):
            if not self.move_isLegal(self.translate_action(action)):
                illegal_list[action] = 1 # 1 == illegal, 0 == legal
        return illegal_list


    # Determine if move is legal
    def move_isLegal(self, action):
        if action == True: # Stop Action
            return bool(self.with_stopAction)
        if action[0] > self.s-1 or action[0] < 0:
            return False
        if action[1] > self.s-1 or action[1] < 0:
            return False
        return True


    # translate agent action into real action
    def translate_action(self, agent_action: int):
        # stop action
        if agent_action == 2*self.p**2:
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


    def reward(self):
        reward = 0
        similarity = 0
        overdrawn = 0  

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
                reward -= 0.01 * (overdrawn - free_overdraw)

        """ if self.with_overdraw:
            # Penality for the overdrawn pixel
            free_overdraw = 3
            if overdrawn - free_overdraw > 0:  # Agent can overdraw 3 pixel for free
                max_penalty_per_pixel = 0.02
                penalty_per_pixel = (max_penalty_per_pixel / 1) * score
                # log.debug(f"Overdrawn penalty: {penalty_per_pixel * overdrawn}")
                reward -= penalty_per_pixel * (overdrawn - free_overdraw) """
            
        return reward

    # Calculate stop reward
    def stop_reward(self, score: float, step: int):
        if self.with_stopAction == 1:
            ACC_THRESHOLD = 0.80
            SPEED = 2.5
            WEIGHT = 0.5
            if score < ACC_THRESHOLD: accuracy_factor = -0.01
            else: accuracy_factor = 0.1
            if SPEED == 0: speed_factor = 1
            else: speed_factor = 1 - (step/64)**SPEED
            return accuracy_factor * speed_factor * WEIGHT
        elif self.with_stopAction == 2:
            if self.dataset == "emnist":
                prediction = self.rec_model.emnist(self.canvas, mode="soft")[self.label] - 0.8
            elif self.dataset == "quickdraw":
                prediction = self.rec_model.quickdraw(self.canvas, mode="soft")[self.label] - 0.8
            else:
                prediction = self.rec_model.mnist(self.canvas, mode="soft")[self.label] - 0.8

            if prediction < 0:
                return prediction*0.5
            else:
                return prediction
        else:
            return 0

    # set agent position
    def set_agentPos(self, pos: list):
        if self.isDrawing:
            self.canvas = drawline(self.agentPos, pos, self.canvas, with_overdrawn=self.with_overdraw)
            self.renderCanvas = drawline(self.agentPos, pos, self.renderCanvas, color=0.25+0.75*self.curStep/64)
        self.agentPos = pos
        self.update_distmap()
        self.update_patch()
        self.update_colmap()

    def update_distmap(self):
        x0 = self.agentPos[0]
        y0 = self.agentPos[1]
        for y in range(self.s):
            for x in range(self.s):
                dist = ma.sqrt((x-x0)**2 + (y-y0)**2)/self.s  # Calculate the distance to that pixel
                self.distmap[y][x] = dist  # Save the distance to the distmap

    def update_colmap(self):
        for y in range(self.s):
            for x in range(self.s):
                self.colmap[y][x] = self.isDrawing

    def update_patch(self):
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
        # Get another reference image
        self.curRef += 1
        self.curRef = self.curRef % len(self.referenceData)
        if len(self.referenceData[0]) == 2:
            self.label, self.reference = self.referenceData[self.curRef]
        else:
            self.reference = self.referenceData[self.curRef]
        
        self.score = 0
        self.overdrawn_perepisode = 0
        self.isDrawing = 0
        
        # Reset canvas and agent position
        self.canvas = np.zeros((self.s, self.s))
        self.renderCanvas = np.zeros((self.s, self.s))
        
        #reset position
        if self.generative:
            self.set_agentPos([4,4])
        else:
            self.set_agentPos([np.random.choice(range(1, 27)), np.random.choice(range(1, 27))])
    

        # Reset the reward by rerunning it on an empty canvas
        # This should clear the last similarity variable
        self.maxScore = 1
        self.reward()

        #show reference or not
        if self.show_Reference:
            shown_ref = self.reference
            shown_patch = self.ref_patch
        else:
            if self.with_noisy:
                shown_ref = np.random.random((self.s,self.s))*0.5
                shown_patch = np.random.random((self.p,self.p))*0.5
            else:
                shown_ref = np.zeros((self.s,self.s))
                shown_patch = np.zeros((self.p,self.p))

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
