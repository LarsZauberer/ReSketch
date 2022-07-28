import random
import numpy as np
import math as ma
import matplotlib.pyplot as plt
from agent_modules.physics import Physic_Engine


class ShapeDraw(object):
    def __init__(self, sidelength: int, patchsize: int, referenceData: np.array, n_actions : int,  do_render : bool = True, max_action_strength : int = 2, friction: float = 0.2, vel_1: float = 0.8, vel_2: float = 1.2):
        self.s = sidelength
        self.p = patchsize  # sidelength of patch (local Input). must be odd
        self.max_action_strength = max_action_strength

        # Input gloabal stream
        self.referenceData = referenceData
        self.canvas = np.zeros((self.s, self.s))
        self.distmap = np.zeros((self.s, self.s))
        self.colmap = np.zeros((self.s, self.s))

        # Input local stream
        self.ref_patch = np.zeros((self.p, self.p))
        self.canvas_patch = np.zeros((self.p, self.p))

        # Physics
        self.phy_settings = {"friction": friction, "action_scale": 1.0}
        self.phy = Physic_Engine(**self.phy_settings)


        # possible outputs
        # For each pixel, is an action option (location of that pixel)
        self.n_actions = n_actions
        self.actions = [(0,0)]
        for i in range(8):
            angle = ma.pi/4*i
            self.actions.append( (float('%.2f' % (ma.cos(angle)*vel_1)) , float('%.2f' % (ma.sin(angle)*vel_1))) )
        for i in range(12):
            angle = ma.pi/6*i
            self.actions.append( (float('%.2f' % (ma.cos(angle)*vel_2)) , float('%.2f' % (ma.sin(angle)*vel_2))) )

        print(self.actions)

        # initializes rest
        self.lastSim = 0  # Last similarity between reference and canvas
        self.maxScore = 1 # Maximum Reward (changes with reference Image) = base Similarity between empty canvas and reference        
        self.reference = referenceData[0] # Pick just the first image of the reference data in the first initialization
        self.curRef = -1 #current reference image, counter that increments with every episode
        self.isDrawing = 0 # 0 = not Drawing, 1 = Drawing (not bool because NN)
        self.agentPos = [0, 0] # initialize agent position to top left corner of the image
        self.set_agentPos([random.randint(1, self.s-2),
                          random.randrange(1, self.s-2)], weight=1)  # Set a random start location for the agent (but with one pixel margin)

        if do_render: self.fig, self.axs = plt.subplots(1, 2, figsize=[10,7])
        
    def step(self, agent_action: int):
        """
        step execute a timestep. Creates a new canvas state in account of the action
        index input

        :param agent_action: the index of the action to be executed
        :type agent_action: int
        :return: the data of the environment after the timestep (observation for the agent). It is containing an np array with
            `reference`, `canvas`, `distmap` and `colmap` another np array with
            `ref_patch` and `canvas_patch` and an int with the `reward`
        :rtype: tuple
        """
        
        x, y, weight = self.translate_action(agent_action)
        action = self.phy.calc_new_pos(self.agentPos, [x, y], update_velocity=True)

        penalty = 0
        if self.move_isLegal(action):
            self.set_agentPos(action, weight)
        else:
            # Give a penalty for an illegal move
            penalty = -0.05
            self.phy.velocity = [0, 0]

        # Calculate the reward for the action in this turn
        # The reward can be 0 because it is gaining the reward only for new pixels
        reward = self.reward() if self.isDrawing else 0.0
        reward += penalty

        # Ending the timestep
        return np.array([self.reference, self.canvas, self.distmap, self.colmap]), np.array([self.ref_patch, self.canvas_patch]), reward

    def illegal_actions(self, illegal_list : np.array):
        for action in range(self.n_actions):
            x, y, _ = self.translate_action(action)
            act = self.phy.calc_new_pos(self.agentPos, [x, y], update_velocity = False)
            illegal_list[action] = int(not self.move_isLegal(act))
        
        

        return illegal_list

    def move_isLegal(self, action):
        """
        move_isLegal Check if an action is legel.

        :param action: The action to validate
        :type action: list
        :return: Wether it is legal or not
        :rtype: bool
        """
        if action[0] > len(self.canvas[0])-1 or action[0] < 0:
            return False
        elif action[1] > len(self.canvas)-1 or action[1] < 0:
            return False
        return True

    def translate_action(self, action):
        """
        action_to_direction Convert the action index of the agent to a direction and return the strength of the action. Sets also the isDrawing variable.

        :param action: the index of the action to be executed
        :type action: int
        :return: x and y delta of the action
        :rtype: tuple
        """
        self.isDrawing = 1
        weight = int(action/(self.n_actions/self.max_action_strength))
        action = action % int(self.n_actions/self.max_action_strength)
        if weight == 0:
            self.isDrawing = 0

        x, y = self.actions[action]
        return x, y, weight


    def reward(self):
        """
        reward Calculate the reward based on gained similarity and length of step

        :return: The reward value
        :rtype: float
        """
        reward = 0
        similarity = 0
        for i in range(self.s):
            for j in range(self.s):
                similarity += (self.canvas[i][j] - self.reference[i][j])**2
                
        similarity /= self.maxScore

        # Only use the newly found similar pixels for the reward
        reward = (self.lastSim - similarity) 
        
        if self.maxScore == 1:
            self.maxScore = similarity
            self.lastSim = 1
        else:
            self.lastSim = similarity

        return reward


    def set_agentPos(self, pos: list, weight: int):
        """
        set_agentPos Sets the agent to a new position.

        :param pos: coordinates of the new position
        :type pos: list
        """
        if self.isDrawing:
            self.canvas = drawline(self.agentPos, pos, self.canvas, weight=weight)
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
        vel_pos = self.phy.calc_new_pos(self.agentPos, [0,0], update_velocity=False)

        # Get start locations of the patch
        patchX = int(vel_pos[0]-(self.p-1)/2)
        patchY = int(vel_pos[1]-(self.p-1)/2)
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
        self.reference = self.referenceData[self.curRef]
        
        
        # Reset canvas 
        self.canvas = np.zeros((self.s, self.s))

        #reset Agent position
        self.isDrawing = 0
        self.phy.velocity = [.0, .0]
        self.set_agentPos((random.randint(1, self.s-2),
                          random.randint(1, self.s-2)), weight = 1)

        
        # Reset the reward by rerunning it on an empty canvas
        # This should clear the last similarity variable
        self.maxScore = 1
        self.reward()

        return np.array([self.reference, self.canvas, self.distmap, self.colmap]), np.array([self.ref_patch, self.canvas_patch])

    def render(self, mode="None", realtime=False):
        """
        render Renders the current canvas state in matplotlib to visualize the
        state. It has two modes. Compare (value=Compare) and default
        (value=None). In the compare mode it shows the reference and the
        rendered image and visualizes which pixels are the same. In the default
        option it renders only the canvas state without any comparements to the
        reference image.

        :param mode: The mode how to render the canvas, defaults to "None"
        :type mode: str, optional
        :param realtime: If the canvas should render in realtime, defaults to False
        :type realtime: bool, optional
        """
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
            rendRef = rendRef.reshape(28,28)
            self.axs[0].imshow(rendRef.reshape(28, 28), cmap='gray',
                       interpolation = 'none', label='Original', vmin=0, vmax=255)
            self.axs[0].set_title('Original')
            self.axs[0].axis("off")
            
            
            # AI Generated Image
            rendCanv = rendCanv.reshape(28, 28)
            if realtime:
                rendCanv[self.agentPos[1]][self.agentPos[0]] = 150
            self.axs[1].imshow(rendCanv, cmap='gray', interpolation='none',
                       label='AI Canvas', vmin=0, vmax=255)
            self.axs[1].set_title('AI Canvas')
            self.axs[1].axis('off')

            plt.pause(0.005)
            plt.pause(0.005)
        else:
            plt.imshow(self.canvas, interpolation='none', cmap='gray')
            plt.pause(0.01)


###############################################################################

# Utility Functions

###############################################################################


# draws Line directly on bitmap to save convert
def drawline(setpos, pos, canvas, weight):
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
    weight = weight-1  # The weight of a stroke
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
                canvas[pix[1]+weight-y][pix[0]] = 1
                
                    
    
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
                canvas[pix[1]][pix[0]+weight-x] = 1

    return canvas



