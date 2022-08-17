import pandas as pd
import json
import numpy as np

class AI_Data():
    def __init__(self, path : str = "src/data/train_ref_Data.json"):
        self.path = path
        self.ref_data = []
        
        sorted_data = [] 
        with open(path, "r") as f:
            sorted_data = json.load(f)
        self.ref_data = sorted_data
        
        #processed data
        self.pro_data = []
       
        
    def sample(self, number: int):
        """ 
        Selects random instances in the reference data. 
        Selects an equal amount of data from each category (each number)

        :param number: the number of instances to be contained in the selection
        :type number: int

        :return: None. It updates the pro_data variable
        :rtype: None
        """
        sampled = []
        num = int(number/len(self.ref_data))
        for n in range(len(self.ref_data)):
            ind = np.random.choice(len(self.ref_data[n]), num, replace=False)
            for i in ind:
                sampled.append(np.array(self.ref_data[n][i]).reshape(28,28))

        self.pro_data = sampled

        
    def shuffle(self):
        """ 
        changes the order in the sampled data randomly

        :return: None. It updates the pro_data variable
        :rtype: None
        """
        ind = np.arange(len(self.pro_data))
        np.random.shuffle(ind)
        shuffled = []
        for i in ind:
            shuffled.append(self.pro_data[i])

        self.pro_data = shuffled



    








        









    
    




