import pandas as pd
import json
import numpy as np

class AI_Data():
    def __init__(self, dataset : str = "mnist_train"):
        self.dataset = dataset 

        with open(f"data/processed_data/{dataset}", "r") as f:
            self.ref_data = json.load(f)
            
        #processed data
        self.pro_data = []
       

    def sample(self, number : int):
        sampled = []
        num = int(number/len(self.ref_data))
        #for every category
        for n in range(self.ref_data):
            ind = np.random.choice(len(self.ref_data[n]), num, replace=False)

            for i in ind:
                arr = np.array(self.ref_data[n][i]).reshape(28,28)
                sampled.append(arr)
            
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



    








        









    
    




