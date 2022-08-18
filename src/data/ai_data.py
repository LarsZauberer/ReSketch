import pandas as pd
import json
import numpy as np

class AI_Data():
    def __init__(self, dataset : str = "mnist"):
        self.dataset = dataset 
        json_import = True
        if dataset == "emnist":
            path = "src/data/json/emnist_test_data.json"
        elif dataset == "quickdraw":
            json_import = False
            path = "src/data/quickdraw/full_numpy_bitmap_apple.npy"
        else:
            path = "src/data/json/mnist_test_data.json"
    

        if json_import:
            self.ref_data = []
            sorted_data = [] 
            with open(path, "r") as f:
                sorted_data = json.load(f)
            self.ref_data = sorted_data
        else:
            self.ref_data = np.load(path, encoding="latin1", allow_pickle=True)
            
        
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

        if self.dataset == "emnist":
            reshape_ordering = "F"
        else:
            reshape_ordering = "C"

        if self.dataset == "quickdraw":
            sampled = []
            ind = np.random.choice(len(self.ref_data), number, replace=False)
            for i in ind:
                arr = np.array(self.ref_data[i]).reshape(28,28, order=reshape_ordering)
                arr = np.where(arr > 0, 1, arr)
                print(arr)
                sampled.append(arr)
        else:
            sampled = []
            num = int(number/len(self.ref_data))
            for n in range(len(self.ref_data)):
                ind = np.random.choice(len(self.ref_data[n]), num, replace=False)
                for i in ind:
                    arr = np.array(self.ref_data[n][i]).reshape(28,28, order=reshape_ordering)
                    sampled.append(arr)

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



    








        









    
    




