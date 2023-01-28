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
            path = [
                    "src/data/quickdraw/full_numpy_bitmap_anvil.npy",
                    "src/data/quickdraw/full_numpy_bitmap_apple.npy",
                    "src/data/quickdraw/full_numpy_bitmap_broom.npy",
                    "src/data/quickdraw/full_numpy_bitmap_bucket.npy",
                    "src/data/quickdraw/full_numpy_bitmap_bulldozer.npy",
                    "src/data/quickdraw/full_numpy_bitmap_clock.npy",
                    "src/data/quickdraw/full_numpy_bitmap_cloud.npy",
                    "src/data/quickdraw/full_numpy_bitmap_computer.npy",
                    "src/data/quickdraw/full_numpy_bitmap_eye.npy",
                    "src/data/quickdraw/full_numpy_bitmap_flower.npy",
                   ]
        else:
            path = "src/data/json/mnist_test_data.json"
    

        self.ref_data = []
        if json_import:
            sorted_data = [] 
            with open(path, "r") as f:
                sorted_data = json.load(f)
            self.ref_data = sorted_data
        else:
            self.ref_data = np.array(self.ref_data)
            for i in path:
                data = np.load(i, encoding="latin1", allow_pickle=True)
                # Use only 100'000 instances
                data = data[:100000]
                self.ref_data = np.append(self.ref_data, data)
            
            # Reshape the array so it's not flattend anymore due to the append function
            self.ref_data = self.ref_data.reshape(len(path), 100000, 784)
            
        
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
            for n in range(self.ref_data.shape[0]):
                ind = np.random.choice(self.ref_data.shape[1], number, replace=False)
                for i in ind:
                    arr = np.array(self.ref_data[n][i]).reshape(28, 28, order=reshape_ordering)
                    
                    # Remove the grayscale
                    arr = np.where(arr > 0, 1, arr)
                    
                    sampled.append(arr)
        else:
            sampled = []
            num = int(number/len(self.ref_data))
            for n in range(len(self.ref_data)):
                ind = np.random.choice(len(self.ref_data[n]), num, replace=False)
                

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



    








        









    
    




