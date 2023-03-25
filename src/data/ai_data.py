import pandas as pd
import json
import numpy as np

class AI_Data():
    def __init__(self, dataset : str = "mnist_train", motive=0):
        self.dataset = dataset.split("_")[0]
        self.motive = motive

        with open(f"src/data/processed_data/{dataset}.json", "r") as f:
            self.ref_data = json.load(f)
            
        #processed data
        self.pro_data = []
        self.labeled_pro_data = []
       

    def sample(self, number : int):
        sampled = []
        label_sampled = []

        num = int(number/len(self.ref_data))
        #for every category
        for n in range(len(self.ref_data)):
            ind = np.random.choice(len(self.ref_data[n]), num, replace=False)

            for i in ind:
                arr = np.array(self.ref_data[n][i]).reshape(28,28)
                sampled.append(arr)
                label_sampled.append( (n, arr) )


        self.pro_data = sampled
        self.labeled_pro_data = label_sampled
        self.shuffle()

    def sample_by_category(self, category, number):
        sampled = []
        label_sampled = []

        ind = np.random.choice(len(self.ref_data[category]), number)

        for i in ind:
            arr = np.array(self.ref_data[category][i]).reshape(28,28)
            sampled.append(arr)
            label_sampled.append( (category, arr) )

        self.pro_data = sampled
        self.labeled_pro_data = label_sampled
        self.shuffle()

            
    def shuffle(self):
        """ 
        changes the order in the sampled data randomly

        :return: None. It updates the pro_data variable
        :rtype: None
        """
        ind = np.arange(len(self.pro_data))
        np.random.shuffle(ind)
        shuffled = []
        label_shuffled = []
        for i in ind:
            shuffled.append(self.pro_data[i])
            label_shuffled.append(self.labeled_pro_data[i])

        self.pro_data = shuffled
        self.labeled_pro_data = label_shuffled



    








        









    
    




