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



    def generate_json(self):
        """
        Creates a json file containing the given reference data ordered into
        categories from 0 to 9

        :return: None. Creates a json file
        :rtype: None
        """
        label_Data = pd.read_csv("src/data/labels.csv")
        ref_Data = pd.read_csv("src/data/ref_Data.csv")
        label_Data = label_Data.drop('Unnamed: 0', axis=1)
        ref_Data = ref_Data.drop('Unnamed: 0', axis=1)
        

        data = [[] for _ in range(10)]


        ref_Data["label"] = label_Data["label"]

        ref_Data = ref_Data.tail(6000)

        

        for i in range(10):
            num = ref_Data.loc[ref_Data["label"] == i]
            num = num.drop("label", axis=1)
        
            lis = []
            for j in  range(num.shape[0]):
                lis.append( num.iloc[j].tolist() )
            
            
            data[i] = lis

        with open(self.path, "w") as f:
            json.dump(data, f)








        









    
    




