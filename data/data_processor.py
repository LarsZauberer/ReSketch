import pandas as pd
import numpy as np
import math
import json


class Data_Processer():
    train_test_split = 0.75

    def mnist(self):
        ref_data = pd.read_csv("data/reference_data/mnist.csv")

        ref_train = ref_data.head( math.floor(ref_data.shape[0] * self.train_test_split))
        ref_test = ref_data.tail( math.floor(ref_data.shape[0] * (1-self.train_test_split)))
    
        train_data = self.mnist_json_formatter(ref_train)
        test_data = self.mnist_json_formatter(ref_test)

        with open("data/processed_data/mnist_train.json", "w") as f:
            json.dump(train_data, f)

        with open("data/processed_data/mnist_test.json", "w") as f:
            json.dump(test_data, f)

    def mnist_json_formatter(self, ref_data, grayscales=False):
        pro_data = [[] for _ in range(10)]
        #10 categories
        for i in range(10):
            #filter by images of given number
            num = ref_data.loc[ref_data["label"] == i]
            num = num.drop("label", axis=1)
        
            #append all images as list
            lis = []
            for j in  range(num.shape[0]):
                img = num.iloc[j].tolist()
                #remove grayscales if wanted
                if not grayscales:
                    for k, pix in enumerate(img):
                        if pix > 0: img[k] = 1
                lis.append(img)
            
            pro_data[i] = lis
        return pro_data

    

        

        


    def emnist():
        pass

    def quickDraw():
        pass



""" def generate_json():
    
    label_Data = pd.read_csv("datascience/mnist-before-after/labels.csv")
    ref_Data = pd.read_csv("datascience/mnist-before-after/train.csv")
    label_Data = label_Data.drop('Unnamed: 0', axis=1)
    ref_Data = ref_Data.drop('label', axis=1)
    

    data = [[] for _ in range(10)]


    ref_Data["label"] = label_Data["label"]

    ref_Data = ref_Data.tail(6000)

    

    for i in range(10):
        num = ref_Data.loc[ref_Data["label"] == i]
        num = num.drop("label", axis=1)
    
        lis = []
        for j in  range(num.shape[0]):
            lis.append( num.iloc[j].tolist() )
        
        
        data[i] = lis """

if __name__ == "__main__":
    dp = Data_Processer()
    dp.mnist()
    