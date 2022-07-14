import pandas as pd
import json
import numpy as np


def generate_json():
    label_Data = pd.read_csv("src/data/labels.csv")
    ref_Data = pd.read_csv("src/data/ref_Data.csv")
    ref_Data = ref_Data.drop('Unnamed: 0', axis=1)
    label_Data = label_Data.drop('Unnamed: 0', axis=1)

    data = [[] for _ in range(10)]


    ref_Data["label"] = label_Data["label"]



    for i in range(10):
        num = ref_Data.loc[ref_Data["label"] == i]
        num = num.drop("label", axis=1)
    
        lis = []
        for j in  range(num.shape[0]):
            lis.append( num.iloc[j].tolist() )
        
        
        data[i] = lis

    print(data[0][0])

    with open("src/data/ref_Data.json", "w") as f:
        json.dump(data, f)



def sample_data(data, number):
    """  """
    sampled = []
    num = int(number/10)

    for n in range(10):
        ind = np.random.choice(len(data[n]), num)
        for i in ind:
            sampled.append(np.array(data[n][i]).reshape(28,28))
        
    return sampled


def shuffle_data(data: list):
    ind = np.arange(len(data))
    np.random.shuffle(ind)
    d = []
    for i in ind:
        d.append(data[i])

    return d

        









    
    




