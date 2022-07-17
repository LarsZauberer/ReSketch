from matplotlib.font_manager import json_dump
import matplotlib.pyplot as mp
import json

def train():
    data = []
    with open("src/result_stats/plotlearn_data.json", "r") as p:
        data = json.load(p)


    print(data)

    mp.plot(data[0], data[1])
    mp.show()

def test_train():
    data = []
    with open("src/result_stats/plotlearn_data.json", "r") as p:
        data = json.load(p)
    
    train_data = data["train"]
    test_data = data["test"]

    mp.plot(train_data[0], train_data[1], color="b")
    mp.plot(test_data[0], test_data[1], color="r")

    mp.show()
    

train()

