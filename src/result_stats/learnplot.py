from matplotlib.font_manager import json_dump
import matplotlib.pyplot as mp
import json

data = []
with open("src/result_stats/plotlearn_data.json", "r") as p:
    data = json.load(p)


print(data)

mp.plot(data[0], data[1])
mp.show()
