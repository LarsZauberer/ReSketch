from matplotlib.font_manager import json_dump
import matplotlib.pyplot as mp
import json


class Learn_Plotter():
    def __init__(self, path : str = "src/result_stats/plotlearn_data.json"):
        self.path = path
        self.episodes = []
        self.averages = []

    def update_plot(self, episode, average):
        self.episodes.append(episode)
        self.averages.append(average)

    def save_plot(self):
        with open(self.path, "w") as f:
            json.dump([self.episodes, self.averages], f)

    def plot(self):
        data = []
        with open(self.path, "r") as p:
            data = json.load(p)
        mp.plot(data[0], data[1])
        mp.show()


    


class Traintest_Learn_Plotter():
    def __init__(self, path : str = "src/result_stats/traintest_plotlearn_data.json"):
        self.path = path
        self.episodes = []
        self.train_avgs = []
        self.test_avgs = []

    def update_plot(self, episode, train_score, test_score):
        self.episodes.append(episode)
        self.train_avgs.append(train_score)
        self.test_avgs.append(test_score)

    def save_plot(self):
        with open(self.path, "w") as f:
            json.dump([self.episodes, self.train_avgs, self.test_avgs], f)


    def plot(self):
        data = []
        with open("src/result_stats/plotlearn_data.json", "r") as p:
            data = json.load(p)
        ep, tr, te = data
    
        mp.plot(ep, tr, color="b")
        mp.plot(ep, te, color="r")
        mp.show()



if __name__ == '__main__':
    lp = Learn_Plotter(path="src/result_stats/plotlearn_data.json")
    lp.plot()



