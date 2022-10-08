from matplotlib.font_manager import json_dump
import matplotlib.pyplot as mp
import numpy as np
import json


class Learn_Plotter():
    def __init__(self, path : str = "src/result_stats/plotlearn_data.json"):
        self.path = path
        self.episodes = []
        self.averages = []

    def update_plot(self, episode: int, average: float):
        """
        Appends new information for a future plot

        :param episode: the episode at which the function is called
        :type episode: int
        :param average: the data to be plotted, corresponding to the episode
        :type average: float

        :rtype: None
        """

        self.episodes.append(episode)
        self.averages.append(average)

    def save_plot(self):
        """
        Saves the collected information into a json

        :rtype: None
        """
        with open(self.path, "w") as f:
            json.dump([self.episodes, self.averages], f)

    def plot(self):
        data = []
        with open(self.path, "r") as p:
            data = json.load(p)

        
        fig, ax = mp.subplots(dpi=300)

        ax.grid()
        ax.plot(data[0], data[1], color="cornflowerblue") # "navy", "cornflowerblue"
       
        ax.set_title("Physik-Basis", fontsize=16)
        ax.set_xlabel("Episode", fontsize=13)
        ax.set_ylabel("akkumulierter Reward", fontsize=13)
        

        fig.savefig(f"src_physics/result_stats/learnplot.png", bbox_inches='tight')
        
        mp.show()



    


class Traintest_Learn_Plotter():
    def __init__(self, path : str = "src/result_stats/traintest_plotlearn_data.json"):
        self.path = path
        self.episodes = []
        self.train_avgs = []
        self.test_avgs = []

    def update_plot(self, episode: int, train_score: float, test_score: float):
        """
        Appends new information for a future plot

        :param episode: the episode at which the function is called
        :type episode: int
        :param train_score: The score of the training, corresponding to the episode
        :type train_score: float
        :param test_score: The score of the testing, corresponding to the episode
        :type test_score: float

        :rtype: None
        """
        self.episodes.append(episode)
        self.train_avgs.append(train_score)
        self.test_avgs.append(test_score)

    def save_plot(self):
        """
        Saves the collected information into a json

        :rtype: None
        """
        with open(self.path, "w") as f:
            json.dump([self.episodes, self.train_avgs, self.test_avgs], f)


    def plot(self):
        """
        Plots the collected data
        The training data is portrayed in blue
        The testing data is portrayed in red

        :rtype: None
        """
        data = []
        with open("src/result_stats/plotlearn_data.json", "r") as p:
            data = json.load(p)
        ep, tr, te = data
    
        mp.plot(ep, tr, color="b")
        mp.plot(ep, te, color="r")
        mp.show()



if __name__ == '__main__':
    lp = Learn_Plotter(path="src_physics/result_stats/plotlearn_data.json")
    lp.plot()



