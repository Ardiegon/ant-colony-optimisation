import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import time
from project.src.thread_with_return import ThreadWithReturn
from random import uniform
from project.src.grouped_data import GroupedData

class Model:
    def __init__(self, _n_ants, _backpack_size, _pheromone_weight, _distance_weight):
        self.points, self.groups, self.groups_neighbours, self.start_point, self.start_group = (None, None, None, None, None)
        self.n_ants = _n_ants
        self.pheromone_weight = _pheromone_weight
        self.backpack_size = _backpack_size
        self.distance_weight = _distance_weight

    def load_data(self, gdata: GroupedData):
        self.points, self.groups, self.groups_neighbours, self.start_point, self.start_group = gdata.get_data()

    def show_routes(self, groups, routes = None):
        def randomRGB():
            return (uniform(0.05,0.8),uniform(0.05,0.8),uniform(0.05,0.8))
        for group in groups:
            clr = randomRGB()
            for point in group:
                if point[0] == 0:
                    plt.scatter(point[1], point[2], marker = "X", color = (0.0,0.0,0.0))
                    plt.annotate("Start", (point[1], point[2]))
                else:
                    plt.scatter(point[1], point[2], color = clr)
        plt.show()

if __name__ == "__main__":
    model = Model(5, 100, 1, 1)
    model.load_data(GroupedData())
    model.show_routes(model.groups)

