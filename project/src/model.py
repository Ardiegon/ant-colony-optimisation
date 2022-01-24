import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import time
from thread_with_return import ThreadWithReturn
from random import uniform
from grouped_data import GroupedData

class Model:
    def __init__(self, _n_ants, _backpack_size, _pheromone_weight, _distance_weight):
        self.points, self.groups, self.groups_neighbours, self.start_point, self.start_group = (None, None, None, None, None)
        self.n_ants = _n_ants
        self.pheromone_weight = _pheromone_weight
        self.backpack_size = _backpack_size
        self.distance_weight = _distance_weight

    def load_data(self, gdata: GroupedData):
        self.points, self.groups, self.groups_neighbours, self.start_point, self.start_group = gdata.get_data()

    def show_routes(self, routes = None):
        def randomRGB(min, max):
            return (uniform(min,max),uniform(min,max),uniform(min,max))
        def connectpoints(p1, p2, clr):
            x1, x2 = self.points[p1][1], self.points[p2][1]
            y1, y2 = self.points[p1][2], self.points[p2][2]
            plt.plot([x1, x2], [y1, y2], 'k-', color = clr)

        groups = self.groups
        for group in groups:
            clr = randomRGB(0.0, 0.8)
            for point in group:
                if point[0] == 0:
                    plt.scatter(point[1], point[2], marker = "X", color = (0.0,0.0,0.0))
                    plt.annotate("Start", (point[1], point[2]))
                else:
                    plt.scatter(point[1], point[2], color = clr)
        for route in routes:
            clr = randomRGB(0.5,0.8)
            for i in range(len(route)-1):
                connectpoints(route[i], route[i+1], clr)
        plt.show()

if __name__ == "__main__":
    model = Model(5, 100, 1, 1)
    model.load_data(GroupedData())
    model.show_routes([[0,1,2,3,0],[0,4,5,6,7,8,0]])

