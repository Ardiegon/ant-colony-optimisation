import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import time
from project.src.thread_with_return import ThreadWithReturn
from random import uniform
from math import sqrt
from queue import Queue
from project.src.grouped_data import GroupedData


class Model:
    def __init__(self, _n_ants, _backpack_size, _pheromone_weight, _distance_weight):
        self.points, self.groups, self.groups_neighbours, self.start_point, self.start_group, self.picked_points= (None, None, None, None, None, None)
        self.n_ants = _n_ants
        self.pheromone_weight = _pheromone_weight
        self.backpack_size = _backpack_size
        self.distance_weight = _distance_weight

    def load_data(self, gdata: GroupedData):
        self.points, self.groups, self.groups_neighbours, self.start_point, self.start_group = gdata.get_data()
        self.picked_points = np.zeros(len(self.points))

    def show_routes(self, routes = None):
        def randomRGB(min, max):
            return (uniform(min,max),uniform(min,max),uniform(min,max))
        def connectpoints(p1, p2, clr):
            x1, x2 = self.points[p1][1], self.points[p2][1]
            y1, y2 = self.points[p1][2], self.points[p2][2]
            plt.plot([x1, x2], [y1, y2], color = clr)

        groups = self.groups
        for iter, group in enumerate(groups):
            print(f"plottig group {iter}")
            clr = randomRGB(0.0, 0.8)
            for point in group:
                if point[0] == 0:
                    plt.scatter(point[1], point[2], marker = "X", color = (0.0,0.0,0.0))
                    plt.annotate("Start", (point[1], point[2]))
                else:
                    plt.scatter(point[1], point[2], color = clr)
        for route in routes:
            clr = randomRGB(0.3,0.8)
            for i in range(len(route)-1):
                connectpoints(route[i], route[i+1], clr)
        plt.show()

    def plot_points(self, points):
        for point in points:
            if point[0] == 0:
                plt.annotate("Start", (point[1], point[2]))
                plt.scatter(point[1], point[2], marker = "X", color = (0.0,0.0,0.0))
            else:
                plt.scatter(point[1], point[2], color= (0.0,0.0,1.0))
        plt.show()

    def pick_points(self, min_points):
        using_points = []
        group_q = Queue()
        group_q.put(self.start_group)
        group_history = [self.start_group]
        print(f"Chosing points, minimum {min_points}")
        while len(using_points) < min_points:
            if group_q.empty():
                print(f"Only {len(using_points)} left")
                break
            curr_group = int(group_q.get())
            for g in self.groups_neighbours[curr_group]:
                if g not in group_history:
                    group_q.put(g)
                    group_history.append(g)
            for point in self.groups[curr_group]:
                if self.picked_points[int(point[0])] ==0:
                    using_points.append(point)
            print(f"\r\taccumulated: {len(using_points)}", end="")
        print(f"\r\taccumulated: {len(using_points)}, finished")
        return using_points

    def calc_sq_distances(self, chosen_points):
        distances = {}
        n_all = len(chosen_points)
        print(f"Calculating distances of {len(chosen_points)} squared...")
        for i in range(len(chosen_points)):
            for j in range(i+1, len(chosen_points)):
                distances[(i, j)] = (chosen_points[i][1] - chosen_points[j][1])**2+(chosen_points[i][2] - chosen_points[j][2])**2
                print(f"\r\tcalculated {(int(((i+1)*n_all)-((i+1)**2-(i+1))/2+i+1)/int((n_all*n_all-n_all)/2))*100 :.2f}%", end="")
        print(f"\r\tcalculated 100%, finished")
        return distances


    def ant_search_route(self, searching_point_ids, distances):  #TODO ścieżka mrówki
        route = []

        return route


if __name__ == "__main__":
    size_points = 100000
    model = Model(5, 100, 1, 1)
    model.load_data(GroupedData(n_points=size_points, box_size=10000, group_size= 1000))
    # model.show_routes([[0,1,2,3,0],[0,4,5,6,7,8,0]])
    p = model.pick_points(1000)
    # model.plot_points(p)
    d = model.calc_sq_distances(p)





