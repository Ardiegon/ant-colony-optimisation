import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import time
from project.src.thread_with_return import ThreadWithReturn
from project.src.project_utils import set_seed_for_random
from random import uniform, seed
from math import sqrt
from queue import Queue
from project.src.grouped_data import GroupedData


def get_key(point_1_id, point_2_id):
    return f"{int(min(point_1_id, point_2_id))}/{int(max(point_1_id, point_2_id))}"

class Model:
    def __init__(self, _n_ants, _backpack_size, _pheromone_weight, _distance_weight):
        self.points, self.groups, self.groups_neighbours, self.start_point, self.start_group, self.picked_points, self.added_points, self.used_groups = (None, None, None, None, None, None, None, None)
        self.n_ants = _n_ants
        self.pheromone_weight = _pheromone_weight
        self.backpack_size = _backpack_size
        self.distance_weight = _distance_weight

    def load_data(self, gdata: GroupedData):
        self.points, self.groups, self.groups_neighbours, self.start_point, self.start_group = gdata.get_data()
        print(len(self.points))
        # print(self.points[:5])
        # print(self.groups[:5])
        # print(self.groups_neighbours[:15])
        # print(self.start_point)
        # print(self.start_group)
        # print(self.groups[int(self.start_group)])

        self.picked_points = np.zeros(len(self.points))
        self.added_points = np.zeros(len(self.points))
        self.used_groups = np.zeros(len(self.groups))


    def show_routes(self, routes = None):
        def randomRGB(min, max):
            return (uniform(min,max),uniform(min,max),uniform(min,max))
        def connectpoints(p1, p2, clr):
            x1, x2 = self.points[p1][1], self.points[p2][1]
            y1, y2 = self.points[p1][2], self.points[p2][2]
            plt.plot([x1, x2], [y1, y2], color = clr)

        groups = self.groups
        for iter, group in enumerate(groups):
            clr = randomRGB(0.0, 0.8)
            for point in group:
                if point[0] == 0:
                    plt.scatter(point[1], point[2], marker = "X", color = (0.0,0.0,0.0))
                    plt.annotate("Start", (point[1], point[2]))
                else:
                    plt.scatter(point[1], point[2], color = clr)
                    plt.annotate(f"{int(point[0])} / {int(point[3])}", (point[1], point[2]),  fontsize=5)
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
                plt.scatter(point[1], point[2], color= (1.0,0.0,0.0))
                plt.annotate(f"{int(point[0])} / {int(point[3])}", (point[1], point[2]), fontsize=5)
        plt.show()

    def pick_points(self, min_points):
        using_points = []
        group_q = Queue()
        group_q.put(self.start_group)
        group_history = [self.start_group]
        print(f"Chosing points, minimum {min_points}")
        while len(using_points) < min_points:
            if group_q.empty():
                print(f"\r\tOnly {len(using_points)} left")
                break
            curr_group = int(group_q.get())
            print(f"Current group: {curr_group}")
            print(f"Current group neighbours: {self.groups_neighbours[curr_group]}")
            print(f"All added groups: {group_history}")

            for g in self.groups_neighbours[curr_group]:
                if g not in group_history:
                    group_q.put(g)
                    group_history.append(g)
            for point in self.groups[curr_group]:
                if self.picked_points[int(point[0])] ==0:
                    using_points.append(point)
                    self.added_points[int(point[0])] = 1
            print(f"\r\taccumulated: {len(using_points)}", end="")
            self.used_groups[curr_group] = 1
        print(f"\r\taccumulated: {len(using_points)}, finished")
        return using_points

    def init_sq_distances(self, chosen_points):
        distances = {}
        n_all = len(chosen_points)
        print(f"Initiating distances of {len(chosen_points)} points squared ...")
        for i in range(len(chosen_points)):
            for j in range(i+1, len(chosen_points)):
                key = get_key(chosen_points[i][0], chosen_points[j][0])
                distances[key] = (chosen_points[i][1] - chosen_points[j][1])**2+(chosen_points[i][2] - chosen_points[j][2])**2
            print(f"\r\tcalculated {(int(((i+1)*n_all)-((i+1)**2-(i+1))/2+i+1)/int((n_all*n_all-n_all)/2))*100 :.2f}%", end="")
        print(f"\r\tcalculated 100%, finished")
        return distances

    def init_pheromones(self, chosen_points):
        pheromones = {}
        n_all = len(chosen_points)
        print(f"Initiating pheromones of {len(chosen_points)} points squared ...")
        for i in range(len(chosen_points)):
            for j in range(i+1, len(chosen_points)):
                key = get_key(chosen_points[i][0], chosen_points[j][0])
                pheromones[key] = 0
            print(f"\r\tcalculated {(int(((i+1)*n_all)-((i+1)**2-(i+1))/2+i+1)/int((n_all*n_all-n_all)/2))*100 :.2f}%", end="")
        print(f"\r\tcalculated 100%, finished")
        return pheromones

    def update_state_params(self, c_points, c_distances, c_pheromones, route, min_points):
        for point_id in route:
            self.picked_points[int(point_id)] = 1
        del_ids = []
        old_p_ids = []
        for i, point in enumerate(c_points):
            if int(point[0]) in route and int(point[0]) != 0:
                del_ids.append(i)
            else:
                old_p_ids.append((point[0], point[1], point[2]))
        del_ids.reverse()
        for i in del_ids:
            c_points.pop(i)
        new_p_ids = self.update_points(c_points, min_points)
        self.update_distances_and_pheromones(c_points, c_distances, c_pheromones, old_p_ids, new_p_ids, route)


    def update_points(self, c_points, min_points):
        group_q = Queue()
        group_q.put(self.start_group)
        group_history = [self.start_group]
        new_p_ids = []
        print(f"\nUpdating points, minimum {min_points}, current {len(c_points)}")
        while len(c_points) < min_points:
            if group_q.empty():
                print(f"\r\tOnly {len(c_points)} left")
                break
            curr_group = int(group_q.get())
            print(f"Current group: {curr_group}")
            print(f"Current group neighbours: {self.groups_neighbours[curr_group]}")
            print(f"All added groups: {group_history}")
            for g in self.groups_neighbours[curr_group]:
                if g not in group_history:
                    group_q.put(g)
                    group_history.append(g)
            if self.used_groups[curr_group] == 1:
                continue
            for point in self.groups[curr_group]:
                if self.added_points[int(point[0])] == 0:
                    c_points.append(point)
                    new_p_ids.append((point[0], point[1], point[2]))
                    self.added_points[int(point[0])] = 1
            self.used_groups[curr_group] = 1
            print(f"\r\taccumulated: {len(c_points)}", end="")
        print(f"\r\taccumulated: {len(c_points)}, finished")
        return new_p_ids

    def update_distances_and_pheromones(self, c_points, c_distances, c_pheromones, old_p, new_p, route): #TODO kontynuuj xDD
        print("Updating distances and pheromones...", end = "")
        # print("\nLOG before:")
        # print([x[0] for x in c_points])
        # print([x[0] for x in new_p])
        # print([x[0] for x in old_p])
        # print(c_distances)
        # print(c_pheromones)
        # print(route)
        route_len_without_zero = len(route)-1
        new_p_len = len(new_p)
        # Usuwanie wszystkich wartości między ścieżką i ścieżką
        for r1 in range(1, route_len_without_zero):
            for r2 in range(r1+1, route_len_without_zero):
                c_distances.pop(get_key(route[r1], route[r2]))
                c_pheromones.pop(get_key(route[r1], route[r2]))
        for p1 in old_p:
            # Usuwanie wszystkich wartości między ścieżką i wpisami które zostały w punktach
            for val, p2_id in enumerate(route):
                if p2_id == 0:
                    continue
                c_distances.pop(get_key(p1[0], p2_id))
                c_pheromones.pop(get_key(p1[0], p2_id))
            #dodawanie nowych dystansów i feromonów dla nowych i starych punktów
            for p2 in new_p:
                c_distances[get_key(p1[0], p2[0])] = (p1[1] - p2[1])**2+(p1[2] - p2[2])**2
                c_pheromones[get_key(p1[0], p2[0])] = 0
        for r1 in range(0, new_p_len):
            for r2 in range(r1+1, new_p_len):
                c_distances[get_key(new_p[r1][0], new_p[r2][0])] = (new_p[r1][1] - new_p[r2][1])**2+(new_p[r1][2] - new_p[r2][2])**2
                c_pheromones[get_key(new_p[r1][0], new_p[r2][0])] = 0
        # print("LOG after:")
        # print([x[0] for x in c_points])
        # print([x[0] for x in new_p])
        # print([x[0] for x in old_p])
        # print(c_distances)
        # print(c_pheromones)
        # print(route)
        print("\rUpdating distances and pheromones... finished")


    def ant_trivial(self, c_points, c_distances, c_pheromones):
        route = [0]
        backpack = 0
        i = 1
        while True:
            if i< len(c_points) and backpack + c_points[i][3] < self.backpack_size:
                backpack += c_points[i][3]
                route.append(int(c_points[i][0]))
                i += 1
            else:
                break
        route.append(0)
        return route

    def search_routes(self, n_points, n_ants):
        points_to_use = self.pick_points(n_points)
        sq_distances = self.init_sq_distances(points_to_use)
        pheromones = self.init_pheromones(points_to_use)
        # print("\nSTART:")
        # print([x[0] for x in points_to_use])
        # print(sq_distances)
        # print(pheromones)
        result = []
        while(len(points_to_use)>1):
            routes = []
            ants = []
            for i in range(n_ants):
                ants.append(ThreadWithReturn(target=self.ant_trivial, args=(points_to_use, sq_distances, pheromones)))
                ants[i].start()
            for i in range(n_ants):
                routes.append(ants[i].join())
            result.append(routes[0])
            self.update_state_params(points_to_use, sq_distances, pheromones, routes[0], n_points)
        return result


if __name__ == "__main__":
    processed_data_path = '../data/processed/'
    # groupedData = GroupedData(processed_data_path + 'points_test.csv', processed_data_path + 'groups.csv')
    # set_seed_for_random(20)
    # processed_data_path = '../data/processed/'
    model = Model(5, 50, 1, 1)
    model.load_data(GroupedData(processed_data_path + 'points.csv', processed_data_path + 'groups.csv'))
    # model.load_data(GroupedData(n_points=100, box_size=20, group_size=4))
    # model.show_routes([])
    routes = model.search_routes(50, 3)
    # model.show_routes(routes)
    print(routes)





