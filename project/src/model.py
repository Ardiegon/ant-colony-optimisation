import matplotlib.pyplot as plt
import numpy as np
from project.src.thread_with_return import ThreadWithReturn
from random import uniform
from math import radians
from queue import Queue
from project.src.grouped_data import GroupedData


def get_key(point_1_id, point_2_id):
    '''
    Creates key for further use in tables with distances and pheromones for wo specific points
    :param point_1_id: first point id
    :param point_2_id: second point id
    :return: String key, for example p1 = 3, p2 = 0 => result = "0/3"
    '''
    return f"{int(min(point_1_id, point_2_id))}/{int(max(point_1_id, point_2_id))}"

class Model:
    '''
    Model calculating optimal routes for kaggle problem:
    https://www.kaggle.com/c/santas-stolen-sleigh/overview/description
    '''
    def __init__(self, _n_ants, _backpack_size, _pheromone_weight, _distance_weight, _size_weight, _home_weight, _selection_size):
        '''
        Model constructor.
        :param _n_ants: Size of population, number of threads to calculate one route at the same time.
        :param _selection_size: How many routes from single generation will leave pheromone trails for further generations
        :param _backpack_size: Maximum weight of presents to put in sleigh
        :param _pheromone_weight: Used to count desirability in single ant
        :param _distance_weight: Used to count desirability in single ant
        :param _size_weight: Used to count desirability in single ant
        :param _home_weight: Used to count desirability in single ant
        '''
        self.points, self.groups, self.groups_neighbours, self.start_point, self.start_group, self.picked_points, self.added_points, self.used_groups = (None, None, None, None, None, None, None, None)
        self.n_ants = _n_ants
        self.pheromone_weight = _pheromone_weight
        self.backpack_size = _backpack_size
        self.distance_weight = _distance_weight
        self.size_weight = _size_weight
        self.home_weight = _home_weight
        self.selection_size = _selection_size

    def load_data(self, gdata: GroupedData):
        '''
        Loads data from GroupedData class. Data format is:
        - points: list of all presents as [point1, point2, ...]
            - single point is defined as [point_id, latitude, longitude, weight, distance_to_arctica]
        - groups: list of grouped points into smaller portions
        - group_neighbours: all neighbors of group given by id of this list
        - start_point: arctica, point with id 0
        - start group: group which arctica belongs to
        :param gdata: object of class with prepared data
        '''
        self.points, self.groups, self.groups_neighbours, self.start_point, self.start_group = gdata.get_data()
        self.picked_points = np.zeros(len(self.points))
        self.added_points = np.zeros(len(self.points))
        self.used_groups = np.zeros(len(self.groups))


    def pick_points(self, min_points):
        '''
        Initiates first batch of points to go into processing. We chose groups of points that are close together,
        and use nly them for less memory complexity, because those points will often calculate things with O(n^2)

        Rule is that from starting group, we get neighbours of it into priority queue. When all points from current
        group are taken we move into next group, and take it's neighbours and points and so on.

        :param min_points: Minimum number of picked points. There could be more of them, but only about size of single group.
        :return list of points to further process.
        '''
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
        '''
        For picked batch of point we calculates distances beetween them, as it's required to process single ant with
        optimal time complexity.
        :param chosen_points: batch of points (look: pick_points)
        :return: dictionary of distances between every two points in chosen_points. Uses key from get_key(p1,p2)
        '''
        distances = {}
        n_all = len(chosen_points)
        print(f"Initiating distances of {len(chosen_points)} points squared ...")
        for i in range(len(chosen_points)):
            for j in range(i+1, len(chosen_points)):
                key = get_key(chosen_points[i][0], chosen_points[j][0])
                distances[key] = self.distance(chosen_points[i], chosen_points[j])
            print(f"\r\tcalculated {(int(((i+1)*n_all)-((i+1)**2-(i+1))/2+i+1)/int((n_all*n_all-n_all)/2))*100 :.2f}%", end="")
        print(f"\r\tcalculated 100%, finished")
        return distances

    def init_pheromones(self, chosen_points):
        '''
        Rule is the same as init_sq_distances, it initiates pheromones dictionary with zeros.
        :param chosen_points: batch of points (look: pick_points)
        :return: dictionary of pheromones between every two points in chosen_points. Uses key from get_key(p1,p2)
        '''
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
        """
        As we got our route, points of that route are no longer use for us. This function updates batch of points by
        adding new if number of points in batch is lower than minimum. After new batch is taken, we need to delete old
        distances and pheromones, because they are bound to points that are no longer in batch.
        For newly added points we need to calculate distances and initialize pheromones.

        It works strongly on references, to avoid any deepcopy.

        :param c_points: batch of points
        :param c_distances: distances dictionary
        :param c_pheromones: pheromones dictionary
        :param route: newly picked route
        :param min_points: minimum number of points in c_points
        """
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
        '''
        Updates points as there was a new route picked.
        :param c_points: reference to batch of points
        :param min_points: minimum number of points in c_points
        :return: list of vectors [id_new_point, latitude_new_point, longitude_new_point]
        representing points newly added to c_points
        '''
        group_q = Queue()
        group_q.put(self.start_group)
        group_history = [self.start_group]
        new_p_ids = []
        print(f"\nUpdating points, minimum {min_points}, current {len(c_points)}, left {int(len(self.points) - sum(self.added_points))}")
        while len(c_points) < min_points:
            if group_q.empty():
                print(f"\r\tOnly {len(c_points)} left")
                break
            curr_group = int(group_q.get())
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

    def update_distances_and_pheromones(self, c_points, c_distances, c_pheromones, old_p, new_p, route):
        '''
        Updates distances and pheromones dictionaries after new points were added to c_points and old deleted
        :param c_points: reference to updated batch of points
        :param c_distances: distances between points
        :param c_pheromones: pheromone between points
        :param old_p: points that last from previous generation
        :param new_p: points added to current generation
        :param route: points ids that were removed from previous generation
        :return:
        '''
        print("Updating distances and pheromones...", end = "")
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
                c_distances[get_key(p1[0], p2[0])] = self.distance(p1, p2)
                c_pheromones[get_key(p1[0], p2[0])] = 0
        for r1 in range(0, new_p_len):
            for r2 in range(r1+1, new_p_len):
                c_distances[get_key(new_p[r1][0], new_p[r2][0])] = self.distance(new_p[r1], new_p[r2])
                c_pheromones[get_key(new_p[r1][0], new_p[r2][0])] = 0
        print("\rUpdating distances and pheromones... finished")


    def ant_trivial(self, c_points, c_distances, c_pheromones):
        '''
        Trivial algorithm for generation specimens.
        It has zero desision-making skills, and it simply takes more and more random points from
        neighbourhood untill backpack is full.
        :param c_points: reference to batch of points
        :param c_distances: reference to distances dict
        :param c_pheromones: reference to pheromones dict (unused)
        :return: route, score
        score is weighted reindeer weariness for this route only.
        '''
        route = [0]
        weights = []
        backpack = 0
        i = 1
        while True:
            if i< len(c_points) and backpack + c_points[i][3] < self.backpack_size:
                weights.append(c_points[i][3])
                backpack += c_points[i][3]
                route.append(int(c_points[i][0]))
                i += 1
            else:
                break
        weights.append(0)
        route.append(0)
        score = self.get_score(route, weights, c_distances)
        return route, score

    def ant(self, c_points, c_distances, c_pheromones):
        '''
        Sophisticated algorithm for generation and mutation of new specimen in genetic algorithm.
        From starting point (arctica) it's job is to chose new point to add to the route. It does it by calculating
        four important indicators:
        - distance to point
        - pheromone trail (this route was awarded in previous iterations of this generation)
        - weight of present on that point
        - distance to arctica
        All of them are combined into desirability of point, and then chosen by random number generator.

        :param c_points: reference to batch of points
        :param c_distances: reference to distances dict
        :param c_pheromones: reference to pheromones dict
        :return: route, score
        score is weighted reindeer weariness for this route only.
        '''
        route = [0]
        weights = []
        backpack = 0
        used = np.zeros([len(c_points)])
        used[0] = 1
        last_point = c_points[0]
        while True:  # while we have space in backpack
            backpack_occupancy = backpack/self.backpack_size
            desirability = np.zeros([len(c_points)])
            counter = 0
            for i, point in enumerate(c_points):  # for every point in batch
                key = get_key(route[-1], point[0])
                if not used[i] and backpack + point[3] < self.backpack_size:  # that is not in route and can be put in backpack
                    counter += 1       # calculate desirability
                    desirability[i] = (1/c_distances[key])**self.distance_weight * (c_pheromones[key]**self.pheromone_weight + point[3]**self.size_weight + point[4]**self.home_weight*backpack_occupancy)
            if not counter:
                # (if there is no points of that type that means backpack is full or we run out of points)
                break
            max_rand = np.sum(desirability)
            decision = uniform(0, max_rand)  # and generate random number
            des_sum = 0
            best_point_id = 0
            for i, desire in enumerate(desirability):
                if desire == 0:
                    continue
                des_sum += desire
                if des_sum>decision:
                    best_point_id = i  # which is used to find new point
                    break
            used[i] = 1
            backpack += c_points[best_point_id][3]
            weights.append(c_points[best_point_id][3])
            route.append(int(c_points[best_point_id][0]))
        weights.append(0)
        route.append(0)
        score = self.get_score(route, weights, c_distances)
        return route, score

    def search_routes(self, n_points, gen_counter = 4):
        '''
        Genetic algorithm with route as specimen, and "ants" as specimen generators.
        All of Reproduction, Mutation and Succesion are handled by ants and their pheromones.
        It's further described in documentation.
        :param n_points: minimum number of points in batch
        :param gen_counter: how many generations will it take to generate best route
        :return: route, all_score
        all_score is weighted reindeer weariness for all_routes.
        '''
        pheromone_values = [0.5*((1/self.n_ants)*x-1)**2 for x in range(self.n_ants)]
        points_to_use = self.pick_points(n_points)
        sq_distances = self.init_sq_distances(points_to_use)
        pheromones = self.init_pheromones(points_to_use)
        result = []
        all_score = 0
        while len(points_to_use)>1: # while we have points other than arctica
            picked_route = 0
            picked_score = 0
            for g in range(gen_counter):
                g_routes = []
                g_scores = np.zeros(self.n_ants)
                ants = []
                for i in range(self.n_ants):
                    ants.append(ThreadWithReturn(target=self.ant, args=(points_to_use, sq_distances, pheromones)))  # send an ant for quest of searching new route for santa
                    ants[i].start()
                for i in range(self.n_ants):
                    route, score = ants[i].join()  # and wait till they all found some route
                    g_routes.append(route)
                    g_scores[i] = score
                best_ant = np.argmin(g_scores)
                max_val = np.max(g_scores)
                if picked_score < g_scores[best_ant]:  # remember best route from all generations
                    picked_route = g_routes[best_ant]
                    picked_score = g_scores[best_ant]
                for i in range(self.selection_size):  # and award best ants from this generation with pheromone to leave on their routes
                    top_id = np.argmin(g_scores)
                    g_scores[top_id]+=max_val
                    for j in range(len(g_routes[top_id])-1):
                        key = get_key(g_routes[top_id][j], g_routes[top_id][j+1])
                        pheromones[key] += pheromone_values[i]

            result.append(picked_route)  # add best route from this generation to all routes
            all_score += picked_score  # calculate step of weighted reindeer weariness
            self.update_state_params(points_to_use, sq_distances, pheromones, picked_route, n_points) # and update batch of points.
        return result, all_score


    def distance(self, p1, p2):
        """
        Haversine.
        """
        # convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [p1[1], p1[2], p2[1], p2[2]])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return c

    def get_score(self, route, weights, c_distances):
        '''
        single iteration of weighted reindeer weariness
        :param route: route that starts with arctica and ends with arctica
        :param weights: weight of all presents on that route
        :param c_distances: distances between points in batch
        :return:
        '''
        sc = 0
        weight_now = 10 + sum(weights)
        for i in range(len(route)-1):
            distance = c_distances[get_key(route[i], route[i + 1])]
            sc += weight_now * distance
            weight_now -= weights[i]
        return sc

    def show_routes(self, routes = None):
        '''
        Shows routes on map. For big number of points this function will work very slow.
        It's better to use with synthetic data only.
        :param routes: Routes to show on map
        '''
        def randomRGB(min, max):
            return (uniform(min,max),uniform(min,max),uniform(min,max))
        def connectpoints(p1, p2, clr):
            x1, x2 = self.points[p1][1], self.points[p2][1]
            y1, y2 = self.points[p1][2], self.points[p2][2]
            plt.plot([x1, x2], [y1, y2], color = clr, linewidth= 0.5)

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
        '''
        Plots list of points on map.
        :param points: list of points
        '''
        for point in points:
            if point[0] == 0:
                plt.annotate("Start", (point[1], point[2]))
                plt.scatter(point[1], point[2], marker = "X", color = (0.0,0.0,0.0))
            else:
                plt.scatter(point[1], point[2], color= (1.0,0.0,0.0))
                plt.annotate(f"{int(point[0])} / {int(point[3])}", (point[1], point[2]), fontsize=5)
        plt.show()


