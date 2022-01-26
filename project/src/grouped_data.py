import numpy as np
from math import floor, sqrt

class GroupedData:
    def __init__(self, data_path = "", n_points = 50, box_size = 20, group_size = 4, point_size = (1,50)):
        if data_path == "":
            self.data = self.generate_data(n_points, box_size, group_size, point_size)
        else:
            self.data = self.load_data(data_path)

    def generate_data(self, n_points, box_size, group_size, point_size):
        start_point = np.array([0, box_size/2, box_size/2, 0, 0])
        point_id = (np.arange(n_points-1) + 1)
        x_cor = np.random.rand(n_points-1)*box_size
        y_cor = np.random.rand(n_points-1)*box_size
        weight = (np.random.rand(n_points-1)*(point_size[1]-point_size[0])+point_size[0])
        start_distance = np.zeros(n_points-1)
        for i in range(n_points-1):
            start_distance[i] = sqrt((x_cor[i] - box_size/2)**2+(y_cor[i] - box_size/2)**2)
        points = np.stack((point_id, x_cor, y_cor, weight, start_distance), axis=1)
        start_point = np.expand_dims(start_point, 0)
        points = np.concatenate((start_point, points), axis=0)
        start_point = np.squeeze(start_point, axis=0)
        start_group = 0

        groups_in_side = int(box_size / group_size)
        number_of_groups = groups_in_side**2
        groups = [0 for _ in range(number_of_groups)]
        group_neighbours = []
        print("Generating Data Groups...")
        for group_id in range(int(number_of_groups)):
            neighbours = np.array([])
            g_col = group_id % groups_in_side
            g_row = int(group_id / groups_in_side)
            if g_col < groups_in_side-1:
                neighbours = np.append(neighbours, group_id+1)
            if g_col > 0:
                neighbours = np.append(neighbours, group_id-1)
            if g_row < groups_in_side-1:
                neighbours = np.append(neighbours, group_id+groups_in_side)
            if g_row > 0:
                neighbours = np.append(neighbours, group_id-groups_in_side)
            group_neighbours.append(neighbours)
            g_points = []
            for point in points:
                if group_size*(g_col) <= point[1] < group_size*(g_col + 1) and point[2] >= group_size*(g_row) and point[2] < group_size*(g_row + 1):
                    if point[0] == 0:
                        start_group = group_id
                    g_points.append(point)
            groups[group_id] = g_points
            print(f"\r\tgroup {group_id}", end = "")
        print("")

        return points, groups, group_neighbours, start_point, start_group

    def load_data(self, data_path):
        pass

    def get_data(self):
        return self.data