import numpy as np
from math import floor

class GroupedData:
    def __init__(self, data_path = ""):
        if data_path == "":
            self.data = self.generate_data()
        else:
            self.data = self.load_data(data_path)

    def generate_data(self, n_points = 30, box_size = 10, group_size = 2, point_size = (1,50)):
        x_cor = np.random.rand(n_points)*box_size
        y_cor = np.random.rand(n_points)*box_size
        weight = (np.random.rand(n_points)*(point_size[1]-point_size[0])+point_size[0])
        points = np.stack((x_cor, y_cor, weight), axis=1)

        groups_in_side = int(box_size / group_size)
        number_of_groups = groups_in_side**2
        groups = [0 for _ in range(number_of_groups)]
        group_neighbours = []

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
                if point[0] >= group_size*(g_col) and point[0] < group_size*(g_col+1) and point[1] >= group_size*(g_row) and point[1] < group_size*(g_row+1):
                    g_points.append(point)
            groups[group_id] = g_points

        return points, groups, group_neighbours

    def load_data(self, data_path):
        pass

    def get_data(self):
        return self.data