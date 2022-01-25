import numpy as np
from math import floor
import pandas as pd
from points_group import PointsGroup

class GroupedData:
    def __init__(self, points_data_path = None, groups_data_path = None):
        if points_data_path is None or groups_data_path is None:
            self.data = self.generate_data()
        else:
            self.data = self.load_data(points_data_path, groups_data_path)

    def generate_data(self, n_points = 50, box_size = 20, group_size = 4, point_size = (1,50)):
        point_id = (np.arange(n_points) + 1).astype(int)
        x_cor = np.random.rand(n_points)*box_size
        y_cor = np.random.rand(n_points)*box_size
        weight = (np.random.rand(n_points)*(point_size[1]-point_size[0])+point_size[0])
        points = np.stack((point_id, x_cor, y_cor, weight), axis=1)
        start_point = np.expand_dims(np.array([0, box_size/2, box_size/2, 0]), 0)
        points = np.concatenate((start_point, points), axis=0)
        start_point = np.squeeze(start_point, axis=0)
        print(start_point)
        start_group = 0

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
                if group_size*(g_col) <= point[1] < group_size*(g_col + 1) and point[2] >= group_size*(g_row) and point[2] < group_size*(g_row + 1):
                    if point[0] == 0:
                        start_group = group_id
                    g_points.append(point)
        #     groups[group_id] = g_points
        return points, groups, group_neighbours, start_point, start_group

    def load_data(self, points_data, groups_data):
        points_df = pd.read_csv (points_data)
        groups_df = pd.read_csv (groups_data)
        groups = []
        group_neighbours = []

        for index, row in groups_df.iterrows():
            group_id = row['group_id']
            min_latitude = row['min_latitude']
            max_latitude = row['max_latitude']
            min_longitude = row['min_longitude']
            max_longitude = row['max_longitude']
            neighbors_id = row['neighbours_id']
            neighbors_id = list(neighbors_id)
            neighbors_id.remove('[')
            neighbors_id.remove(']')
            group = PointsGroup(group_id, min_latitude, max_latitude, min_longitude, max_longitude)
            for neighbour in neighbors_id:
                group.add_neighbour(neighbour)
                group_neighbours.append([index, int(neighbour)])
            groups.append(group)
        start_point = points_df.loc[0].tolist()
        start_point = np.array(start_point)
        points = [points_df.loc[i].tolist() for i in range(1, len(points_df))]
        points = np.array(points)
        np.set_printoptions(suppress=True)
        start_group = groups[0].group_id
        # print(points)
        # print(groups)
        # print(group_neighbours)
        # print(start_group)
        # print(start_point)
        return points, groups, group_neighbours, start_point, start_group

    def get_data(self):
        return self.data