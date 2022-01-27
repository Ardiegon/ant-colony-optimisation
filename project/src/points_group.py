import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians

class PointsGroup:
    def __init__(self, group_id, min_latitude, max_latitude, min_longitude, max_longitude):
        self.group_id = group_id
        self.min_latitude = min_latitude
        self.max_latitude = max_latitude
        self.min_longitude = min_longitude
        self.max_longitude = max_longitude
        self.points = []
        self.mid_coordinates = None
        self.neighbours = []

    def calculate_mid_coordinates(self):
        coordinates = [0, 0]
        for point in self.points:
            coordinates[0] += point[0]
            coordinates[1] += point[1]
        self.mid_coordinates = [coordinates[0]/len(self.points), coordinates[1]/len(self.points)]

    def add_point_to_group(self, point):
        self.points.append(point)
        # self.calculate_mid_coordinates()
    
    def add_neighbour(self, neighbour):
        self.neighbours.append(neighbour)

def generate_groups():
    groups = []
    min_latitude = -90
    max_latitude = 90
    min_longitude = -180
    max_longitude = 180
    degree_step = 6

    current_latitude = min_latitude
    group_id = 0

    while current_latitude < max_latitude:
        current_longitude = min_longitude
        while current_longitude < max_longitude:
            group = PointsGroup(group_id, current_latitude, current_latitude + degree_step, current_longitude,
                                current_longitude + degree_step)
            group_id += 1
            current_longitude += degree_step
            groups.append(group)
        current_latitude += degree_step
    return groups

def split_data(df):
    groups = generate_groups()
    i = 0
    df2 = df.reindex(columns=df.columns.tolist() + ['group_id'])
    for group in groups:
        points = df[(df['Latitude'] >= group.min_latitude) & (df['Latitude'] <= group.max_latitude) &
                   (df['Longitude'] >= group.min_longitude) & (df['Longitude'] <= group.max_longitude)]
        for id in points['GiftId'].tolist():
            # print(points['GiftId'].tolist())
            group.add_point_to_group(id)
            df2.loc[id, 'group_id'] = group.group_id
    return groups, df2

def group_list_to_df(group_list):
    data = []
    for group in group_list:
        data.append([group.group_id, group.min_latitude, group.max_latitude, group.min_longitude, group.max_longitude, []])
    df = pd.DataFrame(data, columns=['group_id', 'min_latitude', 'max_latitude', 'min_longitude', 'max_longitude', 'neighbours_id'])
    return df

def find_neighbours(group_df):
    df = group_df.copy()
    for index, row in group_df.iterrows():
        n_id_list = []
        if int(row['min_latitude']) == -90:
            n1 = group_df[(group_df['min_latitude'] == -90.0)]
        else:
            n1 = group_df[(group_df['max_latitude'] == row['min_latitude']) & (group_df['min_longitude'] == row['min_longitude'])]
        if int(row['max_latitude']) == 90:
            n2 = group_df[(group_df['max_latitude'] == 90.0)]
        else:
            n2 = group_df[(group_df['min_latitude'] == row['max_latitude']) & (group_df['min_longitude'] == row['min_longitude'])]
        if int(row['max_longitude']) == 180:
            n3 = group_df[
                (group_df['min_latitude'] == row['min_latitude']) & (group_df['min_longitude'] == -180.0)]
        else:
            n3 = group_df[(group_df['min_latitude'] == row['min_latitude']) & (group_df['min_longitude'] == row['max_longitude'])]
        if int(row['min_longitude']) == -180:
            n4 = group_df[
                (group_df['min_latitude'] == row['min_latitude']) & (group_df['max_longitude'] == 180.0)]
        else:
            n4 = group_df[(group_df['min_latitude'] == row['min_latitude']) & (group_df['max_longitude'] == row['min_longitude'])]
        n1 = n1['group_id'].tolist()
        n2 = n2['group_id'].tolist()
        n3 = n3['group_id'].tolist()
        n4 = n4['group_id'].tolist()
        n_id_list = n1 + n2 + n3 + n4
        df.at[index, 'neighbours_id'] = n_id_list
    return df

        # print(n1, n2, n3, n4)

# def find_neighbours(group_df):
#     df = group_df.copy()
#     for index, row in group_df.iterrows():
#         neighbour_list = []
#         for index2, row2 in group_df.iterrows():
#             if row['min_latitude'] == row2['max_latitude'] or row['max_latitude'] == row2['min_latitude']:
#                 if row['min_longitude'] == row2['max_longitude'] or row['max_longitude'] == row2['min_longitude']:
#                     if index2 not in neighbour_list:
#                         neighbour_list.append(index2)
#         df.at[index, 'neighbours_id'] = neighbour_list
#     return df

def caluclate_distance_from_north_pole(points_df):
    np_lat = radians(90)
    np_long = radians(0)
    R = 6373.0 # approximate radius of earth in km
    df = points_df.copy()
    df['distance_np'] = np.nan
    for index, row in points_df.iterrows():
        lat = radians(row['Latitude'])
        long = radians(row['Longitude'])
        dlon = long - np_long
        dlat = lat - np_lat
        a = sin(dlat / 2)**2 + cos(np_lat) * cos(lat) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        df.at[index, 'distance_np'] = distance
    return df

def save_points_list_to_csv(points_list_df, path):
    points_list_df = caluclate_distance_from_north_pole(points_list_df)
    points_list_df.to_csv(path + 'points_test.csv', index = False) 

def save_groups_to_csv(groups_list, path):
    path = path + 'groups_test.csv'
    group_df = group_list_to_df(groups_list)
    df = find_neighbours(group_df)
    df.to_csv(path, index = False) 

if __name__ == "__main__":
    generate_groups()
    data_path = '../data/raw/'
    gift_df = pd.read_csv(data_path + 'gifts.csv')
    sample_submission_df = pd.read_csv(data_path + 'sample_submission.csv')
    # group_list = split_data(gift_df)
    north_pole_data = []
    north_pole_data.insert(0, {'GiftId': 0, 'Latitude': 90, 'Longitude': 0, 'Weight': 0, 'group_id': None})
    df_test = gift_df.head(2)
    df_test = pd.concat([pd.DataFrame(north_pole_data), df_test], ignore_index=True)
    # print(len(df_test))
    group_list, points_df = split_data(df_test)
    print(group_list, points_df)
    processed_data_path = '../data/processed/'
    # points_df = pd.concat([pd.DataFrame(north_pole_data), points_df], ignore_index=True)
    # save_points_list_to_csv(points_df, processed_data_path)
    save_groups_to_csv(group_list, processed_data_path)
    