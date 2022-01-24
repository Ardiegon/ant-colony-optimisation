import pandas as pd
import numpy as np

class PointsGroup:
    def __init__(self, group_id, min_latitude, max_latitude, min_longitude, max_longitude):
        self.group_id = group_id
        self.min_latitude = min_latitude
        self.max_latitude = max_latitude
        self.min_longitude = min_longitude
        self.max_longitude = max_longitude
        self.points = []
        self.mid_coordinates = None

    def calculate_mid_coordinates(self):
        coordinates = [0, 0]
        for point in self.points:
            coordinates[0] += point[0]
            coordinates[1] += point[1]
        self.mid_coordinates = [coordinates[0]/len(self.points), coordinates[1]/len(self.points)]

    def add_point_to_group(self, point):
        self.points.append(point)
        self.calculate_mid_coordinates()

def split_data(df):
    max_latitude = df['Latitude'].max()
    min_latitude = df['Latitude'].min()
    min_longitude = df['Longitude'].min()
    current_latitude = min_latitude
    current_longitude = min_longitude
    group_list = []
    group_id = 0
    df2 = df.reindex(columns = df.columns.tolist() + ['group_id'])
    while current_latitude <= max_latitude:
        df_latitude = df[df['Latitude'].between(current_latitude, current_latitude+1)]
        current_latitude += 1
        while df_latitude.shape[0] != 0:
            df_lat_long = df_latitude[df_latitude['Longitude'].between(current_longitude, current_longitude+1)]
            if df_lat_long.shape[0] != 0:
                group = PointsGroup(group_id, current_latitude, current_latitude + 1, current_longitude, current_longitude + 1)
                group_id += 1
                for i in df_lat_long.index:
                    group.add_point_to_group(df_lat_long.loc[i, ['Latitude', 'Longitude']].tolist())
                    df2.loc[i, 'group_id'] = group.group_id
                    df_latitude = df_latitude.drop(i)
                group_list.append(group)
            current_longitude += 1
        current_longitude = min_longitude
    return group_list, df2

def group_list_to_df(group_list):
    data = []
    for group in group_list:
        data.append([group.group_id, group.min_latitude, group.max_latitude, group.min_longitude, group.max_longitude, []])
    df = pd.DataFrame(data, columns=['group_id', 'min_latitude', 'max_latitude', 'min_longitude', 'max_longitude', 'neighbours_id'])
    return df

def save_points_list_to_csv(points_list_df, path):
    # TODO: odległość punktu od startu
    points_list_df.to_csv(path + 'points.csv', index = False) 

def save_groups_to_csv(groups_list, path):
    # TODO: sąsiedzi
    path = path + 'groups.csv'
    groups_df = group_list_to_df(groups_list)
    groups_df.to_csv(path, index = False) 

if __name__ == "__main__":
    data_path = '../data/raw/'
    gift_df = pd.read_csv(data_path + 'gifts.csv')
    sample_submission_df = pd.read_csv(data_path + 'sample_submission.csv')
    # group_list = split_data(gift_df)
    df_test = gift_df.head(20)
    group_list, points_df = split_data(df_test)
    processed_data_path = '../data/processed/'
    save_points_list_to_csv(points_df, processed_data_path)
    save_groups_to_csv(group_list, processed_data_path)
    