import pandas as pd
from src.points_group import split_data


if __name__ == "__main__":
    data_path = './data/raw/'
    gift_df = pd.read_csv(data_path + 'gifts.csv')
    sample_submission_df = pd.read_csv(data_path + 'sample_submission.csv')
    # group_list = split_data(gift_df)
    group_list = split_data(gift_df.head(200))
