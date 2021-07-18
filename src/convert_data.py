import numpy as np
from generate_mock_data import read_data_frame
from data_definitions import ages, genders, nationalities, time_of_day
import pandas as pd
import random

path = f'../generated_data/data_5000.csv'

columns = ['id', 'genders', 'ages', 'nationalities', 'number_of_stops', 'time_of_day',
           'market_0', 'market_1', 'market_2', 'market_3', 'market_4',
           'market_5', 'market_6', 'market_7', 'market_8', 'market_9',
           'time_market_0', 'time_market_1', 'time_market_2', 'time_market_3', 'time_market_4',
           'time_market_5', 'time_market_6', 'time_market_7', 'time_market_8', 'time_market_9',
           'rate_market_0', 'rate_market_1', 'rate_market_2', 'rate_market_3', 'rate_market_4',
           'rate_market_5', 'rate_market_6', 'rate_market_7', 'rate_market_8', 'rate_market_9']

cols = [
    'id', 'genders', 'ages', 'nationalities', 'time_of_day', 'market_id', 'market_rate'
]

test_cols = [
    'id', 'genders', 'ages', 'nationalities', 'time_of_day', 'visited_market_0', 'visited_market_1', 'visited_market_2',
    'visited_market_3', 'visited_market_4', 'visited_market_5', 'visited_market_6', 'visited_market_7',
    'visited_market_8', 'visited_market_9', 'visited_market_10', 'visited_market_11', 'visited_market_12',
    'visited_market_13', 'visited_market_14', 'market_id'
]

def add_row(row, market_number, rate, visited_markets=None):
    if visited_markets is None:
        return [row.id, genders[row.genders][0], ages[row.ages][0], nationalities[row.nationalities][0],
                time_of_day[row.time_of_day][0], market_number, rate]
    else:
        return [row.id, genders[row.genders][0], ages[row.ages][0], nationalities[row.nationalities][0],
                time_of_day[row.time_of_day][0], visited_markets[0][0], visited_markets[1][0], visited_markets[2][0],
                visited_markets[3][0], visited_markets[4][0], visited_markets[5][0], visited_markets[6][0],
                visited_markets[7][0], visited_markets[8][0], visited_markets[9][0], visited_markets[10][0],
                visited_markets[11][0], visited_markets[12][0], visited_markets[13][0], visited_markets[14][0], market_number]


def add_visited_to_test(df):
    df_train = pd.DataFrame(columns=cols)
    df_test = pd.DataFrame(columns=test_cols)
    iterator = 0
    for index, row in df.iterrows():
        print(f'row : {index}')
        number_of_stops = row.number_of_stops
        random_index = random.randint(0, number_of_stops - 1)
        visited_markets = -np.ones(shape=(15, 1), dtype=np.int)
        _iter = 0
        for i in range(number_of_stops):
            if i != random_index:
                visited_markets[_iter] = int(row[f'market_{i}']) % 100
                _iter += 1
        for i in range(number_of_stops):
            current_market = int(row[f'market_{i}']) % 100
            if i != random_index:
                df_train.loc[iterator] = add_row(row, current_market, row[f'rate_market_{i}'])
            else:
                df_test.loc[iterator] = add_row(row, current_market, row[f'rate_market_{i}'], visited_markets)
            iterator += 1
    df_train.to_csv('../generated_data/data_5000_train.dat', sep=':', index=False)
    df_test.to_csv('../generated_data/data_5000_test.dat', sep=':', index=False)


df = read_data_frame(path)
add_visited_to_test(df)
