import copy
import random
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_definitions import ages
from data_definitions import genders
from data_definitions import nationalities
from data_definitions import time_of_day

from data_definitions import man_markets_first_floor
from data_definitions import man_markets_ground_floor
from data_definitions import man_markets_second_floor
from data_definitions import man_markets_third_floor
from data_definitions import woman_markets_first_floor
from data_definitions import woman_markets_ground_floor
from data_definitions import woman_markets_second_floor
from data_definitions import woman_markets_third_floor

from data_definitions import market_first_floor_edges
from data_definitions import market_ground_floor_edges
from data_definitions import market_second_floor_edges
from data_definitions import market_third_floor_edges

from data_definitions import super_markets

# data params
CUSTOMER_COUNT = 5000
path = f'../generated_data/data_{CUSTOMER_COUNT}.csv'

ADD_SUPER_MARKET_TO_END_BOOL = False
ADD_SUPER_MARKET_TO_END = 0.8

MALE_NUMBER_OF_STOPS_MEAN = 8
MALE_NUMBER_OF_STOPS_STD = 1

FEMALE_NUMBER_OF_STOPS_MEAN = 10
FEMALE_NUMBER_OF_STOPS_STD = 1

VISIT_TIME_MEAN = 120
VISIT_TIME_STD = 30

# dataframe columns
columns = ['id', 'genders', 'ages', 'nationalities', 'number_of_stops', 'time_of_day',
           'market_0', 'market_1', 'market_2', 'market_3', 'market_4',
           'market_5', 'market_6', 'market_7', 'market_8', 'market_9',
           'market_10', 'market_11', 'market_12', 'market_13', 'market_14',
           'time_market_0', 'time_market_1', 'time_market_2', 'time_market_3', 'time_market_4',
           'time_market_5', 'time_market_6', 'time_market_7', 'time_market_8', 'time_market_9',
           'time_market_10', 'time_market_11', 'time_market_12', 'time_market_13', 'time_market_14',
           'rate_market_0', 'rate_market_1', 'rate_market_2', 'rate_market_3', 'rate_market_4',
           'rate_market_5', 'rate_market_6', 'rate_market_7', 'rate_market_8', 'rate_market_9',
           'rate_market_10', 'rate_market_11', 'rate_market_12', 'rate_market_13', 'rate_market_14']

# plotting params
MARKET_COUNT = 75
MARKET_COUNT_INCREMENT = 1
MAX_VISIT_COUNT = 10000
VISIT_COUNT_INCREMENT = 100

MIN_VISIT_TIME = 0
MAX_VISIT_TIME = 300
VISIT_TIME_INCREMENT = 10
CUSTOMER_VISIT_TIME_COUNT = 1000
CUSTOMER_VISIT_TIME_INCREMENT = 50

'''':arg convert dict to list'''

gender_list = [gender for gender in genders]
gender_weights = [genders[gender][1] for gender in genders]

age_list = [age for age in ages]
age_weights = [ages[age][1] for age in ages]

nationality_list = [n for n in nationalities]
nationality_weights = [nationalities[n][1] for n in nationalities]

time_of_day_list = [time for time in time_of_day]
time_of_day_weights = [time_of_day[t_weight][1] for t_weight in time_of_day]

all_man_markets_dict = {
    **man_markets_ground_floor, **man_markets_first_floor, **man_markets_second_floor, **man_markets_third_floor
}

all_woman_markets_dict = {
    **woman_markets_ground_floor, **woman_markets_first_floor, **woman_markets_second_floor, **woman_markets_third_floor
}

man_markets_list = [market for market in all_man_markets_dict]
woman_markets_list = [market for market in all_woman_markets_dict]

man_market_weight_21_35 = [all_man_markets_dict[market][1] for market in man_markets_list]
man_market_weight_36_40 = [all_man_markets_dict[market][2] for market in man_markets_list]
man_market_weight_41_55 = [all_man_markets_dict[market][3] for market in man_markets_list]
man_market_weight_55_99 = [all_man_markets_dict[market][4] for market in man_markets_list]

woman_market_weight_21_35 = [all_woman_markets_dict[market][1] for market in woman_markets_list]
woman_market_weight_36_40 = [all_woman_markets_dict[market][2] for market in woman_markets_list]
woman_market_weight_41_55 = [all_woman_markets_dict[market][3] for market in woman_markets_list]
woman_market_weight_55_99 = [all_woman_markets_dict[market][4] for market in woman_markets_list]


# get count of nationalities
def get_count_of_nationalities_with_weights(data_frame):
    print('nationality counts : ')
    for nationality in nationality_list:
        print(f'nation : {nationality} weight : {nationalities[nationality][1]} '
              f'total count : {(data_frame.nationalities == nationality).sum()}')
    print('')


# get count of genders
def get_count_of_genders_with_weights(data_frame):
    print('gender counts : ')
    for gender in gender_list:
        print(f'gender : {gender} weight : {genders[gender][1]} '
              f'total count : {(data_frame.genders == gender).sum()}')
    print('')


# get count of ages
def get_count_of_ages_with_weights(data_frame):
    print('age counts : ')
    for age in age_list:
        print(f'age : {age} weight : {ages[age][1]} '
              f'total count : {(data_frame.ages == age).sum()}')
    print('')


# read data frame
def read_data_frame(file_path):
    return pd.read_csv(file_path, usecols=columns, low_memory=False)


# create synthetic data
def create_synthetic_data(file_path):
    data_frame = pd.DataFrame(columns=columns)
    print(f'add person : {datetime.datetime.now()}')
    for i in range(CUSTOMER_COUNT):
        selected_gender = random.choices(gender_list, weights=gender_weights)[0]
        selected_age = random.choices(age_list, weights=age_weights)[0]
        selected_nationality = random.choices(nationality_list, weights=nationality_weights)[0]
        selected_time_of_day = random.choices(time_of_day_list, weights=time_of_day_weights)[0]
        # inserting data_frame
        data_frame.loc[i] = [i, selected_gender, selected_age, selected_nationality, int(0), selected_time_of_day,
                             '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*',
                             int(0), int(0), int(0), int(0), int(0), int(0), int(0), int(0), int(0), int(0),
                             int(0), int(0), int(0), int(0), int(0), int(0), int(0), int(0), int(0), int(0),
                             int(0), int(0), int(0), int(0), int(0), int(0), int(0), int(0), int(0), int(0)]

    print(f'update_number_of_stops_column : {datetime.datetime.now()}')
    data_frame = update_number_of_stops_column(data_frame, path)
    print(f'insert_random_market_for_customer : {datetime.datetime.now()}')
    data_frame = insert_random_market_for_customer(data_frame, path)
    print(f'insert_random_time : {datetime.datetime.now()}')
    data_frame = insert_random_time(data_frame, path)
    print(f'insert_rates : {datetime.datetime.now()}')
    data_frame = insert_rates(data_frame, path)
    data_frame.to_csv(file_path, index=False)


def insert_rates(data_frame, file_path):
    times = np.zeros(shape=(VISIT_TIME_MEAN + 2 * VISIT_TIME_STD + 1), dtype='int')

    for index, row in data_frame.iterrows():
        for i in range(row['number_of_stops']):
            cell = row[f'time_market_{i}']
            print(f'cell : {cell}')
            times[cell] += 1

    sum_time = 0

    for i in range(len(times)):
        sum_time += (times[i] * i)

    sum_time_array = []
    for i in range(len(times)):
        for j in range(times[i]):
            sum_time_array.append(i)

    std_times = np.std(sum_time_array)
    sum_time /= sum(times)

    sum_time = np.floor(sum_time)
    std_times = np.floor(std_times)

    print(f'1 : 0 - {sum_time - std_times - std_times / 2}')
    print(f'2 : {sum_time - std_times - std_times / 2} - {sum_time - std_times}')
    print(f'3 : {sum_time - std_times} - {sum_time + std_times / 2}')
    print(f'4 : {sum_time + std_times / 2} - {sum_time + std_times + std_times / 2}')
    print(f'5 : {sum_time + std_times + std_times / 2} - ')

    ratings = np.ones(shape=(VISIT_TIME_MEAN + 2 * VISIT_TIME_STD + 1), dtype='int')
    for i in range(len(ratings)):
        if sum_time - std_times - std_times / 2 < i <= sum_time - std_times:
            ratings[i] = 2
        elif sum_time - std_times < i <= sum_time + std_times / 2:
            ratings[i] = 3
        elif sum_time + std_times / 2 < i <= sum_time + std_times + std_times / 2:
            ratings[i] = 4
        elif i > sum_time + std_times + std_times / 2:
            ratings[i] = 5

    for index, row in data_frame.iterrows():
        for i in range(row['number_of_stops']):
            data_frame.loc[data_frame.index[index], f'rate_market_{i}'] = int(ratings[row[f'time_market_{i}']])

    data_frame.to_csv(file_path)
    return data_frame


# insert stops
def update_number_of_stops_column(data_frame, file_path):
    for index, row in data_frame.iterrows():
        if row['genders'] == 'male':
            stop = int(np.random.normal(loc=MALE_NUMBER_OF_STOPS_MEAN, scale=MALE_NUMBER_OF_STOPS_STD))
            while stop == 0:
                stop = int(np.random.normal(loc=MALE_NUMBER_OF_STOPS_MEAN, scale=MALE_NUMBER_OF_STOPS_STD))
            data_frame.loc[data_frame.index[index], 'number_of_stops'] = stop
        else:
            stop = int(np.random.normal(loc=FEMALE_NUMBER_OF_STOPS_MEAN, scale=FEMALE_NUMBER_OF_STOPS_STD))
            while stop == 0:
                stop = int(np.random.normal(loc=FEMALE_NUMBER_OF_STOPS_MEAN, scale=FEMALE_NUMBER_OF_STOPS_STD))
            data_frame.loc[data_frame.index[index], 'number_of_stops'] = stop

    data_frame.to_csv(file_path)
    return data_frame


def insert_random_time(data_frame, file_path):
    for index, row in data_frame.iterrows():
        total_time = int(np.random.normal(loc=VISIT_TIME_MEAN, scale=VISIT_TIME_STD))
        total_time = min(total_time, VISIT_TIME_MEAN + 2 * VISIT_TIME_STD)
        random_times = []
        for i in range(row['number_of_stops']):
            random_times.append(np.random.randint(100, 600))
        _sum = sum(random_times)
        random_times = [x / _sum for x in random_times]
        random_times = [x * total_time for x in random_times]
        for i in range(len(random_times)):
            data_frame.loc[data_frame.index[index], f'time_market_{i}'] = int(random_times[i])
    data_frame.to_csv(file_path)
    return data_frame


def insert_market_to_row(data_frame, index, row, market_list, market_weight):
    selected_market_list = []
    for i in range(row['number_of_stops']):
        choice = random.choices(market_list, weights=market_weight)
        j = market_list.index(choice[0])
        del market_weight[j]
        market_list.remove(choice[0])
        selected_market_list.append(choice[0])

    sorted_selected_market_list = sorted(selected_market_list)

    if ADD_SUPER_MARKET_TO_END_BOOL:
        selected_super_markets = []
        for i in range(len(super_markets)):
            if super_markets[i] in sorted_selected_market_list:
                if random.choices([0, 1], weights=[1 - ADD_SUPER_MARKET_TO_END, ADD_SUPER_MARKET_TO_END]) != 0:
                    selected_super_markets.append(super_markets[i])

        for i in range(len(selected_super_markets)):
            market_index = sorted_selected_market_list.index(selected_super_markets[i])
            del sorted_selected_market_list[market_index]
            sorted_selected_market_list.append(selected_super_markets[i])

    for i in range(len(sorted_selected_market_list)):
        data_frame.loc[data_frame.index[index], f'market_{i}'] = int(sorted_selected_market_list[i])


def insert_random_market_for_customer(data_frame, file_path):
    for index, row in data_frame.iterrows():
        market_list = copy.deepcopy(man_markets_list)
        if row['genders'] == 'male':
            if row['ages'] == '21-35':
                insert_market_to_row(data_frame, index, row, market_list, copy.deepcopy(man_market_weight_21_35))
            elif row['ages'] == '36-40':
                insert_market_to_row(data_frame, index, row, market_list, copy.deepcopy(man_market_weight_36_40))
            elif row['ages'] == '41-55':
                insert_market_to_row(data_frame, index, row, market_list, copy.deepcopy(man_market_weight_41_55))
            else:
                insert_market_to_row(data_frame, index, row, market_list, copy.deepcopy(man_market_weight_55_99))
        else:
            if row['ages'] == '21-35':
                insert_market_to_row(data_frame, index, row, market_list, copy.deepcopy(woman_market_weight_21_35))
            elif row['ages'] == '36-40':
                insert_market_to_row(data_frame, index, row, market_list, copy.deepcopy(woman_market_weight_36_40))
            elif row['ages'] == '41-55':
                insert_market_to_row(data_frame, index, row, market_list, copy.deepcopy(woman_market_weight_41_55))
            else:
                insert_market_to_row(data_frame, index, row, market_list, copy.deepcopy(woman_market_weight_55_99))
    data_frame.to_csv(file_path)
    return data_frame


# plot column counts
def plot_number_of_stop_counts(data_frame, title='number of stop counts'):
    plt.title(title)
    stop_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for stop in data_frame['number_of_stops']:
        stop_list[stop] += 1
    plt.plot(stop_list)
    plt.show()


def plot_all_number_of_stop_counts(data_frame):
    '''    plot_number_of_stop_counts(data_frame.loc[(data_frame['genders'] == 'male') & (data_frame['ages'] == '21-35')],
                               title='male 21-35 number of stop counts')
    plot_number_of_stop_counts(data_frame.loc[(data_frame['genders'] == 'male') & (data_frame['ages'] == '36-40')],
                               title='male 36-40 number of stop counts')
    plot_number_of_stop_counts(data_frame.loc[(data_frame['genders'] == 'male') & (data_frame['ages'] == '41-55')],
                               title='male 41-55 number of stop counts')
    plot_number_of_stop_counts(data_frame.loc[(data_frame['genders'] == 'male') & (data_frame['ages'] == '55-99')],
                               title='male 56-99 number of stop counts')

    plot_number_of_stop_counts(data_frame.loc[(data_frame['genders'] == 'female') & (data_frame['ages'] == '21-35')],
                               title='female 21-35 number of stop counts')
    plot_number_of_stop_counts(data_frame.loc[(data_frame['genders'] == 'female') & (data_frame['ages'] == '36-40')],
                               title='female 36-40 number of stop counts')
    plot_number_of_stop_counts(data_frame.loc[(data_frame['genders'] == 'female') & (data_frame['ages'] == '41-55')],
                               title='female 41-55 number of stop counts')
    plot_number_of_stop_counts(data_frame.loc[(data_frame['genders'] == 'female') & (data_frame['ages'] == '55-99')],
                               title='female 55-99 number of stop counts')
    '''
    plot_number_of_stop_counts(data_frame.loc[data_frame['genders'] == 'male'], title='all male number of stop counts')
    plot_number_of_stop_counts(data_frame.loc[data_frame['genders'] == 'female'],
                               title='all female number of stop counts')
    plot_number_of_stop_counts(data_frame, title='all customer number of stop counts')


def plot_gender_market_statistics(data_frame, title=''):
    plt.title(title)

    x = np.arange(market_ground_floor_edges[0], market_ground_floor_edges[1] + 1)
    x_1 = np.arange(market_first_floor_edges[0], market_first_floor_edges[1] + 1)
    x_2 = np.arange(market_second_floor_edges[0], market_second_floor_edges[1] + 1)
    x_3 = np.arange(market_third_floor_edges[0], market_third_floor_edges[1] + 1)
    labels = np.concatenate((x, x_1, x_2, x_3))
    plt.xticks(np.arange(len(labels)), labels=labels, rotation='vertical')

    market_ids = np.zeros(market_third_floor_edges[1] + 1, dtype='int')
    for index, row in data_frame.iterrows():
        for i in range(row['number_of_stops']):
            market_ids[int(row[f'market_{i}'])] += 1

    size = len(market_ids)
    i = 0
    j = 0
    while i < size:
        if j not in labels:
            market_ids = np.delete(market_ids, i)
            size -= 1
        else:
            i += 1
        j += 1

    plt.plot(market_ids, '-ok')
    plt.show()


def plot_all_gender_market_statistics(data_frame):
    '''plot_gender_market_statistics(data_frame.loc[(data_frame['genders'] == 'male') & (data_frame['ages'] == '21-35')],
                                  title='male 21-35 visit counts')
    plot_gender_market_statistics(data_frame.loc[(data_frame['genders'] == 'female') & (data_frame['ages'] == '21-35')],
                                  title='female 21-35 visit counts')

    plot_gender_market_statistics(data_frame.loc[(data_frame['genders'] == 'male') & (data_frame['ages'] == '36-40')],
                                  title='male 36-40 visit counts')
    plot_gender_market_statistics(data_frame.loc[(data_frame['genders'] == 'female') & (data_frame['ages'] == '36-40')],
                                  title='female 36-40 visit counts')

    plot_gender_market_statistics(data_frame.loc[(data_frame['genders'] == 'male') & (data_frame['ages'] == '41-55')],
                                  title='male 41-55 visit counts')
    plot_gender_market_statistics(data_frame.loc[(data_frame['genders'] == 'female') & (data_frame['ages'] == '41-55')],
                                  title='female 41-55 visit counts')

    plot_gender_market_statistics(data_frame.loc[(data_frame['genders'] == 'male') & (data_frame['ages'] == '55-99')],
                                  title='male 56-99 visit counts')
    plot_gender_market_statistics(data_frame.loc[(data_frame['genders'] == 'female') & (data_frame['ages'] == '55-99')],
                                  title='female 56-99 visit counts')'''

    plot_gender_market_statistics(data_frame.loc[data_frame['genders'] == 'male'], title='all male visit counts')
    plot_gender_market_statistics(data_frame.loc[data_frame['genders'] == 'female'], title='all female visit counts')
    plot_gender_market_statistics(data_frame, 'all customers visit counts')


def plot_time_market_statistics(data_frame, title='time statistics'):
    plt.title(title)
    plt.xticks(np.arange(MIN_VISIT_TIME, MAX_VISIT_TIME, VISIT_TIME_INCREMENT))

    timestamps = np.zeros(MAX_VISIT_TIME, dtype='int')
    for index, row in data_frame.iterrows():
        time = 0
        if row['number_of_stops'] == 0:
            print(f'includes zero : {row}')
        for i in range(row['number_of_stops']):
            time += int(row[f'time_market_{i}'])
        timestamps[time] += 1
    plt.plot(timestamps, '-ok')
    plt.show()


def plot_all_time_market_statistics(data_frame):
    '''plot_time_market_statistics(data_frame.loc[(data_frame['genders'] == 'male') & (data_frame['ages'] == '21-35')],
                                title='male 21-35 visit times')
    plot_time_market_statistics(data_frame.loc[(data_frame['genders'] == 'male') & (data_frame['ages'] == '36-40')],
                                title='male 36-40 visit times')
    plot_time_market_statistics(data_frame.loc[(data_frame['genders'] == 'male') & (data_frame['ages'] == '41-55')],
                                title='male 41-55 visit times')
    plot_time_market_statistics(data_frame.loc[(data_frame['genders'] == 'male') & (data_frame['ages'] == '55-99')],
                                title='male 56-99 visit times')

    plot_time_market_statistics(data_frame.loc[(data_frame['genders'] == 'female') & (data_frame['ages'] == '21-35')],
                                title='female 21-35 visit times')
    plot_time_market_statistics(data_frame.loc[(data_frame['genders'] == 'female') & (data_frame['ages'] == '36-40')],
                                title='female 36-40 visit times')
    plot_time_market_statistics(data_frame.loc[(data_frame['genders'] == 'female') & (data_frame['ages'] == '41-55')],
                                title='female 41-55 visit times')
    plot_time_market_statistics(data_frame.loc[(data_frame['genders'] == 'female') & (data_frame['ages'] == '55-99')],
                                title='female 56-99 visit times')'''

    plot_time_market_statistics(data_frame.loc[(data_frame['genders'] == 'male')], title='all male visit times')
    plot_time_market_statistics(data_frame.loc[(data_frame['genders'] == 'female')], title='all female visit times')
    plot_time_market_statistics(data_frame, title='all customer visit times')


# create_synthetic_data(path)
# df = read_data_frame(path)
#
# get_count_of_nationalities_with_weights(df)
# get_count_of_genders_with_weights(df)
# get_count_of_ages_with_weights(df)
#
# plot_all_number_of_stop_counts(df)
# plot_all_gender_market_statistics(df)
# plot_all_time_market_statistics(df)
