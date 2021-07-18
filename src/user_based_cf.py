import numpy as np
from scipy.spatial.distance import cdist
from more_itertools import take


def create_user_item_matrix(data_frame, market_count):
    print('create_user_item_matrix')
    matrix = np.zeros(shape=(data_frame.shape[0] + 1, market_count), dtype='float32')
    test_market = np.zeros(shape=(data_frame.shape[0] + 1, 1), dtype='int')
    for index, row in data_frame.iterrows():
        for i in range(int(row['number_of_stops'])):
            matrix[index][int(row[f'market_{i}']) % 100] = row[f'rate_market_{i}']
            if int(row['number_of_stops']) - 1 == i:
                test_market[index] = int(row[f'market_{i}']) % 100

    print('create_user_item_matrix')
    return matrix, test_market


def calculate_zero_centered_matrix(matrix):
    zero_centered_matrix = []
    for row in matrix:
        rate_mean = np.mean([i for i in row if i != 0])
        new_row = [i - rate_mean if i != 0 else i for i in row]
        zero_centered_matrix.append(new_row)
    return np.array(zero_centered_matrix, dtype='float')


def calculate_similarity(matrix, user_index, similar_user_count):
    current_user = matrix[user_index]

    similarities = 1 - cdist(current_user.reshape(1, -1), matrix, metric='cosine')
    similarities[np.isnan(similarities)] = 0
    similar_users = dict(enumerate(similarities.flatten(), 0))
    similar_users = {k: v for k, v in sorted(similar_users.items(), key=lambda item: item[1], reverse=True)}
    similar_users.pop(user_index)

    return take(similar_user_count, similar_users.items())


def predict_rate(matrix, zero_matrix, similar_users, index, recommendation_count):
    row = np.zeros(shape=(75, 1))
    sum_weight = 0
    for user, weight in similar_users:
        sum_weight += weight
        row += matrix[user].reshape(-1, 1) * weight
    row /= sum_weight
    visited_markets = [i for i, item in enumerate(zero_matrix[index]) if item != 0]
    for i in visited_markets:
        row[i] = -1

    sorted_row = np.argsort(-row.reshape(1, -1))[:, 0: recommendation_count]
    return sorted_row


def predict_next_market(zero_centered_matrix, original_matrix, index, target, similar_user_count, recommendation_count):
    similar_users = calculate_similarity(zero_centered_matrix, index, similar_user_count)
    return predict_rate(original_matrix, zero_centered_matrix, similar_users, index, recommendation_count)


def predict_all_customer(matrix, test_market, sample_count, similar_user_count, recommendation_count):
    zero_centered_matrix = calculate_zero_centered_matrix(matrix)
    acc = 0
    for index, row in enumerate(zero_centered_matrix):
        temp_rate = row[test_market[index]]
        row[test_market[index]] = 0.
        # predict_rate
        if index % 100 == 0:
            print(f'index : {index} - acc : {acc}')
        if test_market[index] in predict_next_market(zero_centered_matrix, matrix, index, test_market[index],
                                                     similar_user_count, recommendation_count):
            acc += 1
        row[test_market[index]] = temp_rate
    print(f'true : {acc} - sample count : {sample_count} - acc : {acc * 100 / sample_count}')
    return acc * 100 / sample_count


def predict_selected_customer(matrix, customer, customer_index, similar_user_count, recommendation_count):
    zero_centered_matrix = calculate_zero_centered_matrix(matrix)
    index = int(customer[f'number_of_stops'])
    target = int(customer[f'market_{index - 1}']) % 100
    zero_centered_matrix[customer_index][int(customer[f'market_{index - 1}']) % 100] = .0
    return target, predict_next_market(zero_centered_matrix, matrix, customer_index, customer,
                                       similar_user_count, recommendation_count)