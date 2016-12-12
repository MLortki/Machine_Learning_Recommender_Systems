# coding: utf-8
import numpy as np
import scipy.sparse as sp
import csv
from helpers import calculate_mse, load_data

def create_submission(path_output, ratings):
    path_sample = "../data/sampleSubmission.csv"
    ratings_nonzero  = load_data(path_sample)
    (rows, cols, data) = sp.find(ratings_nonzero)
    fieldnames = ['Id', 'Prediction']
    with open(path_output, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i,row in enumerate(rows):
            valid_rating = min(max(ratings[row,cols[i]],1),5)
            valid_rating = round(valid_rating)
            _id = "r{0}_c{1}".format(row+1,cols[i]+1)
            writer.writerow({'Id': _id, 'Prediction': valid_rating})

def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Args:
        min_num_ratings:
            all users and items we keep must have at least min_num_ratings per user and per item.
    """
    # set seed
    np.random.seed(988)

    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][:, valid_users]

    count_users = valid_ratings.shape[1]
    count_items = valid_ratings.shape[0]
    count_total = count_items * count_users

    # this is supposedly the fastest way of doing this.
    cx = sp.coo_matrix(valid_ratings)
    test = sp.lil_matrix(ratings.shape)
    train = sp.lil_matrix(ratings.shape)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        # put the element with probability 0.1 in test set.
        if (np.random.uniform()<p_test):
            test[i,j] = v
        # put the element with probability 0.9 in train set.
        else:
            train[i,j] = v

    (rows, cols, datas) = sp.find(valid_ratings)

    #  print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    #  print("Train shape:{s}, num:{n}".format(s=train.shape, n=train.shape[0]*train.shape[1]))
    #  print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    #  print("Test shape:{s}, num:{n}".format(s=test.shape, n=test.shape[0]*test.shape[1]))
    #  print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    print("Percentage of nz train data: % 2.4f, percentage of nz test data: % \
            2.4f" % (train.nnz/valid_ratings.nnz, test.nnz/valid_ratings.nnz))
    assert (train.nnz + test.nnz) == valid_ratings.nnz, "Number of nnz elements in test and train test doesn't sum up!"
    return valid_ratings, train, test

def baseline_global_mean(train, test):
    """baseline method: use the global mean."""
    mean = (train!=0).mean()
    # sum over differences between actual values and global mean.
    sum_mse = 0
    num = 0
    # this is supposedly the fastest way of doing this.
    cx = sp.coo_matrix(test)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        sum_mse += (v-mean)**2
        num += 1
    return sum_mse/(num*2.0)

# TODO: Aida's version only working for dense matrices?
def baseline_global_mean_aida(train, test):
    """baseline method: use the global mean."""
    global_predict_train=(train!=0).mean()
    num_test=np.count_nonzero(test)
    test_real_values=test.ravel()[np.flatnonzero(test)]
    test_mse=1/(2.0*num_test)*calculate_mse(test_real_values, np.full(num_test, global_predict_train))
    return test_mse

def baseline_user_mean(train, test):
    """baseline method: use the user means as the prediction."""
    user_mean = (train!=0).mean(axis=0).reshape(-1,1)
    # sum over differences between actual values and global mean.
    sum_mse = 0
    num = 0
    # this is supposedly the fastest way of doing this.
    cx = sp.coo_matrix(test)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        sum_mse += (v-user_mean[j,0])**2
        num += 1
    return sum_mse/(num*2.0)

def baseline_item_mean(train, test):
    """baseline method: use item means as the prediction."""
    item_mean = (train!=0).mean(axis=1).reshape(-1,1)
    # sum over differences between actual values and global mean.
    sum_mse = 0
    num = 0
    # this is supposedly the fastest way of doing this.
    cx = sp.coo_matrix(test)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        sum_mse += (v-item_mean[i,0])**2
        num += 1
    return sum_mse/(num*2.0)

def baseline_combined(train, test):
    mean = (train!=0).mean()
    item_mean = (train!=0).mean(axis=1).reshape(-1,1)
    user_mean = (train!=0).mean(axis=0).reshape(-1,1)

    # this is supposedly the fastest way of doing this.
    sum_mse = 0
    num = 0
    cx = sp.coo_matrix(test)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        b_user = user_mean[j,0] - mean
        b_item = item_mean[i,0] - mean
        prediction = mean + b_user + b_item 
        sum_mse += (v - prediction)**2
        num += 1
    return sum_mse/(num*2.0)
