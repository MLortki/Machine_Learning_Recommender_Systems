# coding: utf-8
import numpy as np
import scipy.sparse as sp
import plots as pl
import sklearn.model_selection as skm
import pandas as pd
import csv
from helpers import calculate_mse, load_data
from helpers import build_index_groups

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
            
            #ind = abs(pred - np.around(pred)) <= 0.1
            #pred[ ind ] = np.around(pred[ ind ])
            
            _id = "r{0}_c{1}".format(row+1,cols[i]+1)
            writer.writerow({'Id': _id, 'Prediction': valid_rating})
            
def write_predictions(path_output, ratings):
    np.save(path_output, ratings)

def write_predictions_csv(path_output, ratings):
    (rows, cols, data) = sp.find(ratings)
    fieldnames = ['Id', 'Prediction']
    with open(path_output, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i,row in enumerate(rows):
            _id = "r{0}_c{1}".format(row+1,cols[i]+1)
            writer.writerow({'Id': _id, 'Prediction': data[i]})
    print('Saved predictions at',path_output) 

def split_data(ratings,  p_test=0.1, sparse= True):
    """
    split the ratings to training data and test data.
    """
    min_num_ratings = 0
    num_items_per_user, num_users_per_item = pl.plot_raw_data(ratings, False)
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
    if sparse:
        test = sp.lil_matrix(ratings.shape)
        train = sp.lil_matrix(ratings.shape)
    else:
        test = np.empty(ratings.shape)
        train = np.empty(ratings.shape)
        
    for i,j,v in zip(cx.row, cx.col, cx.data):
        # put the element with probability 0.1 in test set.
        if (np.random.uniform()<p_test):
            test[i,j] = v
        # put the element with probability 0.9 in train set.
        else:
            train[i,j] = v

    (rows, cols, datas) = sp.find(valid_ratings)

    if sparse:
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

def create_dataset_blending(path='../data/data_train.csv'):
    ratings = load_data(path)
    __, train, test = split_data(ratings,0.05)
    __, test_test, test_validation = split_data(test, 0.5)
    file_blending_train = '../data/blending_train.csv'
    file_blending_test = '../data/blending_test.csv'
    file_blending_validation = '../data/blending_validation.csv'
    write_predictions_csv(file_blending_test, test_test)
    write_predictions_csv(file_blending_validation, test_validation)
    write_predictions_csv(file_blending_train, train)
    print('number of non-zero elements for train(95%):{}, validation(2.5%):{}, \
            test(2.5%):{}'.format(train.nnz, test_validation.nnz,test_test.nnz))

def create_dataset_surprise(path='../data/data_train.csv',output_path = \
        '../data/data_train_surprise.csv'):
    ratings = load_data(path)
    rows, cols, ratings = sp.find(ratings)
    rows = rows + 1
    cols = cols + 1
    test_pd = pd.DataFrame({'item':rows,'user':cols,'rating':ratings})
    test_pd.to_csv(output_path,index=False)


if __name__=="__main__":
    setup = '''
from __main__ import compute_error2, compute_error2_slow
from helpers import load_data
import scipy.sparse as sp
path_dataset = '../data/data_train.csv'
ratings = load_data(path_dataset)
rows, cols, r = sp.find(ratings)
'''
    from timeit import Timer
    print(min(Timer('compute_error2(ratings,ratings,zip(rows,cols))',setup=setup).repeat(1,3)))
