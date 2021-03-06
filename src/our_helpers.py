""" Helper functions used in various other files."""
# coding: utf-8

import numpy as np
import scipy.sparse as sp
import plots as pl
import pandas as pd
import csv
from helpers import calculate_mse, load_data
from helpers import build_index_groups


def create_submission(path_output, ratings):
    """creating .csv submission.
    
       input:   path_output     -path for output
                ratings         -prediction matrix (D x N)
    """
    
    #path to the sample submission
    path_sample = "../data/sampleSubmission.csv"
    
    #estimating ratings for submission
    ratings_nonzero  = load_data(path_sample)
    (rows, cols, data) = sp.find(ratings_nonzero)
    
    #writing to file
    fieldnames = ['Id', 'Prediction']
    with open(path_output, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i,row in enumerate(rows):
            valid_rating = min(max(ratings[row,cols[i]],1),5) 
            _id = "r{0}_c{1}".format(row+1,cols[i]+1)
            writer.writerow({'Id': _id, 'Prediction': valid_rating})
       
    
def write_predictions(path_output, ratings):
    """saving predictions as .npy file for final blending.
    
       input:   path_output     -path for output
                ratings         -prediction matrix (D x N)
    """
    np.save(path_output, ratings)

def write_predictions_csv(path_output, ratings):
    """saving predictions as .csv file.
    
       input:   path_output     -path for output
                ratings         -prediction matrix (D x N)
    """

    (rows, cols, data) = sp.find(ratings)
    fieldnames = ['Id', 'Prediction']
    with open(path_output, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i,row in enumerate(rows):
            _id = "r{0}_c{1}".format(row+1,cols[i]+1)
            writer.writerow({'Id': _id, 'Prediction': data[i]})
            counter += 1
    print('Saved {} predictions at {}'.format(counter,path_output)) 

def split_data(ratings,  p_test=0.1, sparse= True):
    """split the ratings to training data and test data..
    
       input:   ratings         -prediction matrix (D x N)
                p_test          -ration of test data
                
       output:  valid_ratings   -ratings
                test            -test data
                train           -train data
    """
    
    #estimatin number of rated movies per user and number of ratings per movie
    num_items_per_user, num_users_per_item = pl.plot_raw_data(ratings, False)
    
    # set seed
    np.random.seed(988)


    #get nonzero element
    cx = sp.coo_matrix(ratings)

    test = sp.lil_matrix(ratings.shape)
    train = sp.lil_matrix(ratings.shape)
  
        
    for i,j,v in zip(cx.row, cx.col, cx.data):
        
        # put the element with probability 0.1 in test set.
        if (np.random.uniform()<p_test):
            test[i,j] = v
            
        # put the element with probability 0.9 in train set.
        else:
            train[i,j] = v

    (rows, cols, datas) = sp.find(ratings)

 
    print("Percentage of nz train data: % 2.4f, percentage of nz test data: % \
            2.4f" % (train.nnz/ratings.nnz, test.nnz/ratings.nnz))
    assert (train.nnz + test.nnz) == ratings.nnz, "Number of nnz elements in test and train test doesn't sum up!"
    
    return ratings, train, test


def create_dataset_blending(path='../data/data_train.csv'):
    """ split training set into three parts (test, train, probe)
        to be used by blending.
    
        input:   path            -path to dataset
    """
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
    """ create dataset that can be read in by surprise library. 

        input:  path            -path to dataset
                output_path     -path to where dataset should be saved 
    
    """  
    ratings = load_data(path)
    rows, cols, ratings = sp.find(ratings)
    rows = rows + 1
    cols = cols + 1
    test_pd = pd.DataFrame({'item':rows,'user':cols,'rating':ratings})
    test_pd.to_csv(output_path,index=False)

def bias_correction (full_ratings, test):
    """ shift the obtained ratings matrix by the mean difference between its 
    mean and the test dataset's mean.

        input:  full_ratings         -estimated sparse matrix (D X N), to be corrected
                test                 -test sparse matrix (D X N) with correct ratings

        output: full_ratings         -corrected sparse matrix (D X N)
    """
    
    nz_rows, nz_cols = test.nonzero()
    nz_test = list( zip(nz_rows, nz_cols))
    
    mean_te = 0
    mean_pr = 0
    for i,j in nz_test:
        mean_te += test[i,j]
        mean_pr += full_ratings[i,j]
        
    mean_te /= len(nz_test)
    mean_pr /= len(nz_test)
    
    #mean_pr= full_ratings.mean()
    full_ratings += (mean_te - mean_pr)
    
    return full_ratings
