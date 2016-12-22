import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.model_selection as skm
import plots as p
import helpers as h
import data_preprocess as dp
import sys
import our_helpers as ohe






K = int(sys.argv[1])
lambda_ = float(sys.argv[2])


path_dataset = "../data/data_train.csv"
ratings = h.load_data(path_dataset)

num_items_per_user, num_users_per_item = p.plot_raw_data(ratings)

nz_ratings, nz_row_colindices, nz_col_rowindices = h.build_index_groups(ratings)
user_means = dp.get_user_means(ratings, nz_col_rowindices) 
item_means = dp.get_item_means(ratings, nz_row_colindices)
means = dp.get_global_means(ratings, nz_ratings) 

valid_ratings, train, test, stets = ohe.split_data(
    ratings, num_items_per_user, num_users_per_item, min_num_ratings=0, p_test=0.1)

#comparing two methods to each other
methods=['no','global','item','user','combined']

errors_tr = []
errors_te = []

for m in methods:
    test_n = sp.lil_matrix(ratings.shape)
    train_n = sp.lil_matrix(ratings.shape)
    
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    
    nz_row, nz_col = test.nonzero()
    nz_test  = list(zip(nz_row, nz_col))
    
    ratings_normalized, train_mean = dp.get_unbiased_matrix(ratings.copy(),user_means,item_means,means,m)
    
    for i,j in nz_train:
        train_n[i,j] = ratings_normalized[i,j]
        
    for i,j in nz_test:
        test_n[i,j] = ratings_normalized[i,j]


    print("generating item and user feature matrices")
    item_features, user_features, train_errors, test_errors = ohe.ALS(
        train_n, test_n,K, lambda_, 1e-4)
    
    errors_tr.append(train_errors)
    errors_te.append(test_errors)
    
    
x_axis_name = "Number of Epochs"
y_axis_name = "RMSE"
lambda_str = ('%f' % lambda_).rstrip('0').rstrip('.')
title= "ALS-WR Learning Curve, K = %d, lambda = %s"%(K,lambda_str)
labels =[ methods[i] + " bias" for i in range(len(methods))]
path = "Bias/%d_%s.jpg"%(K,lambda_str)
pl.plot_general (errors_te, labels ,x_axis_name, y_axis_name, title, path)