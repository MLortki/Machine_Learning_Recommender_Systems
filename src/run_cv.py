"""update user feature matrix.
    
   input N:   0               -path to the run_cv.py
              1               -starting lambda
              2               -ending lambda
              3               -number of lambdas
              4               -number of cv iterations
     
                
   output :  plot1             -test error for each lambda
             plot2             -train and avg.test for each lambda
                                !! folder K{argv[1]} should exist in results folder
   
   exampl.:  python3 
"""

import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.model_selection as skm
from helpers import load_data, build_index_groups
from CV import cross_validation
import data_preprocess as dp
import plots as pl
import sys



if len(sys.argv) != 6:
    
    print("usage: program_name K min_lambda max_lambda number_of_lambdas")
    exit()

print("reading data")
path_dataset = "../data/data_train.csv"
ratings = load_data(path_dataset)

num_items_per_user, num_users_per_item = pl.plot_raw_data(ratings, False)


#read parameters
num_features = int(sys.argv[1])   # K in the lecture notes
lambda_start = float(sys.argv[2]) 
lambda_end = float(sys.argv[3] )
number_of_lambdas = int (sys.argv[4])
cv_iter = int (sys.argv[5])
stop_criterion = 1e-4
n_splits = 10

#remove biases
nz_ratings, nz_row_colindices, nz_col_rowindices = build_index_groups(ratings)
user_means = dp.get_user_means(ratings.copy(), nz_col_rowindices) 
item_means = dp.get_item_means(ratings.copy(), nz_row_colindices)
means, mean = dp.get_global_means(ratings.copy(), nz_ratings) 
ratings_normalized, train_mean = dp.get_unbiased_matrix(ratings.copy(),user_means,item_means,means,'combined')


    
#initializing train errors and test errors list
train_errors = []
test_errors = []

#initializing lambdas  
lambdas = np.linspace(lambda_start, lambda_end , number_of_lambdas)
    
print("number of different lambdas : ",len(lambdas))
    
#set seed
np.random.seed(988)
    
    
test_avg_cost, train_avg_cost , errors = cross_validation(
ratings_normalized, n_splits, num_features, lambdas, stop_criterion, cv_iter)
    
#generating plot, test error for each lambda
path = "/K%d/l%d_%f_%f.jpg"%(num_features, len(lambdas),lambdas[0],lambdas[-1] )
pl.plot_cv_errors(errors, lambdas, num_features, path)

#generating plot, avg. train and avg.test for each lambda
path ="K%d/different_lambdas"%num_features
pl.plot_cv_train_test(test_avg_cost, train_avg_cost, lambdas, path)
    
    