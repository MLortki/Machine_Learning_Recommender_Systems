# name_of_file 
# starting_lambda
# ending_lambda
# number_of_lambdas
# number_of_cv_interations


import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.model_selection as skm
from helpers import load_data, preprocess_data
from plots import plot_raw_data
import our_helpers as ohe
import plots as pl
import sys



if len(sys.argv) != 6:
    
    print("usage: program_name K min_lambda max_lambda number_of_lambdas")

print("reading data")
path_dataset = "../data/data_train.csv"
ratings, data = load_data(path_dataset)

num_items_per_user, num_users_per_item = plot_raw_data(ratings, False)


# define parameters
num_features = int(sys.argv[1])   # K in the lecture notes
lambda_start = float(sys.argv[2]) 
lambda_end = float(sys.argv[3] )
number_of_lambdas = int (sys.argv[4])
cv_iter = int (sys.argv[5])
stop_criterion = 1e-4
n_splits = 10

#removing biases
nz_ratings, nz_row_colindices, nz_col_rowindices = h.build_index_groups(ratings)
user_means = dp.get_user_means(ratings.copy(), nz_col_rowindices) 
item_means = dp.get_item_means(ratings.copy(), nz_row_colindices)
means, mean = dp.get_global_means(ratings.copy(), nz_ratings) 
ratings_normalized, train_mean = dp.get_unbiased_matrix(ratings.copy(),user_means,item_means,means,'combined')


    
#initialization
train_errors = []
test_errors = []

#initializing lambdas  
lambdas = np.linspace(lambda_start, lambda_end , number_of_lambdas)
    
print("number of different lambdas : ",len(lambdas))
    
# set seed
np.random.seed(988)
    
    
test_avg_cost, train_avg_cost , errors = ohe.cross_validation(
sratings, n_splits, num_features, lambdas, stop_criterion, cv_iter)
    
#generating plot
path = "/K%d/l%d_%f_%f.jpg"%(num_features, len(lambdas),lambdas[0],lambdas[-1] )
    
pl.plot_cv_errors(errors, lambdas, num_features, path)