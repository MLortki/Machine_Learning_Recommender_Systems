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



    
#initialization
sratings = sp.lil_matrix(ratings)
train_errors = []
test_errors = []
    
lambdas = np.linspace(lambda_start, lambda_end , number_of_lambdas)
#lambdas = [0.06255]
    
print("number of different lambdas : ",len(lambdas))
    
# set seed
np.random.seed(988)
    
    
test_avg_cost, train_avg_cost , errors = ohe.cross_validation(
sratings, n_splits, num_features, lambdas, stop_criterion, cv_iter)
    
#generating plot
path = "/K%d/l%d_%f_%f.jpg"%(num_features, len(lambdas),lambdas[0],lambdas[-1] )
    
pl.plot_cv_errors(errors, lambdas, num_features, path)
