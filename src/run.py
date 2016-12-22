import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
import sklearn.model_selection as skm
from helpers import load_data, preprocess_data
from plots import plot_raw_data
import data_preprocess as dp
import helpers as h
import ALS_WR as ALS
from our_helpers import create_submission


print("reading data")
#load data
path_dataset = "../data/data_train.csv"
ratings = load_data(path_dataset)
print("shape of dataset:",ratings.shape)

#get number of items per user and nuber of users per item
num_items_per_user, num_users_per_item = plot_raw_data(ratings, False)

#get indices of nonzero elements
nz_ratings, nz_row_colindices, nz_col_rowindices = h.build_index_groups(ratings)

print("removing biases")
#removing biases
user_means = dp.get_user_means(ratings.copy(), nz_col_rowindices) 
item_means = dp.get_item_means(ratings.copy(), nz_row_colindices)
means , mean= dp.get_global_means(ratings.copy(), nz_ratings) 
ratings_normalized, train_mean = dp.get_unbiased_matrix(ratings.copy(),user_means,item_means,means,'combined')

print("ALS with number of laten features = 8, ", '$\lambda$', " = 0.06255, threshold: 1e-4")

#matrix factorization with ALS
item_features, user_features, train_errors, test_errors = ALS_WR(
        ratings_normalized, ratings_normalized,8, 0.06255, 1e-4)

#generating unbiased predictions
ratings_full = np.dot(np.transpose(item_features),user_features)

print("generating biased prediction for submission")
#biased predictions for submission
pred = dp.get_predictions(ratings_full, user_means, item_means, mean, 'combined')

print("creating submission")


path_output = "final_l0.06255_k8_submission.csv"
create_submission(path_output, pred )
