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
import our_helpers as ohe


path_dataset = "../data/data_train.csv"
ratings = load_data(path_dataset)
print("shape of dataset:",ratings.shape)
num_items_per_user, num_users_per_item = plot_raw_data(ratings)


nz_ratings, nz_row_colindices, nz_col_rowindices = h.build_index_groups(ratings)
user_means = dp.get_user_means(ratings.copy(), nz_col_rowindices) 
item_means = dp.get_item_means(ratings.copy(), nz_row_colindices)
means , mean= dp.get_global_means(ratings.copy(), nz_ratings) 

ratings_normalized, train_mean = dp.get_unbiased_matrix(ratings.copy(),user_means,item_means,means,'combined')

item_features, user_features, train_errors, test_errors = ohe.ALS(
        ratings_normalized, ratings_normalized,8, 0.06255, 1e-4)

ratings_full = np.dot(np.transpose(item_features),user_features)

pred = dp.get_predictions(ratings_full, user_means, item_means, mean, 'combined')


