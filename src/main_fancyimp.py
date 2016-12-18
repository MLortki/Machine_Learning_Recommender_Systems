import fancyimpute as fi

from helpers import load_data, preprocess_data
from plots import plot_raw_data
import our_helpers as ohe
import plots as pl

path_dataset = "../data/data_train.csv"
ratings = load_data(path_dataset)
num_items_per_user, num_users_per_item = plot_raw_data(ratings, False)
valid_ratings, train, test = ohe.split_data(
    ratings, num_items_per_user, num_users_per_item, min_num_ratings=0, p_test=0.1)

nz_rows, nz_cols = test.nonzero()
nz_test = list( zip(nz_rows, nz_cols))

X_filled_nnm = fi.SoftImpute().complete(ratings)
error = ohe.compute_error2(test, X_filled_nnm, nz_test)
print(error)