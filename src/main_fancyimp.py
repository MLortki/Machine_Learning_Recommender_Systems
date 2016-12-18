import fancyimpute as fi

from helpers import load_data, preprocess_data
from plots import plot_raw_data
import our_helpers as ohe
import plots as pl
import scipy
import scipy.io
import scipy.sparse as sp

path_dataset = "../data/data_train.csv"
ratings, data = load_data(path_dataset)
data = sp.lil_matrix(data)
num_items_per_user, num_users_per_item = plot_raw_data(data, False)


valid_ratings, train, test, stest = ohe.split_data(
    ratings, num_items_per_user, num_users_per_item, min_num_ratings=0, p_test=0.1)

print(data.shape)
nz_rows, nz_cols = stest.nonzero()
nz_test = list( zip(nz_rows, nz_cols))
train[train == 0 ] = 'nan'
X_filled_nnm_1 = fi.NuclearNormMinimization().complete(train)
error = ohe.compute_error2(test, X_filled_nnm_1, nz_test)
print(error)
