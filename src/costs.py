import numpy as np



def compute_error(data, user_features, item_features, nz_data
    """compute the loss (MSE) of the prediction of nonzero elements."""
    X = np.dot ( np.transpose(item_features), user_features )
    return compute_error(data, X, nz_data

def compute_error(data, pred, nz_data
    rmse = 0
    counter = 0
    for i, j in nz_data
        rmse += (data[i,j] - pred[i,j])**2
        counter += 1
    rmse = rmse/(counter)
    return rmse


