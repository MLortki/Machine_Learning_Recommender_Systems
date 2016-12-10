# coding: utf-8
import numpy as np

def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
    
    (num_item, num_user) = train.shape
    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)
    
    return user_features, item_features

def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    x = np.dot(item_features.T, user_features)
    mse = 0
    for d, n in nz:
        mse += (data[d, n] - x[d, n])**2
    return np.sqrt(mse/len(nz))

def gradients(train, item_features, user_features, nz_train, lambda_user=0, lambda_item=0):
    D = item_features.shape[1]
    N = user_features.shape[1]
    num_features = item_features.shape[0]
    X = np.dot(item_features.T, user_features)
    grad_item = np.zeros((D, num_features))
    grad_user = np.zeros((N, num_features))
    for d, n in nz_train:
        err = train[d,n] - X[d,n]
        grad_item[d,:] += -err * user_features[:,n] + lambda_item*item_features[:,d]
        grad_user[n,:] += -err * item_features[:,d] + lambda_user*user_features[:,n]
    return grad_item, grad_user

def matrix_factorization_sgd(train, gamma, num_features, lambda_user,
        lambda_item, num_epochs):
    """matrix factorization by SGD."""
    errors = [0]
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    
    print("learn the matrix factorization using SGD...")
    D = train.shape[0]
    N = train.shape[1]
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        (grad_item, grad_user) = gradients(train, item_features, user_features, nz_train)
        user_features += -gamma * grad_user.T
        item_features += -gamma * grad_item.T
        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        errors.append(rmse)
    return user_features, item_features

