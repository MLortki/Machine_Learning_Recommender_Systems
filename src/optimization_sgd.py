# coding: utf-8
"""Stochastic gradient descent with its helper functions"""

import numpy as np
from costs import compute_error
from helpers import build_index_groups


def init_MF(train, num_features):
    """init the parameter for SGD.
    
       input:   train          -data matrix (D x N)
                num_features   -Number of latent features for matrix factorization (K)
                
       output:  user_features  -user features matrix  (K x N)
                item_features  -movie features matrix (K x D)  
    """
    
    
    (num_item, num_user) = train.shape
    
    nz_train, nz_row_colindices, nz_col_rowindices = build_index_groups(train)
    
    #initialize user_features and item_features with random numbers
    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)
    
    #item_features[0, :] = train[train != 0].mean(axis = 1)
    #user_features[0, :] = train[train != 0].mean(axis = 0)
    
    for i, value in nz_row_colindices:
        item_features[0, i] = train[i, value ].mean()
        
    for i, value in nz_col_rowindices:
        user_features[0, i] = train[ value,i ].mean()
    
    return user_features, item_features


def gradients(train, item_features, user_features, nz_train, lambda_user=0, lambda_item=0):
    """compute gradient for SGD.
    
       input:   train           -data matrix (D x N)
                item_features   -movie features matrix (K x D)
                user_features   -user features matrix  (K x N)
                nz_train        -list of nonzero indices in train
                lambda_user     -regularization parameter for users
                lambda_item     -regularization parameter for movies
                
       output:  grad_item       -gradient for movies latent features matrix
                grad_user       -gradient for users latent features matrix
    """
    
    #initializing dimension and number of latent features
    D = item_features.shape[1]
    N = user_features.shape[1]
    num_features = item_features.shape[0]
    
    #creating predictions
    X = np.dot(item_features.T, user_features)
    
    #initializing gradients for user and item
    grad_item = np.zeros((D, num_features))
    grad_user = np.zeros((N, num_features))
    
    #computing gradients
    for d, n in nz_train:
        err = train[d,n] - X[d,n]
        grad_item[d,:] += -err * user_features[:,n] + lambda_item*item_features[:,d]
        grad_user[n,:] += -err * item_features[:,d] + lambda_user*user_features[:,n]
        
    return grad_item, grad_user

def matrix_factorization_sgd(train, test, gamma, num_features, lambda_user,
        lambda_item, num_epochs):
    """compute gradient for SGD.
    
       input:   train           -test  data (D x N)
                test            -train data (D x N)
                gamma           -step size
                num_features    -number of latent features
                lambda_user     -regularization parameter for users
                lambda_item     -regularization parameter for item
                num_epochs      -number of iterations
                
       output:  user_features   -user features matrix  (K x N)
                item_features   -movie features matrix (K x D)
                errors_tr       -list of traning errors
                errors_te       -list of test errors           
    """
    
    #initializing lists for storing train and test error
    errors_tr = []
    errors_te = []
    
    #set seed
    np.random.seed(988)

    #init matrix
    user_features, item_features = init_MF(train, num_features)
    
    #find the non-zero train indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    
    #find the non-zero test indices 
    nz_row, nz_col = test.nonzero()
    nz_test= list(zip(nz_row, nz_col))
    
    print("learn the matrix factorization using SGD...")
    
    D = train.shape[0]
    N = train.shape[1]
    
    for it in range(num_epochs):      
        
        #shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        #decrease step size
        if it > 1 and abs( errors_tr[-1] - errors_tr[-2]  < 1e-3):
            gamma /= 1.2
            
        #update gradients 
        grad_item, grad_user = gradients(train, item_features, user_features, nz_train)
        user_features -= gamma * grad_user.T
        item_features -= gamma * grad_item.T
        
        
        rmse = compute_error(test, user_features, item_features, nz_test)
        print("iter: {}, RMSE on test set: {}.".format(it, rmse))
        errors_te.append(rmse)
        
        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        errors_tr.append(rmse)
       
        
    return user_features, item_features, errors_tr, errors_te
