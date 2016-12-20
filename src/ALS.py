"""Alternating least squares with its helper functions"""

import numpy as np
from helpers import build_index_groups
from costs import compute_error


def init_MF(train, num_features):
    """init the parameter for matrix factorization.
    
       input:   train          -data matrix (D x N)
                num_features   -Number of latent features for matrix factorization (K)
                
       output:  user_features  -user features matrix  (K x N)
                item_features  -movie features matrix (K x D)  
    """
    
    #initialize user_features and item_features with random numbers	
    user_features = 3 * np.random.rand( num_features, train.shape[1])
    item_features = 3 * np.random.rand( num_features, train.shape[0])
    #set first row of item_features to the average value of each movie
    item_features[0, :] = train[train != 0].mean(axis = 1)
    
    return user_features, item_features

def update_user_feature(
        train, item_features, lambda_user, nnz_items_per_user):   
    """update user feature matrix.
    
       input:   train               -data matrix (D x N)
                item_features       -movie features matrix (K x D) 
                lambda_user         -regularizations parameter
                nnz_items_per_user  -indices of nonzero movie ratings of each user
                
       output:  user_features       -user features matrix  (K x N) 
    """
    
    #estimating dimensions
    N = train.shape[1]
    D = train.shape[0]
    K = item_features.shape[0]
    
    #initializing user features matrix
    user_features = np.zeros((K, N))
    
    #updating user features matrix column by column
    for g , value in nnz_items_per_user:
    
        #retrieving nonzero movie ratings of g+1-th user
        X = train[value, g].toarray()
        #retrieving columns corresponding to the above mentioned nonzero movie ratings
        Wt = item_features[: , value]
        
        #updating g-th column of user_features according to ALS_WR
        user_features[:, g] = np.linalg.solve( 
            np.dot(Wt, np.transpose(Wt))+ lambda_user* value.shape[0]* np.eye(K) ,
            np.dot( Wt, X)).flatten()
      
    return user_features

def update_item_feature(
        train, user_features, lambda_item, nnz_users_per_item):  
    """update item feature matrix.
    
       input:   train               -data matrix (D x N)
                user_features       -movie features matrix (K x N) 
                lambda_item         -regularizations parameter
                nnz_users_per_item  -indices of users that rated each movie 
                
       output:  item_features       -user features matrix  (K x D) 
    """
 
    #estimating dimensions
    N = train.shape[1]
    D = train.shape[0]
    K = user_features.shape[0]
        
    #initializing item features matrix
    item_features = np.zeros((K, D))
   
    #updating item features matrix column by column
    for g, value  in nnz_users_per_item:
        
        #retrieving ratings received by the g-th movie
        X = train[g,value].toarray()
        #retrieving columns corresponding to users who rated g+1-th movie
        Zt = user_features[: , value]
 
        #updating g-th column of item_features according to ALS_WR
        item_features[:, g] = np.linalg.solve( 
            np.dot(Zt, np.transpose(Zt))+ lambda_item * value.shape[0]* np.eye(K)  ,
            np.dot( Zt, np.transpose(X))).flatten()
        
    
    return item_features

def ALS_WR(train ,test, num_features,lambda_, stop_criterion,rng = None):
    """Alternating least squares with weighted regularization
    
       input:   train               -training data matrix   (D x N)
                test                -validation data matrix (D x N)
                num_features        -number of latent features (K)
                lambda_             -regularizations parameter
                stop_criterion      -threshold  
                rng                 -number of interations if specified
                
       output:  item_features       -final item_features after ALS_WR 
                user_features       -final user_features after ALS_WR 
                rmses_tr            -list of training   errors of every iteration
                rmses_te            -list of validating errors of every iteration
    """
    
    # initialize user and movie latent features matrices
    user_features, item_features = init_MF(train, num_features)
    
    #indices of nonzero  elements of traning data
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    
    #building index groups
    nz_train, nz_row_colindices, nz_col_rowindices = build_index_groups(train)   
   
    
       
    #initialize counter, and list for storing test and train errors
    i = 0
    rmses_te = []
    rmses_tr = []
                
    while True:
    
        #if rng specified and we already did enough iterations, break
        if rng != None and i >= rng:
            break
    
        #updating user_features
        user_features_new = update_user_feature(
            train, item_features, lambda_, nz_col_rowindices)
      
        #updating item_features
        item_features_new = update_item_feature(
            train, user_features_new, lambda_, nz_row_colindices)

        #computing train and validation error of new feature matrices
        rmse_te = compute_error(test, user_features_new, item_features_new , nz_test)
        rmse_tr = compute_error(train, user_features_new, item_features_new , nz_train)
        
     
        print("iter: {}, RMSE on training set: {}.".format(i, rmse_tr))
        print("iter: {}, RMSE on test set: {}.".format(i, rmse_te))
        
     
            
        if   i > 0 and abs(rmses_tr[-1] - rmse_tr) < stop_criterion : #or i > 0 and rmse_te > rmses_te[len(rmses_te)-1] :
        #if   i > 0 and abs(rmses_te[-1] - rmse_te) < stop_criterion or i > 0 and rmse_te > rmses_te[-1] :
            rmses_tr.append(rmse_tr)
            rmses_te.append(rmse_te)
            break
        
        rmses_tr.append(rmse_tr)
        rmses_te.append(rmse_te)

        item_features = item_features_new
        user_features = user_features_new
        
        i += 1
    
    
    rmse_te = compute_error(test, user_features, item_features, nz_test)
    rmse_tr = compute_error(train, user_features, item_features, nz_train)
    print("Final RMSE on test data: {}.".format(rmse_te))

    return item_features, user_features,  rmses_tr, rmses_te 
