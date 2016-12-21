"""Alternating least squares with its helper functions"""

import numpy as np
import scipy.sparse as sp
import sklearn.model_selection as skm
from ALS import ALS_WR


def gen_train_test(ratings, nz_ratings, i_ind , j_ind):
    """generating train and test split from ratings.
    
       input:   ratings          -data matrix (D x N)
                nz_ratings       -list of indices of nonzero elements in ratings
                i_ind            -indices of nz_ratings that should go into test split
                j_ind            -indices of nz_ratings that should go into train split
                
       output:  train            -training data
                test             -validation data
    """    
        
    #estimating shape
    d = ratings.shape[0]
    n = ratings.shape[1] 
    
    #creating list of indices from ratings for traning and validating
    train_index = [ nz_ratings[i] for i in i_ind]
    test_index = [ nz_ratings[j] for j in j_ind]

    #initializing training and validating data matrices
    train = np.zeros((d,n )) 
    test  = np.zeros((d,n))

    for ind in train_index:
        i = ind[0]
        j = ind[1]
        train[i,j] = ratings[i,j]
        
    for ind in test_index:
        i = ind[0]
        j = ind[1]
        test[i,j] = ratings[i,j]
        
    #creating sparse matrices    
    train = sp.lil_matrix(train)
    test = sp.lil_matrix(test)
        
    return train, test





def cross_validation(
    ratings, n_of_splits,num_features,lambdas, stop_criterion, check_nb = None):
    """generating train and test split from ratings.
    
       input:   ratings            -data matrix (D x N)
                n_of_splits        -number of splits
                num_features       -number of latent features
                lambdas            -regularization parameter
                stop_criterion     -threshold
                check_nb           -number of iterations, if specified
                
       output:  test_avg_cost      -list of average validation cost of each regularization parameter
                train_avg_cost     -list of average training cost of each regularization parameter
                errors             -list of errors for every iteration for each regularizations parameter
    """  

   
    #initializing 
    kf = skm.KFold(n_splits=n_of_splits, shuffle = True)
    
    #creating matrix of non-zero indices
    nz_row, nz_col = ratings.nonzero()
    nz_ratings = list(zip(nz_row, nz_col))
   
    #number of nonzero elements
    t = range(nz_row.shape[0])
   
    # creating matrix where results of cross validation are stored
    test_avg_cost =  np.zeros(len(lambdas))
    train_avg_cost = np.zeros(len(lambdas))
    
    #creating list for storing test rmse
    errors = []

     
    for ind,lambda_ in enumerate(lambdas):
            
        #number of regularizations parameter being checked
        print(ind+1, "/",len(lambdas))
     
            
        avg_train = 0
        avg_test = 0
            
        print("lambda  = ",lambda_)

        cnt = 1
        
        #for each split run ALS and estimate train and validation error
        for i, j in kf.split(t):
            
            #if check_nb specified and enough splits checked, break
            if check_nb != None and cnt > check_nb:
                break
            
            #create train and validation split from ratings
            train , test = gen_train_test(ratings, nz_ratings, i , j)
            
            #run Alternating least squares with regularized weights
            itf, usf, rmses_tr, rmses_te = ALS_WR (train , test, num_features,lambda_, 
                                               stop_criterion)
                
            avg_train += rmses_tr[-2]
            avg_test += rmses_te[-2]
                      
             
            cnt += 1
                    
                
                
        avg_train /= min(n_of_splits, check_nb)
        avg_test /= min(n_of_splits, check_nb)
        
        print("average train error = ",avg_train)
        print("average test error = ",avg_test)
        
        test_avg_cost[ind] = avg_test
        train_avg_cost[ind] = avg_train
        errors.append(rmses_te)
    
    return test_avg_cost, train_avg_cost, errors
            
