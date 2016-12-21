"""Alternating least squares with its helper functions"""

import numpy as np
import scipy.sparse as sp
import sklearn.model_selection as skm
from ALS import ALS_WR


def gen_train_test(ratings, nz_ratings, i_ind , j_ind):
    
    #estimating shape
    d = ratings.shape[0]
    n = ratings.shape[1] 
    
    
    train_index = [ nz_ratings[i] for i in i_ind]
    test_index = [ nz_ratings[j] for j in j_ind]

    #print("train index size = ", len(train_index))
    #print("test index size = ", len(test_index))

    
    if len(train_index) + len(test_index) != len(nz_ratings):
        print("Wrong !!")

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
        
    train = sp.lil_matrix(train)
    test = sp.lil_matrix(test)
    
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    
    #print("nonzero elems in train: ", len(nz_train))
    #print("nonzero elems in test: ", len(nz_test))
    
    return train, test





def cross_validation(
    ratings, n_of_splits,num_features,lambdas,
    stop_criterion, check_nb):

   
    #cross_validation(n_splits)
    kf = skm.KFold(n_splits=n_of_splits, shuffle = True)
    
    #creating matrix of non-zero indices
    nz_row, nz_col = ratings.nonzero()
    nz_ratings = list(zip(nz_row, nz_col))
    
    #print("Length of list of nonzero elems of ratings is : ", len(nz_ratings) )
    
    t = range(nz_row.shape[0])
    
    #t = nz_row.shape[0]
    #nz_row = nz_row.reshape(t,1)
    #nz_col = nz_col.reshape(t,1)
    #nz_ratings = np.concatenate((nz_row, nz_col), axis = 1)
    
    #print(nz_row.shape)
    #print(nz_col.shape)
    #print(nz_ratings.shape)
    
    # creating matrix where results of cross validation are stored
    test_avg_cost =  np.zeros(len(lambdas))
    train_avg_cost = np.zeros(len(lambdas))
    errors = []

     
    for ind,lambda_ in enumerate(lambdas):
            
        print(ind+1, "/",len(lambdas))
     
            
        avg_train = 0
        avg_test = 0
            
        print("lambda  = ",lambda_)

        cnt = 1
        
        for i, j in kf.split(t):
            
            if cnt > check_nb:
                break
            
            #print("train: ",i.shape)
            #print("test: ",j.shape)
            #print(j)
            #print("Percentage of train and test reps. : ", 
                  #len(i)*100/nz_row.shape[0],"% ", len(j)*100/nz_row.shape[0],"%" )
            
            train , test = gen_train_test(ratings, nz_ratings, i , j)
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
            
