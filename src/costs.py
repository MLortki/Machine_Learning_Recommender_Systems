"""cost functions"""
import numpy as np
import math


def compute_error(data, user_features, item_features, nz_data):
    """compute the loss (RMSE) of the prediction of nonzero elements.
    
       input:   data            -data matrix           (D x N)
                user_features   -user features matrix  (K x N)
                item_features   -item features matrix  (K x N)
                nz_data         -list of nonzero elements of the data matrix
                
       output:  rmse            -root-mean-square error 
    """     
    #creating predictions matrix
    X = np.dot ( np.transpose(item_features), user_features )
    #computing rmse
    rmse =  compute_error2(data, X, nz_data)
                  
    return rmse

def compute_error2(data, pred, nz_data):
    """compute the loss (RMSE) of the prediction of nonzero elements.
    
       input:   data            -data matrix         (D x N)
                pred            -predictions matrix  (D x N)
                nz_data         -list of nonzero elements of the data matrix
                
       output:  rmse            -root-mean-square error of pred
    """     
                  
    #initializing parameters             
    rmse = 0
    counter = 0
                  
    for i, j in nz_data:
        rmse += (data[i,j] - pred[i,j])**2
        counter += 1
    rmse = math.sqrt(rmse/counter)
                  
    return rmse


