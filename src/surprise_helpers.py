from surprise import Dataset, Reader
import pickle
import pandas as pd
from surprise_ML_methods import *

def load_data(file_path):
    """
        Load data from file path
    
        input:   file_path      -The input file path
                
        output:  data frame     -The loaded data frame from file path
    """
    reader = Reader(line_format='item rating user', sep=',',skip_lines=1) 
    return Dataset.load_from_file(file_path, reader=reader)



def load_folds_data(fold_file_paths):
    """
        Load folded data from folded file path
    
        input:   fold_file_paths  -The file paths of folded data
                
        output:  fold data frame  -The loaded data frames from folded file path
    """
    reader = Reader(line_format='item rating user', sep=',',skip_lines=1) 
    return Dataset.load_from_folds(fold_file_paths, reader=reader)



def create_submission_dataframe(df_simple):
    """
        Convert a data frame in simple format to a data framein submission format
      
        input:   df_simple      -Data frame in simple format
                
        output:  df_submission  -Data frame in submission  format
    """
    df_simple["Id"] = "r" + df_simple["iid"].map(str) + "_c" +df_simple["uid"].map(str)
    df_simple["Prediction"] = df_simple["est"].clip(0,5)
    df_submission = df_simple.drop(["iid","uid","est","details","rui"],1)
    return df_submission



def create_submition_csv(prediction, output_path):
    """save final predictions in output file in csv format
    
       input:   prediction      -The final prediction
                output_path     -The submission file path
       
       output:  --              -- 
    """
    
    df_svd = pd.DataFrame(prediction, columns=['uid', 'iid', 'rui', 'est', 'details'])    
    df_svd_submission = create_submission_dataframe(df_svd)
    df_svd_submission.to_csv(output_path, columns=["Id","Prediction"],index=False)




def run_and_save(algo, fold_files, output_path):
    """ 
        Learn the model on trainset based on given algorithm, 
        predict results on testset, 
        save the predictions in output file 
    
        input:   algo            -Learning algorithm
                fold_filse      -List of file paths of (trainset, testset)
                output_path     -Output file path 
       
        output:  mean_rmse       -The mean of rmse on (train_data, test_data) in fold_data
       
    """
    
    #load train_data and test_data from fold_files list
    fold_data=load_folds_data(fold_files)
    #do cross validation on (train_data, test_data) and compute rmse and predictions
    mean_rmse, prediction = cross_validation(algo, fold_data)
    #save prediction in output file path
    create_submition_csv(prediction, output_path)
    return mean_rmse


def run_all_algorithm(fold_files, output_prefix):
    """ 
        Learn and predict result using diffrenet ML methods
    
       input:   
                fold_filse      -List of file paths of (trainset, testset)
                output_prefix   -The prefix of output file path 
       
       output:  rmse_list       -The list of rmse on (train_data, test_data) in fold_data using different ML methods
       
    """
    
    rmse_list = []

    """rmse_list.append(run_and_save(ALS_BaselineOnly(), fold_files, "../submission/"+output_prefix+"ALS_BaselineOnly.csv"))
    rmse_list.append(run_and_save(SGD_BaselineOnly(), fold_files, "../submission/"+output_prefix+"SGD_BaselineOnly.csv"))
    
    rmse_list.append(run_and_save(KNNBasic_ALS_pearson_baseline_user_based(), fold_files, "../submission/"+output_prefix+"KNNBasic_ALS_pearson_baseline_user_based.csv"))
    rmse_list.append(run_and_save(KNNBasic_ALS_pearson_baseline_item_based(), fold_files, "../submission/"+output_prefix+"KNNBasic_ALS_pearson_baseline_item_based.csv"))
    
    rmse_list.append(run_and_save(KNNBasic_pearson_baseline_user_based(), fold_files, "../submission/"+output_prefix+"KNNBasic_pearson_baseline_user_based.csv"))
    rmse_list.append(run_and_save(KNNBasic_pearson_baseline_item_based(), fold_files, "../submission/"+output_prefix+"KNNBasic_pearson_baseline_item_based.csv"))
    rmse_list.append(run_and_save(KNNBasic_cosine_user_based(), fold_files, "../submission/"+output_prefix+"KNNBasic_cosine_user_based.csv"))
    rmse_list.append(run_and_save(KNNBasic_cosine_item_based(), fold_files, "../submission/"+output_prefix+"KNNBasic_cosine_item_based.csv"))
    rmse_list.append(run_and_save(KNNBasic_pearson_user_based(), fold_files, "../submission/"+output_prefix+"KNNBasic_pearson_user_based.csv"))
    rmse_list.append(run_and_save(KNNBasic_msd_user_based(), fold_files, "../submission/"+output_prefix+"KNNBasic_msd_user_based.csv"))
    rmse_list.append(run_and_save(KNNBasic_msd_item_based(), fold_files, "../submission/"+output_prefix+"KNNBasic_msd_item_based.csv"))
    
    rmse_list.append(run_and_save(KNNWithMeans_pearson_baseline_user_based(), fold_files, "../submission/"+output_prefix+"KNNWithMeans_pearson_baseline_user_based.csv"))
    rmse_list.append(run_and_save(KNNWithMeans_pearson_baseline_item_based(), fold_files, "../submission/"+output_prefix+"KNNWithMeans_pearson_baseline_item_based.csv"))
    rmse_list.append(run_and_save(KNNWithMeans_cosine_user_based(), fold_files, "../submission/"+output_prefix+"KNNWithMeans_cosine_user_based.csv"))
    rmse_list.append(run_and_save(KNNWithMeans_cosine_item_based(), fold_files, "../submission/"+output_prefix+"KNNWithMeans_cosine_item_based.csv"))
    rmse_list.append(run_and_save(KNNWithMeans_pearson_user_based(), fold_files, "../submission/"+output_prefix+"KNNWithMeans_pearson_user_based.csv"))
    rmse_list.append(run_and_save(KNNWithMeans_msd_user_based(), fold_files, "../submission/"+output_prefix+"KNNWithMeans_msd_user_based.csv"))
    rmse_list.append(run_and_save(KNNWithMeans_msd_item_based(), fold_files, "../submission/"+output_prefix+"KNNWithMeans_msd_item_based.csv"))"""


    rmse_list.append(run_and_save(KNNBaseline_pearson_baseline_user_based(), fold_files, "../submission/"+output_prefix+"KNNBaseline_pearson_baseline_user_based.csv"))
    rmse_list.append(run_and_save(KNNBaseline_pearson_baseline_item_based(), fold_files, "../submission/"+output_prefix+"KNNBaseline_pearson_baseline_item_based.csv"))
    rmse_list.append(run_and_save(KNNBaseline_cosine_user_based(), fold_files, "../submission/"+output_prefix+"KNNBaseline_cosine_user_based.csv"))
    rmse_list.append(run_and_save(KNNBaseline_cosine_item_based(), fold_files, "../submission/"+output_prefix+"KNNBaseline_cosine_item_based.csv"))
    rmse_list.append(run_and_save(KNNBaseline_pearson_user_based(), fold_files, "../submission/"+output_prefix+"KNNBaseline_pearson_user_based.csv"))
    rmse_list.append(run_and_save(KNNBaseline_msd_user_based(), fold_files, "../submission/"+output_prefix+"KNNBaseline_msd_user_based.csv"))
    rmse_list.append(run_and_save(KNNBaseline_msd_item_based(), fold_files, "../submission/"+output_prefix+"KNNBaseline_msd_item_based.csv"))
    
    return rmse_list



