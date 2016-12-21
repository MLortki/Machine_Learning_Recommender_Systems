# -*- coding: utf-8 -*-
"""some functions for plots."""

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


def plot_raw_data(ratings, pl = True):
    """plot the statistics result on raw rating data."""
    # do statistics.
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    sorted_num_movies_per_user = np.sort(num_items_per_user)[::-1]
    sorted_num_users_per_movie = np.sort(num_users_per_item)[::-1]

    if pl:
    # plot
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(sorted_num_movies_per_user, color='blue')
        ax1.set_xlabel("users")
        ax1.set_ylabel("number of ratings (sorted)")
        ax1.grid()

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(sorted_num_users_per_movie)
        ax2.set_xlabel("items")
        ax2.set_ylabel("number of ratings (sorted)")
        #ax2.set_xticks(np.arange(0, 2000, 300))
        ax2.grid()

        plt.tight_layout()
        plt.savefig("../results/stat_ratings")
        plt.show()
        # plt.close()
    return num_items_per_user, num_users_per_item


def plot_train_test_data(train, test):
    """visualize the train and test data."""
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.spy(train, precision=0.01, markersize=0.5)
    ax1.set_xlabel("Users")
    ax1.set_ylabel("Items")
    ax1.set_title("Training data")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.spy(test, precision=0.01, markersize=0.5)
    ax2.set_xlabel("Users")
    ax2.set_ylabel("Items")
    ax2.set_title("Test data")
    plt.tight_layout()
    plt.savefig("../results/train_test")
    plt.show()
    
    
def plot_train_test_errors(train_errors, test_errors, lambda_str , K , path, rng):

    
    plt.plot(range(rng), train_errors, marker='o', label='Training Data');
    plt.plot(range(rng), test_errors, marker='v', label='Test Data');
    plt.title('ALS-WR Learning Curve, lambda = %s, K = %d'%(lambda_str, K))
    plt.xlabel('Number of Epochs');
    plt.ylabel('RMSE');
    plt.legend()
    plt.grid()
    plt.savefig("../results/test_train_rmse_"+path)
    plt.show()
    
    
def plot_cv_errors(errors, lambdas , K , path):
            
    colors = cycle(["aqua", "black", "blue", "fuchsia", "gray", "green", "lime", "maroon", "navy", "olive", "purple", "red", "silver", "teal", "yellow"])
        
    markers = cycle([ ".", ",", "o", "v" , "^" , ">", "1", "2", "3", "4", "8", "s", "p", "*", "h"])
    
    
    for i, data in enumerate(errors):
        
        lambda_str = ('%f' % lambdas[i]).rstrip('0').rstrip('.')
        plt.plot(range(len(data)), data, marker=next(markers),  label='lambda = %s'%lambda_str);
        
    plt.ylim(0.96 , 2)
    plt.xlim(0 , 50)
    plt.title('ALS-WR Learning Curve,  K = %d'% K)
    plt.xlabel('Number of Epochs');
    plt.ylabel('RMSE');
    plt.legend()
    plt.grid()
    plt.savefig("../results/"+path)
    plt.show()
    
    
def plot_general (errors, labels , x_axis_name, y_axis_name, title, path):
            
    colors = cycle(["aqua", "black", "blue", "fuchsia", "gray", "green", "lime", "maroon", "navy", "olive", "purple", "red", "silver", "teal", "yellow"])
        
    markers = cycle([ ".", ",", "o", "v" , "^" , ">", "1", "2", "3", "4", "8", "s", "p", "*", "h"])
    
    
    for i, data in enumerate(errors):
        
        plt.plot(range(len(data)), data, marker=next(markers),  label=labels[i]);
        
    plt.title(title)
    plt.xlabel(x_axis_name);
    plt.ylabel(y_axis_name);
    plt.legend()
    plt.grid()
    plt.savefig("../results/"+path)
    plt.show()
