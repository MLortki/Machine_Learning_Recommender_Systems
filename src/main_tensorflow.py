# Useful starting lines

import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
import sklearn.model_selection as skm

import tensorflow as tf

from plots import plot_raw_data
from helpers import load_data, preprocess_data



#define parameters
tf.app.flags.DEFINE_string("ps_hosts", "",
                               "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                               "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("log_dir", "", "Logging directory")

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS


# parsing command line arguments
parameter_servers = FLAGS.ps_hosts.split(",")
workers = FLAGS.worker_hosts.split(",")
# Specify the parameter servers and worker hosts.
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# start a server for a specific task
server = tf.train.Server(cluster, 
                          job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)



#import dataset
path_dataset = "../data/data_train.csv"
ratings = load_data(path_dataset)
num_items_per_user, num_users_per_item = plot_raw_data(ratings, False)

#initialize parameters
D = 10000
N = 1000
K = 20

MAX_STEP = 50



if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    
    
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):

        # count the number of updates
        global_step = tf.get_variable('global_step', [], 
                                initializer = tf.constant_initializer(0), 
                                trainable = False)
        
        
        item_features = tf.Variable(np.random.randn(K,D).astype(np.float32), name='Items')
        user_features = tf.Variable(np.random.randn(K,N).astype(np.float32), name='Users')
        
        r = tf.Placeholder(np.zeros((K)))
       
        
        #define loss function
        loss = tf.reduce_sum((Y_tf-Y_est)**2)
              
    
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                              global_step=global_step,
                              #logdir = FLAGS.log_dir,
                              #summary_op = summary_op,
                              summary_op = None,
                              recovery_wait_secs=1,
                              init_op=init_op)
        

    
    
    