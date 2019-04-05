import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
is_valid = False
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for K-means
# Inputs
#    X: is an NxD matrix (N observations and D dimensions)
#    MU: is an KxD matrix (K means and D dimensions)
# Outputs
#   pair_dist: is the pairwise distance matrix (NxK)
#   pair_dist = np.sum(X**2, axis=1) - 2*(X @ MU.T) + np.sum((MU.T)**2, axis=0)
def distanceFunc(X, MU):
    pair_dist = tf.reduce_sum(tf.square(X), axis=1, keepdims=True) \
                - 2 * tf.matmul(X, tf.transpose(MU)) \
                + tf.reduce_sum(tf.square(tf.transpose(MU)), axis=0, keepdims=True)

    return pair_dist

# Squared Distance Loss for K-Means
def calculate_loss(X, MU):
    D = distanceFunc(X, MU)
    e = tf.reduce_min(D, axis=1)
    L = tf.reduce_sum(e)
    return L

# Partitions the data into K clusters based on MU
# returns a Nx1 vector of cluster assignments for x1 - xN
# the clusters are numbered from 0 to K-1
def cluster_assignments(X, MU):
    D = distanceFunc(X, MU)
    s = tf.argmin(D, axis=1)
    return s
