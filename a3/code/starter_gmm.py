import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from helper import *

# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# Constants
pi = 3.141592654

# For Validation set
is_valid = False
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
# Inputs
#   X: is an NxD matrix (N observations and D dimensions)
#   MU: is an KxD matrix (K means and D dimensions)
# Outputs
#   pair_dist: is the pairwise distance matrix (NxK)
def distanceFunc(X, MU):
    #pair_dist = np.sum(X**2, axis=1) - 2*(X @ MU.T) + np.sum((MU.T)**2, axis=0)
    pair_dist = tf.reduce_sum(tf.square(X), axis=1, keepdims=True) \
                - 2 * tf.matmul(X, tf.transpose(MU)) \
                + tf.reduce_sum(tf.square(tf.transpose(MU)), axis=0, keepdims=True)

    return pair_dist

# Inputs
#   X: N X D
#   MU: K X D
#   sigma: 1 X K
# Outputs:
#   log Gaussian PDF N X K
def log_GaussPDF(X, MU, sigma):
    # HEADS UP:  I define sigma to be 1xK NOT Kx1
    pair_dist = distanceFunc(X, MU)
    log_PDF = - dim * tf.log(sigma * np.sqrt(2*pi)) - pair_dist / (2 * tf.square(sigma))
    return log_PDF


# Input
#   log_PDF: log Gaussian PDF N X K
#   log_pi: 1 X K
# Outputs
#   log_post: N X K
def log_posterior(log_PDF, log_pi):
    # HEADS UP:  I define log_pi to be 1xK NOT Kx1
    Z = log_PDF + log_pi
    log_post = Z - reduce_logsumexp(Z, reduction_indices=1, keep_dims=True)
    return log_post

# Input
#   X: N X D
#   MU: K X D
#   sigma: 1 X K
#   w: 1 X K (weights aka. P(k))
# Outputs
#   loss: constant
def calculate_loss(X, MU, sigma, w):
    P = log_GaussPDF(X, MU, sigma)
    Q = tf.reduce_logsumexp(P + tf.log(w), reduction_indices=1, keep_dims=True)
    loss = - tf.reduce_sum(Q, reduction_indices=0, keep_dims=False)
    return loss

# Returns a Nx1 vector of cluster assignments
def cluster_assignments(X, MU, sigma, w):
    log_PDF = log_GaussPDF(X, MU, sigma)
    P_j_x = log_posterior(log_PDF, tf.log(w))
    s = tf.argmax(P_j_x, axis=1)
    return s
