
# ============= Imports =============
from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd.scipy.special import expit as sigmoid

from data.data import load_mnist, save_images

# ============= Data =============
# Load MNIST and Set Up Data
N_data, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = np.round(train_images[0:10000])
train_labels = train_labels[0:10000]
test_images = np.round(test_images[0:10000])
np.random.seed(40)

# ============= Parameters =============
K, D = (30, 784)
# Random initialization, with set seed for easier debugging
init_params = 0.45 + 0.01 * np.random.rand(30, 784)

# ============= Code =============

# Easily learning params using EM (conditioning over n on P(c|..)*x / Cond_n P(C|..) )
def Expectation_Maximization_p_x():

    np.random.seed(40)
    params = init_params
    params_prev =10000 #Number of examples

    while 10 ** -10 < np.max(abs(params-params_prev)):

        params_prev = params

        logproba = np.dot(train_images, np.log(params.T)) + np.dot((1-train_images), np.log(1-params.T))

        div = logproba - logsumexp(logproba, axis=1, keepdims=True)

        numerator = np.dot(np.exp(div.T), train_images)

        params_update = numerator / np.sum(np.exp(div.T), axis=1, keepdims=True)

        # Common technique to deal with underflow
        params = 10 ** -2 + (1 - 10 ** -2) * params_update

    save_images(-params, 'params_plot.png')
    print("Done")

Expectation_Maximization_p_x()

# VS

batch_size = 10
num_batches = int(np.ceil(len(train_images) / batch_size))

def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)

# This is numerically stable code to for the log of a bernoulli density
def bernoulli_log_density(targets, unnormalized_logprobs):
    # Unnormalized_logprobs are in R
    # Targets must be 0 or 1
    t2 = targets * 2 - 1
    # Now t2 is -1 or 1, which makes the following form nice
    label_probabilities = -np.logaddexp(0, -unnormalized_logprobs*t2)
    return np.sum(label_probabilities, axis=-1)   # Sum across pixels.

def batched_loss(params, iter):
    data_idx = batch_indices(iter)
    return neglogprob(params, train_images[data_idx, :])

def neglogprob(params, data):
    joint_log_prob = 0
    # Batch elements
    # Train 1 param at a time.
    for x in range(10):
        # Per class
        lpb = -np.pi + logsumexp(np.dot(data[x], params.T)
         + np.dot((1-data[x]), -params.T))
        neg_norm_bernoulli_log_density = bernoulli_log_density(data[x], lpb)
        sumind = neg_norm_bernoulli_log_density
        joint_log_prob += sumind
    return -joint_log_prob

def print_perf(params, iter, gradient):
    if iter % 30 == 0:
        save_images(sigmoid(params), 'plot.png')
        print(batched_loss(params, iter))

# Get gradient of objective using autograd.
# objective_grad = grad(batched_loss)
# optimized_params = adam(objective_grad, init_params, step_size=0.2, num_iters=10000, callback=print_perf)
