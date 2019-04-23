# Logistic Regression with Prior.

# ============= Imports =============
from Numpy_and_mnist.data.data import *
import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from tqdm import tqdm

# ============= Data =============
N_data, train_images, train_labels, test_images, test_labels = load_mnist()
train_digits = train_images[:1000]
train_labels = train_labels[:1000]
test_digits = test_images[:100]
test_labels= test_labels[:100]

# ============= Code =============

def gradient(theta, inputs, labels):
    probabilities = np.exp(logprobabilities(theta, inputs))
    repor = labels - probabilities
    return (np.dot(repor.T, inputs))

def predicted_ratio(theta, inputs, classes):
    pred_true = np.argmax(logprobabilities(theta, inputs), axis=1)
    rate = sum(np.array(classes == pred_true, dtype=int))/pred_true.shape[0]
    return rate

def logprobabilities(theta, inputs, use_prior=False):
    sigma_squared = 1e2
    probability = np.dot(inputs, theta.T)
    prior = 0.0 if not use_prior else np.sum(np.square(theta)/(2*sigma_squared))
    return probability-logsumexp(probability, axis=1, keepdims=True)-prior

def class_labels(labels):
    return np.argmax(labels, axis=1)

def main():
    inputs, labels = np.array(train_digits), np.array(train_labels)
    theta = np.zeros((10, 784), dtype=np.float32)
    loglikelihoods, ratios = [], []
    optimal_i, optimal_thetas = None, None

    for i in tqdm(range(1000)):
        theta += 0.1 * gradient(theta, inputs, labels)
        ratios.append(predicted_ratio(theta, inputs, class_labels(train_labels)))
        lgl = -np.sum(logprobabilities(theta, inputs, False) * labels)
        loglikelihoods.append(lgl)
        if max(loglikelihoods) == lgl:
            optimal_i, optimal_thetas = i, theta

    for idx, (x, y) in enumerate([(train_digits, train_labels), (test_digits, test_labels)]):
        procedure = ['Train', 'Test'][idx]
        lgl = np.sum(logprobabilities(optimal_thetas, x) * y)
        print('{} Accuracy is at {}'.format(procedure, predicted_ratio(optimal_thetas, x, class_labels(y))))
        print('{} Average predictive log-likelihood :{}'.format(procedure, lgl/x.shape[0]))
        save_images(optimal_thetas, "weights.png")
if __name__ == '__main__':
    main()
