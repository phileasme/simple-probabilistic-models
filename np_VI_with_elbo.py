# Variational  Inference on MNIST

# ============= Imports =============
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam
import autograd.scipy.stats.norm as norm
from autograd.scipy.misc import logsumexp
import autograd.scipy.stats.multivariate_normal as mvn
from Numpy_and_mnist.data.data import load_mnist, save_images
import matplotlib.pyplot as plt

# ============= Data =============
# Load MNIST and Set Up Data
N, D = (300, 784)
N_data, train_images, train_labels, test_images, test_labels = load_mnist()
train_images = np.round(train_images[0:N])
train_labels = train_labels[0:N]
test_images = np.round(test_images[0:10000])
test_labels = np.round(test_labels[0:10000])

# ============= Parameters =============
K = 10
prior_std = .1
# Choose two pixels and plot the K specific weights against eachother
contourK = 2
px1 = 392 # Middle Pixel
px2 = px1 + 28*5 +1 # Middle Pixel + 5 rows down
# px2 = px1+14 # Middle left-most edge
init_params = (npr.RandomState(0).randn(K, D) * 0.01, npr.RandomState(1).randn(K, D) * 0.01)

# ============= Code =============

def logprobabilities(theta, inputs):
	p = np.dot(theta, inputs)
	return p-logsumexp(p, axis=1, keepdims=True)

def logistic_logprob(params, images, labels):
	# params is a block of S x K x D params
	# images is N x D
	# labels is N x K one-hot
	# return S logprobs, summing over N
	mul = np.einsum('skd,nd->snk', params, images)
	normalized = mul - logsumexp(mul, axis=-1, keepdims=True)
	return np.einsum('snk,nk->s', normalized, labels)

def diag_gaussian_log_density(x, mu, log_std):
	# assumes that mu and log_std are (S x K X D),
	# so we sum out the last two dimensions.
	return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=(-1, -2))

def sample_diag_gaussian(mean, log_std, num_samples, rs):
	return rs.randn(num_samples, *np.shape(mean)) * np.exp(log_std) + mean


def elbo_estimate(var_params, logprob, num_samples, rs):
	"""Provides a stochastic estimate of the variational lower bound.
	var_params is (mean, log_std) of a Gaussian."""
	mean, log_std = var_params
	samples = sample_diag_gaussian(mean, log_std, num_samples, rs)
	log_p = logprob_given_data(samples)
	log_q = diag_gaussian_log_density(samples, mean, log_std)
	return np.mean(log_p - log_q)

def logprob_given_data(params):
	data_logprob = logistic_logprob(params, train_images, train_labels)
	prior_logprob = diag_gaussian_log_density(params, np.zeros(params.shape), prior_std)
	return data_logprob + prior_logprob

def objective(var_params, iter):
	return -elbo_estimate(var_params, logprob_given_data, num_samples=100, rs=npr.RandomState(iter))


# Code for plotting the isocontours below
def logprob_given_two(params, two_params):
	N = two_params.shape[0]

	params_adjust = np.zeros((N, K, D))
	params_adjust[:, contourK, px1] = two_params[:, 0]
	params_adjust[:, contourK, px2] = two_params[:, 1]

	adjusted_params = params_adjust

	return logprob_given_data(adjusted_params)

# Set up plotting code
def plot_isocontours(ax, func, xlimits=[-10, 10], ylimits=[-10, 10], numticks=21, **kwargs):
	x = np.linspace(*xlimits, num=numticks)
	y = np.linspace(*ylimits, num=numticks)
	X, Y = np.meshgrid(x, y)
	zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
	Z = zs.reshape(X.T.shape)
	plt.contour(X, Y, Z, **kwargs)
	ax.set_yticks([])
	ax.set_xticks([])

def plot_posterior_contours(mean_params,logstd_params):
	plt.clf()
	logprob_adj = lambda two_params: logprob_given_two(mean_params, two_params)
	plot_isocontours(ax, logprob_adj, cmap='Blues')
	mean_2d = mean_params[contourK, [px1,px2]]
	logstd_2s = logstd_params[contourK, [px1,px2]]
	variational_contour = lambda x: mvn.logpdf(x, mean_2d, np.diag(np.exp(2 * logstd_2s)))
	plot_isocontours(ax, variational_contour, cmap='Reds')
	plt.draw()
	plt.pause(1)


fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

# Gradient of objective using autograd.
objective_grad = grad(objective)

def print_perf(var_params, iter, gradient):
	mean_params, logstd_params = var_params
	print(".", end='')
	if iter % 30 == 0:
		save_images(mean_params, 'plotmean_vi_{}.png'.format(iter % 30))
		save_images(logstd_params, 'plotsgd_vi_{}.png'.format(iter % 30))
		plot_posterior_contours(mean_params, logstd_params)
		print('iteration {}'.format(iter))
		print('loss {}'.format(objective(var_params, iter)))
		sample = sample_diag_gaussian(mean_params, logstd_params, num_samples=1, rs=npr.RandomState(0))
		save_images(sample[0, ...], 'plotsample_vi.png')

def predicted_ratio(images, labels, sample_weights):

	exact_classes = np.argmax(labels, axis=1)
	pred_classes_per_examples = np.zeros((images.shape[0],10))

	for weights in sample_weights:
		pred = logprobabilities(images, weights.T)
		pred_classes_per_examples += np.exp(pred)/100.0
	return np.mean(np.array(exact_classes == np.argmax(pred_classes_per_examples, axis=1), dtype=int))


def get_score(samples, testing=False):
	if testing:
		ti = test_images
		tl = test_labels

	else:
		ti = train_images
		tl = train_labels
	accuracy = predicted_ratio(ti, tl, samples)
	print("{} set accuracy: {}%".format("Testing" if testing else "Training", accuracy*100))


optimized_params = adam(objective_grad, init_params, step_size=0.05, num_iters=100, callback=print_perf)
opt_mean, opt_std = optimized_params
save_images(opt_mean, 'plotmean_vi.png')
save_images(opt_std, 'plotsgd_vi.png')


samples = sample_diag_gaussian(opt_mean, opt_std, 100, npr.RandomState(0))

get_score(samples)
get_score(samples, True)
