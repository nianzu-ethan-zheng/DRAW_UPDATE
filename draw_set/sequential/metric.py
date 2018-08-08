"""
Author: Nianzu Ethan Zheng
Datetime: 2018-1-31
Place: Shenyang China
Copyright
"""
import numpy as np


def parzen_estimation(mu, sigma, mode='gauss'):
    """
    Implementation of a parzen-window estimation
    
    Keyword arguments:
        x: A "nxd"-dimentional numpy array, which each sample is
                  stored in a separate row (=training example)
        mu: point x for density estimation, "dx1"-dimensional numpy array
        sigma: window width
        
    Return the density estimate p(x)
    """

    def log_mean_exp(a):
        max_ = a.max(axis=1)
        return max_ + np.log(np.exp(a - np.expand_dims(max_, axis=-1) + 1e-200).mean(1))

    def gaussian_window(x, mu, sigma):
        a = (np.expand_dims(x, axis=1) - np.expand_dims(mu, axis=0)) / sigma
        b = np.sum(- 0.5 * (a ** 2), axis=-1)
        E = log_mean_exp(b)
        Z = mu.shape[1] * np.log(sigma * np.sqrt(np.pi * 2))
        return E - Z

    return lambda x: gaussian_window(x, mu, sigma)


def get_ll(x, parzen, batch_size=100):
    """
    Get the likelihood of the input sample x,  not put all the sample into the
                parzen for the sake of computionalresource
                
    Keyword arguments:
        x : A nxp dimensional
        parzen: A window estimation function
        batch_size: A singular
        
    Return A singular, which represent the likelihood
    """
    inds = np.arange(x.shape[0])
    n_batches = int(np.ceil(len(inds) / batch_size))
    nlls = []
    for i in range(n_batches):
        nll = parzen(x[inds[i::n_batches]])
        nlls.extend(nll)
        if i % 2 == 0:
            print(i, np.mean(nlls))
    return nlls


def cross_validate_sigma(samples, data, sigmas, batch_size):
    """
    Get the best sigma,i.e., the window width, by cross validation on the
    validation set
    
    Keyword arguments:
        samples: A "nxp"- like numpy array, the training set
        data: A "mxp" -like numpy array, the validation set
        sigmas: Sigma candidates
        batch_size: Batch processing
        
    Return the best sigma, and the likelihoods corresponding to the sigmas
    """
    lls = []
    for sigma in sigmas:
        parzen = parzen_estimation(samples, sigma)
        tmp = get_ll(data, parzen, batch_size=batch_size)
        lls.append(tmp)
    ind = np.argmax(lls)
    return sigmas[ind], np.array(lls)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Parzen window, log-likelihood estimator')
    parser.add_argument('-p', '--path', help='model path')
    parser.add_argument('-s', '--sigma', default=None)
    parser.add_argument('-d', '--dataset', choices=['mnist', 'tfd'])
    parser.add_argument('-f', '--fold', default=0, type=int)
    parser.add_argument('-v', '--valid', default=False, action='store_true')
    parser.add_argument('-n', '--num_samples', default=10000, type=int)
    parser.add_argument('-l', '--limit_size', default=1000, type=int)
    parser.add_argument('-b', '--batch_size', default=100, type=int)
    parser.add_argument('-c', '--cross_val', default=10, type=int,
                        help="Number of cross valiation folds")
    parser.add_argument('--sigma_start', default=-1, type=float)
    parser.add_argument('--sigma_end', default=0., type=float)
    args = parser.parse_args()

    from tensorflow.examples.tutorials import mnist
    import os, numpy, time

    data_directory = os.path.join("../", "mnist")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # 'images', 'labels', 'next_batch', 'num_examples'
    train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train
    test_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).test

    # cross validate sigma
    samples = train_data.images[:8000]
    valid = test_data.images[:1000]
    sigma_range = numpy.logspace(args.sigma_start, args.sigma_end, num=args.cross_val)
    # sigma,_ = cross_validate_sigma(samples, valid, sigma_range, args.batch_size)
    # print(sigma_range)
    sigma = 0.16681005
    print("Using Sigma: {}".format(sigma))

    parzen = parzen_estimation(samples, sigma)
    ll = get_ll(valid, parzen, batch_size=args.batch_size)
    se = np.std(ll)/np.sqrt(test_data.num_examples)

    print("Log_likelihood of test set = {},se,{}".format(np.mean(ll), se))
    # 1k: 64
    # 2k: 101
    # 3k: 156
    # 4k: 177
    # 5k: 194
    # 6k: 204
    # 7k: 213
    # 8k: 220
    # 10k: 235







