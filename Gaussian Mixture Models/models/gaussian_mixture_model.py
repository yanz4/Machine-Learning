"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal
import random
import math

np.set_printoptions(threshold=np.nan)


class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""

    def __init__(self, n_dims, n_components=1,
                 max_iter=10,
                 reg_covar=0.1):
        """
        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar

        # Randomly Initialize model parameters
        self._mu = None  # np.array of size (n_components, n_dims)

        # Initialized with uniform distribution.
        self._pi = np.random.rand(self._n_components,1)  # np.array of size (n_components, 1)
        # Initialized with identity.
        # np.array of size (n_components, n_dims, n_dims)
        #self._sigma = None
        self._sigma = np.zeros((self._n_components, self._n_dims, self._n_dims))
        for i in range(self._n_components):
            self._sigma[i, :] = np.eye(self._n_dims)

    def fit(self, x):
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        self._mu = np.zeros((self._n_components, self._n_dims))
        for i in range(self._n_dims):
            self._mu[:, i] = random.sample(list(x[:,i]), self._n_components)
        counter = 0
        while counter < self._max_iter:
            counter += 1
            print(counter)
            z_ik = self._e_step(x)
            z_ik = (z_ik.T / (np.sum(z_ik, axis=1))).T
            self._m_step(x, z_ik)
            print('z_ik', z_ik)
        pass



    def _e_step(self, x):
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        return self.get_posterior(x)


    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)
        """
        # Update the parameters.
        N = x.shape[0]
        self._pi = np.expand_dims(np.sum(z_ik, axis=0)/N, axis=1)

        for k in range(self._n_components):
            sum = 0
            for i in range(N):
                sum += z_ik[i,k]*x[i, :]
            self._mu[k, :] = sum/(N*self._pi[k])

        for k in range(self._n_components):
            sum = 0
            for i in range(N):
                sum += z_ik[i,k]*np.outer(x[i,:]-self._mu[k,:], x[i,:]-self._mu[k,:])
            self._sigma[k,:] = sum/(N*self._pi[k])
        pass

    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N, n_components).
        """
        ret = []
        for k in range(self._n_components):
            for m in range(len(self._sigma[k])):
                self._sigma[k,m,m] += self._reg_covar
            ret.append(self._pi[k]*self._multivariate_gaussian(x, self._mu[k,:], self._sigma[k,:]))
        ret = np.array(ret).T
        #print('ret shape', ret.shape)
        return ret

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        ret = self.get_conditional(x)
        marginals = np.sum(ret, axis=1)
        return marginals


    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """

        z_ik = []
        for i in range(len(x)):
            z_i = self.get_conditional(x)[i]/self.get_marginals(x)[i]
            z_ik.append(z_i)
        z_ik=np.array(z_ik)
        #print("z_ik.shape", z_ik.shape)
        return z_ik


    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.
        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,1)
            sigma_k(numpy.ndarray): Array containing one signle covariance matrix
                (ndims, ndims)
        """
        return multivariate_normal.pdf(x, mu_k, sigma_k)


    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.
        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """
        self.fit(x)
        z_ik = self.get_posterior(x)
        assignment = np.argmax(z_ik, axis=1)
        print(assignment)
        cluster_label_map = -1*np.ones((self._n_components, 1))
        for i in range(self._n_components):
            if len(scipy.stats.mode(y[assignment == i])[0]) == 0:
                cluster_label_map[i] = (np.random.choice(y))
            else:
                cluster_label_map[i] = (scipy.stats.mode(y[assignment == i])[0])
        self.cluster_label_map = np.array(cluster_label_map)
        print(cluster_label_map)
        pass


    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """
        N = len(x)
        z_ik = self.get_posterior(x)
        y_hat = []
        assignment = np.argmax(z_ik, axis=1)
        for i in range(N):
            y_hat.append(self.cluster_label_map[assignment[i]])
        print('yhat', y_hat)
        return np.array(y_hat)
