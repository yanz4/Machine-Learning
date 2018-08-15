"""Generative adversarial network."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers


class Gan(object):
    """Adversary based generator network.
    """

    def __init__(self, ndims=784, nlatent=2):
        """Initializes a GAN

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent

        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])
        print(1)
        # Input noise
        self.z_placeholder = tf.placeholder(tf.float32, [None, nlatent])

        # Build graph.
        self.x_hat = self._generator(self.z_placeholder)
        y_hat = self._discriminator(self.x_hat)
        y = self._discriminator(self.x_placeholder, reuse=True)

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)

        # Generator loss
        self.g_loss = self._generator_loss(y_hat)

        # Add optimizers for appropriate variables
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])
        self.update_op_tensor_d, self.update_op_tensor_g = self.update_op(self.d_loss, self.g_loss,
                                                                          self.learning_rate_placeholder)
        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake image(None, 1). 
              DO NOT USE AN ACTIVATION FUNCTION AT THE OUTPUT LAYER HERE.

        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            self.D_W1 = tf.get_variable("D_W1", [784, 128])
            self.D_b1 = tf.get_variable("D_b1", [128])
            self.D_W2 = tf.get_variable("D_W2", [128, 1])
            self.D_b2 = tf.get_variable("D_b2", [1])
            self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]
            D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
            D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
            y = tf.nn.sigmoid(D_logit)
            return y

    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator.

        Args:
            y (tf.Tensor): The output tensor of the discriminator for true images of dimension (None, 1).
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        l = -tf.reduce_mean(tf.log(y) + tf.log(1. - y_hat))
        return l

    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation 
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, 784).
        """
        with tf.variable_scope("generator", reuse=reuse) as scope:
            self.G_W1 = tf.get_variable("G_W1", [2, 128])
            self.G_b1 = tf.get_variable("G_b1", [128])
            self.G_W2 = tf.get_variable("G_W2", [128, 784])
            self.G_b2 = tf.get_variable("G_b2", [784])
            self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]
            G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
            G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
            x_hat = tf.nn.sigmoid(G_log_prob)
            return x_hat

    def _generator_loss(self, y_hat):
        """Loss for the discriminator.

        Args:
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        l = -tf.reduce_mean(tf.log(y_hat))
        return l

    def update_op(self, d_loss, g_loss, learning_rate):
        """Creates the update optimizer.

        Use tf.train.AdamOptimizer to obtain the update op.

        Args:
            loss(tf.Tensor): Tensor of shape () containing the loss function.
            learning_rate(tf.Tensor): Tensor of shape (). Learning rate for
                gradient descent.
        Returns:
            train_op(tf.Operation): Update opt tensorflow operation.
        """
        ####### Implementation Here ######
        train_op_d = tf.train.AdamOptimizer(learning_rate).minimize(d_loss,var_list=self.theta_D)
        train_op_g = tf.train.AdamOptimizer(learning_rate).minimize(g_loss,var_list=self.theta_G)
        return train_op_d, train_op_g

    def generate_samples(self, z_np):
        """Generates random samples from the provided z_np.

        Args:
            z_np(numpy.ndarray): Numpy array of dimension
                (batch_size, _nlatent).

        Returns:
            out(numpy.ndarray): The sampled images (numpy.ndarray) of
                dimension (batch_size, _ndims).
        """
        ####### Implementation Here ######
        '''d_out = self.session.run(self.y_hat,feed_dict={self.z_placeholder:z_np,self.x_placeholder:x_np})'''
        g_out = self.session.run(self.x_hat, feed_dict={self.z_placeholder: z_np})
        return g_out
