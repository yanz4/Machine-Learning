"""Generative Adversarial Networks
"""

import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.gan import Gan


def train(model, mnist_dataset, learning_rate=0.0005, batch_size=16,
          num_steps=5000):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size and
    learning_rate.

    Args:
        model(GAN): Initialized generative network.
        mnist_dataset: input_data.
        learning_rate(float): Learning rate.
        batch_size(int): batch size used for training.
        num_steps(int): Number of steps to run the update ops.
    """

    for step in range(0, num_steps):
        batch_x, _ = mnist_dataset.train.next_batch(batch_size)
        batch_z = np.random.uniform(-1, 1, [batch_size, 2])
        # Train generator and discriminator
        model.session.run(
            [model.update_op_tensor_d, model.update_op_tensor_g],
            feed_dict={model.x_placeholder: batch_x, model.z_placeholder: batch_z,
                       model.learning_rate_placeholder: learning_rate}
        )


def main(_):
    """High level pipeline.

    This scripts performs the training for GANs.
    """
    # Get dataset.
    mnist_dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Build model.
    model = Gan()

    # Start training
    train(model, mnist_dataset)

    # Plot out latent space, for +/- 3 std.
    std = 1
    x_z = np.linspace(-3 * std, 3 * std, 20)
    y_z = np.linspace(-3 * std, 3 * std, 20)

    out = np.empty((28 * 20, 28 * 20))
    for x_idx, x in enumerate(x_z):
        for y_idx, y in enumerate(y_z):
            z_mu = np.array([[y, x]])
            img = model.generate_samples(z_mu)
            out[x_idx * 28:(x_idx + 1) * 28,
            y_idx * 28:(y_idx + 1) * 28] = img[0].reshape(28, 28)
    plt.imsave('latent_space_vae.png', out, cmap="gray")


if __name__ == "__main__":
    tf.app.run()
