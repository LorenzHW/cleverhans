import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from easydict import EasyDict

from tutorials.future.tf2.vae import CVAE, compute_loss

FLAGS = flags.FLAGS


def ld_mnist():
    """Load training and test data."""

    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    dataset, info = tfds.load('mnist', data_dir='gs://tfds-data/datasets', with_info=True,
                              as_supervised=True)
    mnist_train, mnist_test = dataset['train'], dataset['test']
    mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(128).take(1)
    mnist_test = mnist_test.map(convert_types).batch(128)
    return EasyDict(train=mnist_train, test=mnist_test)


def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('./images/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


def main(_):
    latent_dim = 50
    num_examples_to_generate = 16

    data = ld_mnist()
    model = CVAE(latent_dim)
    optimizer = tf.keras.optimizers.Adam(1e-4)

    train_loss = tf.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # Random vector constant for generation (prediction) so it will be easier to see the improvement.
    random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    if FLAGS.train_new:
        for epoch in range(1, FLAGS.nb_epochs + 1):
            # keras like display of progress
            progress_bar_train = tf.keras.utils.Progbar(60000)
            for i, (x, y) in enumerate(data.train):
                train_step(x)
                progress_bar_train.add(x.shape[0], values=[('loss', train_loss.result())])

            # After every epoch generate an image
            generate_and_save_images(model, epoch, random_vector_for_generation)
        model.save_weights('./weights/vae_weights', save_format='tf')
    else:
        model.load_weights('./weights/vae_weights')

    progress_bar_test = tf.keras.utils.Progbar(10000)

    for (x, y) in data.test:
        test_loss(compute_loss(model, x))
        elbo = -test_loss.result()
        progress_bar_test.add(x.shape[0], values=[('loss', elbo)])


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 1, 'Number of epochs.')
    # flags.DEFINE_float('eps', 0.3, 'Total epsilon for FGM and PGD attacks.')
    flags.DEFINE_bool('train_new', True,
                      'If true a new model is trained and weights are saved to /weights, else weights are loaded from /weights')
    app.run(main)
