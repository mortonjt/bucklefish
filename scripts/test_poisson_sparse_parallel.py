import numpy as np
import tensorflow as tf
from poisson_sparse_parallel import Options, PoissonRegression
from scipy.sparse import coo_matrix


class PoissonRegressionTest(tf.test.TestCase):

    def setUp(self):
        pass

    def test_batch(self):
        # Need to define the metadata and bioms
        row = np.array([1, 2, 1, 5, 1, 0, 3], dtype=np.int32)
        col = np.array([2, 1, 4, 2, 0, 3, 5], dtype=np.int32)
        data = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)

        opts = Options(batch_size=4, num_neg_samples=3)
        with tf.Graph().as_default(), tf.Session() as sess:
            tf.set_random_seed(1)

            y_data = tf.SparseTensorValue(
                indices=np.array([row,  col]).T,
                values=data,
                dense_shape=(5, 5)
            )

            model = PoissonRegression(opts, sess)
            pos, neg, acc = model.batch(y_data)
            pos_res, neg_res = sess.run(
                [pos, neg]
            )

            print(pos_res.indices)
            print(pos_res.values)
            print(neg_res.indices)
            print(neg_res.values)


if __name__ == "__main__":
    tf.test.main()
