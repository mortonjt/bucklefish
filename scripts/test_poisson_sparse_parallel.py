import numpy as np
import tensorflow as tf
from poisson_sparse_parallel import Options, PoissonRegression
from scipy.sparse import coo_matrix


class PoissonRegressionTest(tf.test.TestCase):

    def setUp(self):
        pass

    def test_sample(self):
        # Need to define the metadata and bioms
        row = np.array([1, 2, 1, 4, 1, 0, 3], dtype=np.int32)
        col = np.array([2, 1, 4, 2, 0, 3, 5], dtype=np.int32)
        data = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
        exp = np.vstack((row, col, data)).T

        opts = Options(batch_size=4, num_neg_samples=3)
        N, D = 5, 6
        for _ in range(10):
            with tf.Graph().as_default(), tf.Session() as sess:
                y_data = tf.SparseTensorValue(
                    indices=np.array([row,  col]).T,
                    values=data,
                    dense_shape=(N, D)
                )

                model = PoissonRegression(opts, sess)
                pos, neg, acc, p, n = model.sample(y_data)
                pos_res, neg_res, acc_res = sess.run(
                    [pos, neg, acc]
                )

                # convert to tables with 3 columns (row, col, data)
                res_pos = np.hstack((pos_res.indices, pos_res.values.reshape(-1, 1)))
                res_neg = np.hstack((neg_res.indices, neg_res.values.reshape(-1, 1)))

                intersect = set(tuple(x) for x in exp) & \
                            set(tuple(x) for x in res_pos)
                # There may be duplicates
                self.assertLessEqual(len(intersect), len(res_pos))
                self.assertGreater(len(intersect), 1)

                # there may be positive entries, so
                # only test if there is a zero
                self.assertIn(0, list(neg_res.values))

                # Make sure that there aren't any zeros in the accidient.
                if len(acc_res) > 0:
                    acc_row, acc_col, acc_data = acc_res
                    self.assertNotIn(0, list(acc_data))


if __name__ == "__main__":
    tf.test.main()
