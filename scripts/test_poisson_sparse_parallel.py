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

        # TODO: Remove Options class
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

    def test_inference(self):
        # Just a simple run to make sure that this produces nonzero results.
        md = np.array([[1, 2, 3, 4]]).T
        opts = Options(batch_size=5, num_neg_samples=3,
                       beta_mean=0, beta_scale=1,
                       gamma_mean=0, gamma_scale=1)
        N, D, p = 4, 7, 1
        with tf.Graph().as_default(), tf.Session() as sess:
            model = PoissonRegression(opts, sess)
            model.N = N
            model.D = D
            model.p = p
            obs_id = tf.constant([1, 2, 1, 6, 0], dtype=tf.int32)
            samp_id = tf.constant([1, 2, 1, 3, 0], dtype=tf.int32)
            md_tf = tf.gather(tf.constant(md, dtype=tf.float32), samp_id, axis=0)
            y_res = model.inference(md_tf, obs_id)
            tf.global_variables_initializer().run()
            y_pred, beta, gamma = sess.run(
                    [y_res, model.qbeta, model.qgamma]
            )

            self.assertIsNotNone(model.qbeta)
            self.assertIsNotNone(model.qgamma)
            self.assertIsNotNone(model.theta)
            self.assertIsNotNone(y_pred)
            self.assertIsNotNone(beta)
            self.assertIsNotNone(gamma)

    def test_loss(self):
        table = np.array([
            [1, 2, 1, 0, 0, 0],
            [0, 1, 2, 1, 0, 0],
            [0, 0, 1, 2, 1, 0],
            [0, 0, 0, 1, 2, 1]
        ])
        md = np.array([[1, 2, 3, 4]]).T
        N, D = table.shape
        p = md.shape[1]
        table = coo_matrix(table)
        opts = Options(batch_size=5, num_neg_samples=3,
                       beta_mean=0, beta_scale=1,
                       gamma_mean=0, gamma_scale=1)
        with tf.Graph().as_default(), tf.Session() as sess:
            y_data = tf.SparseTensorValue(
                indices=np.array([table.row,  table.col]).T,
                values=table.data,
                dense_shape=(N, D)
            )
            G_data = tf.constant(md, dtype=tf.float32)

            model = PoissonRegression(opts, sess)
            model.N = N
            model.D = D
            model.p = p
            model.num_nonzero = table.nnz

            batch = model.sample(y_data)
            log_loss = model.loss(G_data, y_data, batch)
            tf.global_variables_initializer().run()
            loss_, beta, gamma = sess.run(
                    [model.loss, model.qbeta, model.qgamma]
            )
            self.assertIsNotNone(loss_)
            self.assertIsNotNone(beta)
            self.assertIsNotNone(gamma)


    def test_optimize(self):
        pass

    def test_evaluate(self):
        pass

    def test_train(self):
        pass


if __name__ == "__main__":
    tf.test.main()
