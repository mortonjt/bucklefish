import numpy as np
import tensorflow as tf
from poisson_sparse_parallel import Options, PoissonRegression
from scipy.sparse import coo_matrix
from skbio.stats.composition import _gram_schmidt_basis, closure
from sklearn.utils import check_random_state
from biom import Table
import pandas as pd
from scipy.stats import pearsonr
from skbio.stats.composition import clr_inv
import matplotlib.pyplot as plt


def random_poisson_model(num_samples, num_features,
                         reps=1,
                         tree=None,
                         low=2, high=10,
                         alpha_mean=0,
                         alpha_scale=5,
                         theta_mean=0,
                         theta_scale=5,
                         gamma_mean=0,
                         gamma_scale=5,
                         kappa_mean=0,
                         kappa_scale=5,
                         beta_mean=0,
                         beta_scale=5,
                         seed=0):
    """ Generates a table using a random poisson regression model.

    Here we will be simulating microbial counts given the model, and the
    corresponding model priors.

    Parameters
    ----------
    num_samples : int
        Number of samples
    num_features : int
        Number of features
    tree : np.array
        Tree specifying orthonormal contrast matrix.
    low : float
        Smallest gradient value.
    high : float
        Largest gradient value.
    alpha_mean : float
        Mean of alpha prior  (for global bias)
    alpha_scale: float
        Scale of alpha prior  (for global bias)
    theta_mean : float
        Mean of theta prior (for sample bias)
    theta_scale : float
        Scale of theta prior (for sample bias)
    gamma_mean : float
        Mean of gamma prior (for feature bias)
    gamma_scale : float
        Scale of gamma prior (for feature bias)
    kappa_mean : float
        Mean of kappa prior (for overdispersion)
    kappa_scale : float
        Scale of kappa prior (for overdispersion)
    beta_mean : float
        Mean of beta prior (for regression coefficients)
    beta_scale : float
        Scale of beta prior (for regression coefficients)

    Returns
    -------
    table : biom.Table
        Biom representation of the count table.
    metadata : pd.DataFrame
        DataFrame containing relevant metadata.
    beta : np.array
        Regression parameter estimates.
    theta : np.array
        Bias per sample.
    gamma : np.array
        Bias per feature
    kappa : np.array
        Dispersion rates of counts per sample.
    """
    # generate all of the coefficient using the random poisson model
    state = check_random_state(seed)
    alpha = state.normal(alpha_mean, alpha_scale)
    theta = state.normal(theta_mean, theta_scale, size=(num_samples, 1)) + alpha
    beta = state.normal(beta_mean, beta_scale, size=num_features-1)
    gamma = state.normal(gamma_mean, gamma_scale, size=num_features-1)
    kappa = state.lognormal(kappa_mean, kappa_scale, size=num_features-1)

    if tree is None:
        basis = coo_matrix(_gram_schmidt_basis(num_features), dtype=np.float32)
    else:
        basis = sparse_balance_basis(tree)[0]

    G = np.hstack([np.linspace(low, high, num_samples // reps)]
                  for _ in range(reps))
    G = np.sort(G)
    N, D = num_samples, num_features
    G_data = np.vstack((np.ones(N), G)).T
    B = np.vstack((gamma, beta))

    mu = G_data @ B @ basis
    # we use kappa here to handle overdispersion.
    #eps = lambda x: state.normal([0] * len(x), x)
    eps_ = np.vstack([state.normal([0] * len(kappa), kappa)
                      for _ in range(mu.shape[0])])
    eps = eps_ @ basis
    table = np.vstack(
        state.poisson(
            np.exp(
                mu[i, :] + theta[i] + eps[i, :]
            )
        )
        for i in range(mu.shape[0])
    ).T

    samp_ids = ['S%d' % i for i in range(num_samples)]
    feat_ids = ['F%d' % i for i in range(num_features)]
    balance_ids = ['L%d' % i for i in range(num_features-1)]

    table = Table(table, feat_ids, samp_ids)
    metadata = pd.DataFrame({'G': G.ravel()}, index=samp_ids)
    beta = pd.DataFrame({'beta': beta.ravel()}, index=balance_ids)
    gamma = pd.DataFrame({'gamma': gamma.ravel()}, index=balance_ids)
    kappa = pd.DataFrame({'kappa': kappa.ravel()}, index=balance_ids)
    theta = pd.DataFrame({'theta': theta.ravel()}, index=samp_ids)
    return table, metadata, basis, alpha, beta, theta, gamma, kappa, eps_


class PoissonRegressionTest(tf.test.TestCase):

    def setUp(self):
        # specifies if plots should be displayed for diagnostics
        self.display_plots = False


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
                model.N = N
                model.D = D

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
        for _ in range(10):
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
                        [log_loss, model.qbeta, model.qgamma]
                )
                self.assertIsNotNone(loss_)
                self.assertIsNotNone(beta)
                self.assertIsNotNone(gamma)
                # Make sure that the loss is not nan
                self.assertFalse(np.isnan(loss_))


    def test_optimize(self):
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
                       learning_rate=1e-1,
                       clipping_size=10,
                       beta_mean=0, beta_scale=1,
                       gamma_mean=0, gamma_scale=1)
        for _ in range(10):
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
                train, g, v = model.optimize(log_loss)
                tf.global_variables_initializer().run()
                train_, grads, loss_1, beta, gamma = sess.run(
                        [train, g, log_loss, model.qbeta, model.qgamma]
                )
                print(beta)
                train_, loss_2, beta, gamma = sess.run(
                        [train, log_loss, model.qbeta, model.qgamma]
                )
                print(beta)
                self.assertIsNotNone(beta)
                self.assertIsNotNone(gamma)
                # make sure that there is an actual improvement wrt loss
                self.assertLess(loss_2, loss_1)

    def test_evaluate(self):
        table = np.array([
            [1, 2, 1, 0, 0, 0],
            [0, 1, 2, 1, 0, 0],
            [0, 0, 1, 2, 1, 0],
            [0, 0, 0, 1, 2, 1]
        ])
        md = np.array([[1, 2, 3, 4]]).T
        md_holdout = np.array([[1.5, 2.5]]).T
        table_holdout = np.array([
            [1, 2, 1, 0, 0, 0],
            [0, 1, 2, 1, 0, 0]
        ], dtype=np.float32)
        N, D = table.shape
        M, D = table_holdout.shape
        p = md.shape[1]
        table = coo_matrix(table)
        table_holdout = coo_matrix(table_holdout)

        opts = Options(batch_size=5, num_neg_samples=3,
                       learning_rate=1e-1,
                       clipping_size=10,
                       beta_mean=0, beta_scale=1,
                       gamma_mean=0, gamma_scale=1)
        for _ in range(10):
            with tf.Graph().as_default(), tf.Session() as sess:
                y_data = tf.SparseTensorValue(
                    indices=np.array([table.row,  table.col]).T,
                    values=table.data,
                    dense_shape=(N, D)
                )
                y_holdout = tf.SparseTensorValue(
                    indices=np.array([table_holdout.row, table_holdout.col]).T,
                    values=table_holdout.data,
                    dense_shape=table_holdout.shape
                )

                G_data = tf.constant(md, dtype=tf.float32)
                G_holdout = tf.constant(md_holdout, dtype=tf.float32)

                model = PoissonRegression(opts, sess)
                model.N = N
                model.D = D
                model.p = p
                model.num_nonzero = table.nnz

                batch = model.sample(y_data)
                log_loss = model.loss(G_data, y_data, batch)
                train = model.optimize(log_loss)
                mad = model.evaluate(G_holdout, y_holdout)
                tf.global_variables_initializer().run()
                train_, mad_, loss_, beta, gamma = sess.run(
                    [train, mad, log_loss, model.qbeta, model.qgamma]
                )

                self.assertIsNotNone(beta)
                self.assertIsNotNone(gamma)
                # Look at mean absolute error
                self.assertFalse(np.isnan(mad_))


    def test_with_simulation(self):

        num_samples = 100
        num_features = 1000
        ex = random_poisson_model(num_samples, num_features,
                                  reps=1,
                                  low=-1, high=1,
                                  alpha_mean=-4,
                                  alpha_scale=1,
                                  theta_mean=0,
                                  theta_scale=1,
                                  gamma_mean=0,
                                  gamma_scale=1,
                                  kappa_mean=0,
                                  kappa_scale=0,
                                  beta_mean=0,
                                  beta_scale=2
        )

        (table, md, basis, sim_alpha, sim_beta, sim_theta,
         sim_gamma, sim_kappa, sim_eps) = ex

        N, D = num_samples, num_features
        p = md.shape[1]   # number of covariates
        table = table.matrix_data.tocoo().T

        # Building the model
        opts = Options(batch_size=500, num_neg_samples=500,
                       learning_rate=1e-1,
                       clipping_size=10,
                       beta_mean=0, beta_scale=2,
                       save_path='tf_debug',
                       gamma_mean=0, gamma_scale=1)
        with tf.Graph().as_default(), tf.Session() as sess:
            y_data = tf.SparseTensorValue(
                indices=np.array([table.row,  table.col]).T,
                values=table.data,
                dense_shape=(N, D)
            )
            G_data = tf.constant(md.values, dtype=tf.float32)

            model = PoissonRegression(opts, sess)
            model.N = N
            model.D = D
            model.p = p
            model.num_nonzero = table.nnz

            batch = model.sample(y_data)
            log_loss = model.loss(G_data, y_data, batch)
            train, g, v = model.optimize(log_loss)
            tf.global_variables_initializer().run()
            train_, grads, loss_1, beta, gamma, theta = sess.run(
                    [train, g, log_loss,
                     model.qbeta, model.qgamma, model.theta]
            )
            for _ in range(100):
                train_, loss_2, beta, gamma, theta = sess.run(
                        [train, log_loss,
                         model.qbeta, model.qgamma, model.theta]
                )

            beta_corr, bval = pearsonr(beta.ravel(),
                                       sim_beta.values.ravel() @ basis)
            gamma_corr, gval = pearsonr(gamma.ravel(),
                                        sim_gamma.values.ravel() @ basis)
            theta_corr, tval = pearsonr(theta.ravel(),
                                        sim_theta.values.ravel())

            if self.display_plots:
                x, y = beta.ravel(), sim_beta.values.ravel() @ basis
                fig, ax = plt.subplots()
                mx = np.linspace(min([x.min(), y.min()]),
                                 max([x.max(), y.max()]))
                ax.plot(mx, mx, '-k')
                ax.scatter(x, y)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                fig.savefig('beta.pdf')

                x, y = gamma.ravel(), sim_gamma.values.ravel() @ basis
                fig, ax = plt.subplots()
                mx = np.linspace(min([x.min(), y.min()]),
                                 max([x.max(), y.max()]))
                ax.plot(mx, mx, '-k')
                ax.scatter(x, y)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                fig.savefig('gamma.pdf')

                x, y = theta.ravel(), sim_theta.values.ravel()
                fig, ax = plt.subplots()
                mx = np.linspace(min([x.min(), y.min()]),
                                 max([x.max(), y.max()]))
                ax.plot(mx, mx, '-k')
                ax.scatter(x, y)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                fig.savefig('theta.pdf')

            self.assertGreater(beta_corr, 0.7)
            self.assertGreater(gamma_corr, 0.7)
            self.assertGreater(theta_corr, 0.7)


if __name__ == "__main__":
    tf.test.main()
