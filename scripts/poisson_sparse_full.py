"""
Here, there is no negative sampling performed.
Instead, the full likelihood is evaluated.

1. First sample positive entries (i, j) for sample i and feature j
   (make sure to int64 uniform sampling)
2. Then compute gradients for all coefficients in sample i
3. Perform SGD update.
"""
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from biom import load_table, Table

from gneiss.balances import _balance_basis
from gneiss.composition import ilr_transform
from gneiss.util import match, match_tips, rename_internal_nodes

from tensorflow.contrib.distributions import Poisson, Normal
from patsy import dmatrix
from skbio import TreeNode
from skbio.stats.composition import closure, clr_inv
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix
from util import cross_validation
from tqdm import tqdm
import time


flags = tf.app.flags
flags.DEFINE_string("save_path", None, "Directory to write the model and "
                    "training summaries.")
flags.DEFINE_string("train_biom", None, "Input biom table for training. "
                    "i.e. input.biom")
flags.DEFINE_string("test_biom", None, "Input biom table for testing. "
                    "i.e. input.biom")
flags.DEFINE_string("train_metadata", None, "Input sample metadata for training. "
                    "i.e. metadata.txt")
flags.DEFINE_string("test_metadata", None, "Input sample metadata for testing. "
                    "i.e. metadata.txt")
flags.DEFINE_string("tree", None, "Input tree. "
                    "i.e. tree.nwk")
flags.DEFINE_string("formula", None, "Statistical formula for "
                    "specifying covariates.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_float("clipping_size", 10, "Gradient clipping size.")
flags.DEFINE_float("beta_mean", 0,
                   'Mean of prior distribution for covariates')
flags.DEFINE_float("beta_scale", 1.0,
                   'Scale of prior distribution for covariates')
flags.DEFINE_float("gamma_mean", 0,
                   'Mean of prior distribution for feature bias')
flags.DEFINE_float("gamma_scale", 1.0,
                   'Scale of prior distribution for feature bias')
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training sample.")
flags.DEFINE_integer("batch_size", 512,
                     "Number of nonzero entries processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("block_size", 10,
                     "Number of training samples processed per step")
flags.DEFINE_integer("min_sample_count", 1000,
                     "The minimum number of counts a feature needs for it to be "
                     "included in the analysis")
flags.DEFINE_integer("min_feature_count", 5,
                     "The minimum number of counts a sample needs to be  "
                     "included in the analysis")
flags.DEFINE_integer("statistics_interval", 10,
                     "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 10,
                     "Save training summary to file every n seconds (rounded "
                     "up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 600,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "seconds (rounded up to statistics interval).")
flags.DEFINE_boolean("verbose", False,
                     "Specifies if cross validation and summaries "
                     "are saved during training. ")
FLAGS = flags.FLAGS


def preprocess(formula,
               train_table, train_metadata,
               test_table, test_metadata,
               min_sample_count=10, min_feature_count=10):
  """ Performs data preprocessing.

  Parameters
  ----------
  formula : str
     Statistical formula specifying the design matrix of covariates
     in the study design.
  train_table : biom.Table
     Biom table containing the feature counts within the training dataset.
  train_metadata : pd.DataFrame
     Sample metadata table containing all of the measured covariates in
     the training dataset.
  test_table : biom.Table
     Biom table containing the feature counts within the holdout dataset.
  test_metadata : pd.DataFrame
     Sample metadata table containing all of the measured covariates in
     the holdout test dataset.
  min_sample_counts : int
     Minimum number of total counts within a sample to be kept.
  min_feature_counts : int
     Minimum number of total counts within a feature to be kept.

  Returns
  -------
  train_table : biom.Table
     Biom table containing the feature counts within the training dataset.
  train_metadata : pd.DataFrame
     Sample metadata table containing all of the measured covariates in
     the training dataset.
  test_table : biom.Table
     Biom table containing the feature counts within the holdout dataset.
  test_metadata : pd.DataFrame
     Sample metadata table containing all of the measured covariates in
     the holdout test dataset.

  Notes
  -----
  This assumes that the biom tables can fit into memory - will
  require some extra consideration when this is no longer the case.
  """
  # preprocessing
  train_table, train_metadata = train_table, train_metadata
  sample_filter = lambda val, id_, md: (
    (id_ in train_metadata.index) and np.sum(val) > min_sample_count)
  read_filter = lambda val, id_, md: np.sum(val) > min_feature_count
  train_table = train_table.filter(sample_filter, axis='sample')
  train_table = train_table.filter(read_filter, axis='observation')
  train_metadata = dmatrix(formula, train_metadata, return_type='dataframe')
  train_table, train_metadata = match(train_table, train_metadata)

  # hold out data preprocessing
  test_table, test_metadata = test_table, test_metadata
  metadata_filter = lambda val, id_, md: id_ in test_metadata.index
  obs_lookup = set(train_table.ids(axis='observation'))
  feat_filter = lambda val, id_, md: id_ in obs_lookup
  test_table = test_table.filter(metadata_filter, axis='sample')
  test_table = test_table.filter(feat_filter, axis='observation')
  test_metadata = dmatrix(formula, test_metadata, return_type='dataframe')
  test_table, test_metadata = match(test_table, test_metadata)

  # pad extra columns with zeros, so that we can still make predictions
  extra_columns = list(set(train_metadata.columns) - set(test_metadata.columns))
  df = pd.DataFrame({C: np.zeros(test_metadata.shape[0])
                     for C in extra_columns}, index=test_metadata.index)
  test_metadata = pd.concat((test_metadata, df), axis=1)

  return train_table, test_table, train_metadata, test_metadata


class PoissonRegression(object):

  def __init__(self, session, **kwargs):
    """ Constructor for Poisson regression.

    This configures parameters required to perform poisson regression.

    Parameters
    ----------
    session : tf.Session
      Tensorflow session.
    save_path : str
       Directory to write the model and training summaries
    train_biom : str or biom.Table
       Input biom table for training
    test_biom : str or biom.Table
       Input biom table for testing
    train_metadata : str or pd.DataFrame
       Input sample metadata for training.
    test_metadata : str or pd.DataFrame
       Input sample metadata for testing.
    formula : str
       Statistical formula for specifying covariates.
    learning_rate : float
       Initial learning rate
    clipping_size : float
       Gradient clipping size.
    beta_mean : float
       Mean of prior distribution for covariates
    beta_scale : float
       Scale of prior distribution for covariates
    gamma_mean : float
       Mean of prior distribution for feature bias
    gamma_scale
       Scale of prior distribution for feature bias
    epochs_to_train : int
      Number of epochs to train. Each epoch processes the training data once
    num_neg_samples : int
      Negative samples per training sample
    batch_size : int
      Number of nonzero entries processed per step (size of a minibatch).
    block_size : int
      Number of training samples processed per step.
    min_sample_count : int
      The minimum number of counts a feature needs for it to be
      included in the analysis.
    min_feature_count : int
      The minimum number of counts a sample needs to be
      included in the analysis.
    statistics_interval : int
      Time interval in seconds to print statistics.
    summary_interval : int
      Time interval in seconds to summarize statistics.
    checkpoint_interval : int
      Time interval in seconds to checkpoint model.
    verbose : boolean
      Specifies if cross validation and summaries are saved during training.

    Notes
    -----
    Note that all of the parameters are optional, since we don't
    necessarily need all of the parameters, particularly for
    testing.
    """
    self.sess = session
    for k, v in kwargs.items():
      setattr(self, k, v)

    try:
      if not os.path.exists(self.save_path):
        os.makedirs(self.save_path)

      if isinstance(self.train_biom, str):
        self.train_table = load_table(self.train_biom)
      elif isinstance(self.train_biom, Table):
        self.train_table = self.train_biom

      # D = number of features.  N = number of samples
      # num_nonzero = number of nonzero entries in the table.
      self.D, self.N = self.train_table.shape
      self.N, self.p = self.train_metadata.shape
      self.num_nonzero = self.train_table.nnz

      if isinstance(self.test_biom, str):
        self.test_table = load_table(self.test_biom)
      elif isinstance(self.test_biom, Table):
        self.test_table = self.test_biom

      if isinstance(self.train_metadata, str):
        self.train_metadata = pd.read_table(self.train_metadata, index_col=0)
      elif isinstance(self.train_metadata, pd.DataFrame):
        self.train_metadata = self.train_metadata
      if isinstance(self.test_metadata, str):
        self.test_metadata = pd.read_table(self.test_metadata, index_col=0)
      elif isinstance(self.train_metadata, pd.DataFrame):
        self.test_metadata = self.test_metadata

    except Exception as err:
      print(err)


  def sample(self, y_data):
    """ Creates a minibatch.

    Parameters
    ----------
    y_data : tf.SparseTensor
       Sparse tensor to sample

    Returns
    -------
    positive_batch
       Sparse tensor of positive examples
    random_batch
       Sparse tensor of random examples

    Notes
    -----
    For the random batch, we will care about the actual counts
    corresponding to the row and column indices.
    """
    with tf.name_scope('sample'):
      N, D, M = self.N, self.D, self.batch_size

      nnz = tf.size(y_data.values, out_type=tf.int64)

      batch_ids = tf.random_uniform([M], maxval=nnz, dtype=tf.int64)

      indeces = tf.gather(y_data.indices, batch_ids, axis=0)
      values = tf.gather(y_data.values, batch_ids)

      positive_batch = tf.SparseTensor(
        indices=tf.cast(indeces, dtype=tf.int64),
        values=tf.cast(values, tf.int32),
        dense_shape=[M, D])

      sample_rows = tf.random_uniform(
        [self.num_neg_samples], maxval=N, dtype=tf.int64)
      sample_cols = tf.random_uniform(
        [self.num_neg_samples], maxval=D, dtype=tf.int64)

      random_indices = tf.concat(
        [tf.reshape(sample_rows, [-1, 1]),
         tf.reshape(sample_cols, [-1, 1])], axis=1)
      random_values = tf.zeros([self.num_neg_samples])
      random_batch = tf.SparseTensor(
        indices=tf.cast(random_indices, dtype=tf.int64),
        values=tf.cast(random_values, tf.int32),
        dense_shape=[self.num_neg_samples, D])

      return positive_batch, random_batch


  def loss(self, G_data, y_data, positive_batch, random_batch):
    """ Computes the loss.

    Parameters
    ----------
    G_data : tf.Tensor
       Design matrix
    y_data : tf.SparseTensor
       Sparse tensor of counts
    positive_batch : tf.Tensor
       A Sparse tensor representing a batch of positive examples.
    random_batch : tf.Tensor
       A Sparse tensor representing a batch of random examples.

    Returns
    -------
    log_loss : tf.Tensor
       Tensor representing the log likelihood of the model.
    """
    with tf.name_scope('loss'):
      gamma_mean, gamma_scale = self.gamma_mean, self.gamma_scale
      beta_mean, beta_scale = self.beta_mean, self.beta_scale
      N, D, p = self.N, self.D, self.p
      num_nonzero = tf.size(y_data.values, out_type=tf.float32)

      # unpack sparse tensors
      pos_data = tf.cast(positive_batch.values, dtype=tf.float32)
      pos_row = tf.gather(positive_batch.indices, 0, axis=1)
      pos_col = tf.gather(positive_batch.indices, 1, axis=1)

      rand_row = tf.gather(random_batch.indices, 0, axis=1)
      rand_col = tf.gather(random_batch.indices, 1, axis=1)

      num_sampled = tf.size(pos_row, out_type=tf.float32)

      theta = tf.log(  # basically log total counts
        tf.cast(tf.sparse_reduce_sum(y_data, axis=1), dtype=tf.float32))

      # Regression coefficients
      qgamma = tf.Variable(tf.random_normal([1, D]), name='qgamma')
      qbeta = tf.Variable(tf.random_normal([p, D]), name='qbeta')
      self.V = tf.concat([qgamma, qbeta], axis=0, name='V')
      G = tf.concat(
        [tf.ones([G_data.shape[0], 1]), G_data],
        axis=1, name='G')

      with tf.name_scope('positive_log_prob'):
        # add bias terms for samples
        Gpos = tf.gather(G, pos_row, axis=0)
        Vpos = tf.transpose(
          tf.gather(self.V, pos_col, axis=1), name='Vprime')
        # sparse matrix multiplication for positive samples
        y_pred = tf.reduce_sum(
          tf.multiply(Gpos, Vpos), axis=1)

        theta_pos = tf.squeeze(tf.gather(theta, pos_row))
        pos_prob = tf.reduce_sum(
          tf.multiply(pos_data, y_pred + theta_pos))
        sparse_scale = num_nonzero / num_sampled

      with tf.name_scope('coefficient_log_prob'):
        Grand = tf.gather(G, rand_row, axis=0)
        Vrand = tf.transpose(
          tf.gather(self.V, rand_col, axis=1), name='Vprime')
        # sparse matrix multiplication for random indices
        y_rand = tf.reduce_sum(
          tf.multiply(Grand, Vrand), axis=1)
        theta_rand = tf.squeeze(tf.gather(theta, rand_row))
        coef_prob = tf.reduce_sum(
          tf.exp(y_rand + theta_rand)
        )
        coef_scale = N * D / self.num_neg_samples

      #pos_prob = tf.Print(pos_prob, [pos_prob])
      #coef_prob = tf.Print(coef_prob, [coef_prob])
      total_poisson = pos_prob * sparse_scale - coef_prob * coef_scale

      with tf.name_scope('priors'):
        # Normal priors (a.k.a. L2 regularization)
        # species intercepts
        gamma = Normal(loc=tf.zeros([1, D]) + gamma_mean,
                       scale=tf.ones([1, D]) * gamma_scale,
                       name='gamma')
        # regression coefficents distribution
        beta = Normal(loc=tf.zeros([p, D]) + beta_mean,
                      scale=tf.ones([p, D]) * beta_scale,
                      name='B')

        total_gamma = tf.reduce_sum(gamma.log_prob(qgamma))
        total_beta = tf.reduce_sum(beta.log_prob(qbeta))

      log_loss = - (total_gamma + total_beta + \
                    total_poisson)

      # save parameters to model
      self.qbeta = qbeta
      self.qgamma = qgamma

      return log_loss

  def optimize(self, log_loss):
    """ Perform optimization (via Gradient Descent)"""
    with tf.name_scope('optimize'):
      learning_rate = self.learning_rate
      clipping_size = self.clipping_size

      optimizer = tf.train.AdamOptimizer(learning_rate)
      gradients, variables = zip(*optimizer.compute_gradients(log_loss))
      gradients, _ = tf.clip_by_global_norm(gradients, clipping_size)

      train_ = optimizer.apply_gradients(zip(gradients, variables))
      return train_, gradients, variables

  def evaluate(self, G_holdout, Y_holdout):
    """ Perform cross validation on the hold-out set.

    This calculates the mean absolute error.

    Parameters
    ----------
    G_holdout : tf.Tensor
       Sample metadata for the hold-out test dataset
    Y_holdout : tf.Tensor
       Dense feature table for the hold-out test dataset

    Returns
    -------
    mad : tf.Tensor
       Mean absolute deviation.  This represents the average error
       for each cell value in the matrix.
    """
    with tf.name_scope('evaluate'):

      # evaluate the accuracy
      holdout_count = tf.cast(
        tf.sparse_reduce_sum(Y_holdout, axis=1), dtype=tf.float32)
      obs_ids = tf.gather(Y_holdout.indices, 1, axis=1)
      samp_ids = tf.gather(Y_holdout.indices, 0, axis=1)

      g_data = tf.gather(G_holdout, samp_ids, axis=0)

      # Calculate predicted abundance
      Gpos = tf.concat(
        [tf.ones([g_data.shape[0], 1]), g_data],
        axis=1, name='g_holdout')
      Vprime = tf.transpose(
        tf.gather(self.V, obs_ids, axis=1), name='V_holdout')
      # sparse matrix multiplication for positive samples
      y_pred = tf.reduce_sum(
        tf.multiply(
            Gpos, Vprime),
        axis=1)
      smax = tf.SparseTensorValue(
        indices=Y_holdout.indices,
        values=y_pred,
        dense_shape=Y_holdout.dense_shape)

      smax = tf.sparse_softmax(smax)

      holdout_count = tf.gather(holdout_count, samp_ids, axis=0)
      pred_values = tf.cast(tf.multiply(holdout_count, smax.values),
                            tf.float32)

      Y_values = tf.cast(Y_holdout.values, tf.float32)
      mse = tf.reduce_mean(
        tf.squeeze(tf.abs(pred_values - Y_values)))
      return mse


def main(_):

  # preprocessing (i.e. biom table, metadata, ...)
  train_biom = load_table(FLAGS.train_biom)
  test_biom = load_table(FLAGS.test_biom)
  train_metadata = pd.read_table(FLAGS.train_metadata, index_col=0)
  test_metadata = pd.read_table(FLAGS.test_metadata, index_col=0)
  formula = FLAGS.formula + '+0'
  (train_table, test_biom,
   train_metadata, test_metadata) = preprocess(
     formula,
     train_biom, train_metadata,
     test_biom, test_metadata,
     FLAGS.min_sample_count, FLAGS.min_feature_count
   )
  samp_ids = train_table.ids(axis='sample')

  obs_ids = train_table.ids(axis='observation')
  md_ids = np.array(train_metadata.columns)

  # Model code
  with tf.Graph().as_default(), tf.Session() as session:
    model = PoissonRegression(
      session, save_path=FLAGS.save_path,
      train_biom=train_biom,
      test_biom=test_biom,
      train_metadata=train_metadata,
      test_metadata=test_metadata,
      formula=formula,
      learning_rate=FLAGS.learning_rate,
      clipping_size=FLAGS.clipping_size,
      beta_mean=FLAGS.beta_mean,
      beta_scale=FLAGS.beta_scale,
      gamma_mean=FLAGS.gamma_mean,
      gamma_scale=FLAGS.gamma_scale,
      epochs_to_train=FLAGS.epochs_to_train,
      num_neg_samples=FLAGS.num_neg_samples,
      batch_size=FLAGS.batch_size,
      block_size=FLAGS.block_size,
      min_sample_count=FLAGS.min_sample_count,
      min_feature_count=FLAGS.min_feature_count,
      statistics_interval=FLAGS.statistics_interval,
      summary_interval=FLAGS.summary_interval,
      checkpoint_interval=FLAGS.checkpoint_interval
    )
    table = train_table.matrix_data.tocoo().T
    biom_test = test_biom.matrix_data.tocoo().T
    y_data = tf.SparseTensorValue(
        indices=np.array([table.row,  table.col]).T,
        values=table.data,
        dense_shape=table.shape
    )
    G_data = tf.constant(train_metadata.values, dtype=tf.float32)

    # setup cross validation data
    G_test = tf.constant(test_metadata.values, dtype=tf.float32)
    y_test = tf.SparseTensorValue(
      indices=np.array([biom_test.row,  biom_test.col]).T,
      values=biom_test.data,
      dense_shape=biom_test.shape)

    positive_batch, random_batch = model.sample(y_data)
    log_loss = model.loss(G_data, y_data, positive_batch, random_batch)
    train_step, grads, variables = model.optimize(log_loss)
    mean_err = model.evaluate(G_test, y_test)
    tf.global_variables_initializer().run()

    # summary information
    tf.summary.histogram('qbeta', model.qbeta)
    tf.summary.histogram('qgamma', model.qgamma)
    tf.summary.scalar('mean_absolute_error', mean_err)
    for i, g in enumerate(grads):
      tf.summary.histogram('gradient/%s' % variables[i], g)
    merged = tf.summary.merge_all()
    last_checkpoint_time = 0
    last_summary_time = 0
    last_statistics_time = 0

    # initialize with small minibatch
    train_, loss, err, beta, gamma = session.run(
      [train_step, log_loss, mean_err,
       model.qbeta, model.qgamma]
    )

    epoch = model.num_nonzero // model.batch_size
    num_iter = int(model.epochs_to_train * epoch)
    saver = tf.train.Saver()

    writer = tf.summary.FileWriter(model.save_path, session.graph)

    start_time = time.time()
    k = 0
    for i in tqdm(range(1, num_iter)):
      now = time.time()
      # grab the next block
      if i % model.block_size == 0:
        train_, loss, err, beta, gamma = session.run(
          [train_step, log_loss, mean_err,
            model.qbeta, model.qgamma]
        )
        k = k % model.block_size
      # check for summary
      elif now - last_summary_time > model.summary_interval:
        run_metadata = tf.RunMetadata()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        _, summary  = session.run(
          [train_step, merged],
          options=run_options,
          run_metadata=run_metadata
        )
        writer.add_summary(summary, i)
        writer.add_run_metadata(run_metadata, 'step%d' % i)
        last_summary_time = now
      elif now - last_checkpoint_time > model.checkpoint_interval:
        saver.save(
          session, os.path.join(model.save_path, "model.ckpt"),
          global_step=i)
        last_checkpoint_time = now
      else:
        train_, loss, beta, gamma = session.run(
          [train_step, log_loss,
           model.qbeta, model.qgamma]
        )

    elapsed_time = time.time() - start_time
    print('Elapsed Time: %f seconds' % elapsed_time)

    # save all parameters to the save path
    train_, loss, beta, gamma = session.run(
      [train_step, log_loss,
       model.qbeta, model.qgamma]
    )
    pd.DataFrame(
      beta, index=md_ids, columns=obs_ids,
    ).to_csv(os.path.join(model.save_path, 'beta.csv'))
    pd.DataFrame(
      gamma, index=['intercept'], columns=obs_ids,
    ).to_csv(os.path.join(model.save_path, 'gamma.csv'))

    # Run final round of cross validation
    y_test = np.array(test_biom.matrix_data.todense()).T
    G_test = test_metadata.values

    # Cross validation
    pred_beta = model.qbeta.eval()
    pred_gamma = model.qgamma.eval()

    mse, mrc = cross_validation(
      G_test, pred_beta, pred_gamma, y_test)
    print("MSE: %f, MRC: %f" % (mse, mrc))


if __name__ == "__main__":
  tf.app.run()
