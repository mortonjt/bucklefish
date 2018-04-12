import tensorflow as tf
import os
import numpy as np
import pandas as pd
from biom import load_table, Table

from gneiss.balances import _balance_basis
from gneiss.composition import ilr_transform
from gneiss.util import match, rename_internal_nodes

from tensorflow.contrib.distributions import Poisson, Normal
from patsy import dmatrix
from skbio import TreeNode
from skbio.stats.composition import closure, clr_inv
from scipy.stats import spearmanr
from util import match_tips, sparse_balance_basis
from util import cross_validation, get_batch
from skbio.stats.composition import clr_inv
import time


flags = tf.app.flags
flags.DEFINE_string("save_path", None, "Directory to write the model and "
                    "training summaries.")
flags.DEFINE_string("train_biom", None, "Input biom table. "
                    "i.e. input.biom")
flags.DEFINE_string("test_biom", None, "Input biom table. "
                    "i.e. input.biom")
flags.DEFINE_string("train_metadata", None, "Input sample metadata. "
                    "i.e. metadata.txt")
flags.DEFINE_string("test_metadata", None, "Input sample metadata. "
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
                   'Mean of prior distribution for sample bias')
flags.DEFINE_float("gamma_scale", 1.0,
                   'Scale of prior distribution for sample bias')
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training sample.")
flags.DEFINE_integer("batch_size", 512,
                     "Number of training samples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("min_sample_count", 1000,
                     "The minimum number of counts a feature needs for it to be "
                     "included in the analysis")
flags.DEFINE_integer("min_feature_count", 5,
                     "The minimum number of counts a sample needs to be  "
                     "included in the analysis")
flags.DEFINE_integer("statistics_interval", 5,
                     "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5,
                     "Save training summary to file every n seconds (rounded "
                     "up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 600,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "seconds (rounded up to statistics interval).")
flags.DEFINE_boolean("verbose", False,
                     "Specifies if cross validation and summaries "
                     "are saved during training. ")
FLAGS = flags.FLAGS


class Options(object):
  """Options used by our Poisson Niche model."""

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

    if isinstance(self.train_biom, str):
      self.train_table = load_table(self.train_biom)
    elif isinstance(self.train_biom, Table):
      self.train_table = self.train_biom
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

    try:
      if isinstance(self.tree, str):
        self.tree = TreeNode.read(self.tree)
      elif isinstance(self.tree, TreeNode):
        self.tree = self.tree
    except:
      pass

    self.formula = self.formula + "+0"
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)



def main(_):

  opts = Options(
    save_path=FLAGS.save_path,
    train_biom=FLAGS.train_biom,
    test_biom=FLAGS.test_biom,
    train_metadata=FLAGS.train_metadata,
    test_metadata=FLAGS.test_metadata,
    formula=FLAGS.formula,
    tree=FLAGS.tree,
    learning_rate=FLAGS.learning_rate,
    clipping_size=FLAGS.clipping_size,
    beta_mean=FLAGS.beta_mean,
    beta_scale=FLAGS.beta_scale,
    gamma_mean=FLAGS.gamma_mean,
    gamma_scale=FLAGS.gamma_scale,
    epochs_to_train=FLAGS.epochs_to_train,
    num_neg_samples=FLAGS.num_neg_samples,
    batch_size=FLAGS.batch_size,
    min_sample_count=FLAGS.min_sample_count,
    min_feature_count=FLAGS.min_feature_count,
    statistics_interval=FLAGS.statistics_interval,
    summary_interval=FLAGS.summary_interval,
    checkpoint_interval=FLAGS.checkpoint_interval
  )

  # preprocessing
  train_table, train_metadata = opts.train_table, opts.train_metadata
  train_metadata = train_metadata.loc[train_table.ids(axis='sample')]

  sample_filter = lambda val, id_, md: (
    (id_ in train_metadata.index) and np.sum(val) > opts.min_sample_count)
  read_filter = lambda val, id_, md: np.sum(val) > opts.min_feature_count
  metadata_filter = lambda val, id_, md: id_ in train_metadata.index

  train_table = train_table.filter(metadata_filter, axis='sample')
  train_table = train_table.filter(sample_filter, axis='sample')
  train_table = train_table.filter(read_filter, axis='observation')

  sort_f = lambda xs: [xs[train_metadata.index.get_loc(x)] for x in xs]
  train_table = train_table.sort(sort_f=sort_f, axis='sample')
  train_metadata = dmatrix(opts.formula, train_metadata, return_type='dataframe')
  tree = opts.tree
  train_table, tree = match_tips(train_table, tree)
  basis, _ = sparse_balance_basis(tree)
  basis = basis.T

  # hold out data preprocessing
  test_table, test_metadata = opts.test_table, opts.test_metadata
  metadata_filter = lambda val, id_, md: id_ in test_metadata.index
  obs_lookup = set(train_table.ids(axis='observation'))
  feat_filter = lambda val, id_, md: id_ in obs_lookup
  test_table = test_table.filter(metadata_filter, axis='sample')
  test_table = test_table.filter(feat_filter, axis='observation')

  sort_f = lambda xs: [xs[test_metadata.index.get_loc(x)] for x in xs]
  test_table = test_table.sort(sort_f=sort_f, axis='sample')
  test_metadata = dmatrix(opts.formula, test_metadata,
                          return_type='dataframe')
  test_table, tree = match_tips(test_table, tree)

  p = train_metadata.shape[1]   # number of covariates
  G_data = train_metadata.values
  y_data = train_table.matrix_data.tocoo().T
  N, D = y_data.shape
  save_path = opts.save_path
  learning_rate = opts.learning_rate
  batch_size = opts.batch_size
  gamma_mean, gamma_scale = opts.gamma_mean, opts.gamma_scale
  beta_mean, beta_scale = opts.beta_mean, opts.beta_scale
  num_iter = (N // batch_size) * opts.epochs_to_train
  num_neg = opts.num_neg_samples

  # Model code
  with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):

      # Place holder variables to accept input data
      Gpos_ph = tf.placeholder(tf.float32, [batch_size, p], name='G_pos')
      Gneg_ph = tf.placeholder(tf.float32, [num_neg, p], name='G_neg')
      Y_ph = tf.placeholder(tf.float32, [batch_size], name='Y_ph')
      pos_row = tf.placeholder(tf.int32, shape=[batch_size], name='pos_row')
      pos_col = tf.placeholder(tf.int32, shape=[batch_size], name='pos_col')
      neg_row = tf.placeholder(tf.int32, shape=[num_neg], name='neg_row')
      neg_col = tf.placeholder(tf.int32, shape=[num_neg], name='neg_col')
      neg_data = tf.zeros(shape=[num_neg], name='neg_data', dtype=tf.float32)
      total_zero = tf.constant(y_data.shape[0] * y_data.shape[1] - y_data.nnz,
                               dtype=tf.float32)
      total_nonzero = tf.constant(y_data.nnz, dtype=tf.float32)

      # Define PointMass Variables first
      qgamma = tf.Variable(tf.random_normal([1, D-1]), name='qgamma')
      qbeta = tf.Variable(tf.random_normal([p, D-1]), name='qB')
      theta = tf.Variable(tf.random_normal([N, 1]), name='theta')

      # Distributions species bias
      gamma = Normal(loc=tf.zeros([1, D-1]) + gamma_mean,
                     scale=tf.ones([1, D-1]) * gamma_scale,
                     name='gamma')
      # regression coefficents distribution
      beta = Normal(loc=tf.zeros([p, D-1]) + beta_mean,
                    scale=tf.ones([p, D-1]) * beta_scale,
                    name='B')
      Bprime = tf.concat([qgamma, qbeta], axis=0)

      # Add bias terms for samples
      Gpos = tf.concat([tf.ones([batch_size, 1]), Gpos_ph], axis=1)
      Gneg = tf.concat([tf.ones([num_neg, 1]), Gneg_ph], axis=1)

      # Convert basis to SparseTensor
      psi = tf.SparseTensor(
          indices=np.mat([basis.row, basis.col]).transpose(),
          values=basis.data,
          dense_shape=basis.shape)

      V = tf.transpose(
          tf.sparse_tensor_dense_matmul(psi, tf.transpose(Bprime))
      )

      # sparse matrix multiplication for positive samples
      pos_prime = tf.reduce_sum(
          tf.multiply(
              Gpos, tf.transpose(
                  tf.gather(V, pos_col, axis=1))),
          axis=1)
      pos_phi = tf.reshape(tf.gather(theta, pos_row),
                           shape=[batch_size]) + pos_prime
      Y = Poisson(log_rate=pos_phi, name='Y')

      # sparse matrix multiplication for negative samples
      neg_prime = tf.reduce_sum(
          tf.multiply(
              Gneg, tf.transpose(
                  tf.gather(V, neg_col, axis=1))),
          axis=1)
      neg_phi = tf.reshape(tf.gather(theta, neg_row),
                           shape=[num_neg]) + neg_prime
      neg_poisson = Poisson(log_rate=neg_phi, name='neg_counts')

      loss = -(
          tf.reduce_mean(gamma.log_prob(qgamma)) + \
          tf.reduce_mean(beta.log_prob(qbeta)) + \
          tf.reduce_mean(Y.log_prob(Y_ph)) + \
          tf.reduce_mean(neg_poisson.log_prob(neg_data))
      )

      optimizer = tf.train.AdamOptimizer(learning_rate)
      gradients, variables = zip(*optimizer.compute_gradients(loss))
      gradients, _ = tf.clip_by_global_norm(gradients, opts.clipping_size)
      train = optimizer.apply_gradients(zip(gradients, variables))

      tf.summary.scalar('loss', loss)
      tf.summary.histogram('qbeta', qbeta)
      tf.summary.histogram('qgamma', qgamma)
      tf.summary.histogram('theta', theta)
      merged = tf.summary.merge_all()

      tf.global_variables_initializer().run()

      writer = tf.summary.FileWriter(save_path, session.graph)
      losses = np.array([0.] * num_iter)
      idx = np.arange(train_metadata.shape[0])
      log_handle = open(os.path.join(save_path, 'run.log'), 'w')
      start_time = time.time()
      for i in range(num_iter):
        batch_idx = np.random.choice(idx, size=batch_size)
        batch = get_batch(batch_size, y_data, num_neg=num_neg)
        (positive_row, positive_col, positive_data,
         negative_row, negative_col, negative_data) = batch
        feed_dict={
          Y_ph: positive_data,
          Gpos_ph: G_data[positive_row, :],
          Gneg_ph: G_data[negative_row, :],
          pos_row: positive_row,
          pos_col: positive_col,
          neg_row: negative_row,
          neg_col: negative_col
        }

        _, summary, train_loss, grads = session.run(
          [train, merged, loss, gradients],
          feed_dict=feed_dict
        )
        writer.add_summary(summary, i)
        losses[i] = train_loss
      elapsed_time = time.time() - start_time
      print('Elapsed Time: %f seconds' % elapsed_time)

      # Cross validation
      y_test = np.array(test_table.matrix_data.todense()).T

      pred_beta = qbeta.eval()
      pred_gamma = qgamma.eval()
      mse, mrc = cross_validation(test_metadata.values, pred_beta @ basis.T,
                                  pred_gamma @ basis.T, y_test)
      print("MSE: %f, MRC: %f" % (mse, mrc))


if __name__ == "__main__":
  tf.app.run()
