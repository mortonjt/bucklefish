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
from util import cross_validation, subsampler
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
      if isinstance(self.tree_file, str):
        self.tree = TreeNode.read(self.tree_file)
      elif isinstance(self.tree_file, TreeNode):
        self.tree = self.tree_file
    except:
      pass

    self.formula = self.formula + '+0'

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

  sample_filter = lambda val, id_, md: (
    (id_ in train_metadata.index) and np.sum(val) > opts.min_sample_count)
  read_filter = lambda val, id_, md: np.sum(val) > opts.min_feature_count
  train_table = train_table.filter(sample_filter, axis='sample')
  train_table = train_table.filter(read_filter, axis='observation')
  train_metadata = dmatrix(opts.formula, train_metadata, return_type='dataframe')
  train_table, train_metadata = match(train_table, train_metadata)

  # hold out data preprocessing
  test_table, test_metadata = opts.test_table, opts.test_metadata
  metadata_filter = lambda val, id_, md: id_ in test_metadata.index
  obs_lookup = set(train_table.ids(axis='observation'))
  feat_filter = lambda val, id_, md: id_ in obs_lookup
  test_table = test_table.filter(metadata_filter, axis='sample')
  test_table = test_table.filter(feat_filter, axis='observation')
  test_metadata = dmatrix(opts.formula, test_metadata, return_type='dataframe')
  test_table, test_metadata = match(test_table, test_metadata)

  # pad extra columns with zeros, so that we can still make predictions
  extra_columns = list(set(train_metadata.columns) - set(test_metadata.columns))
  df = pd.DataFrame({C: np.zeros(test_metadata.shape[0])
                     for C in extra_columns}, index=test_metadata.index)

  test_metadata = pd.concat((test_metadata, df), axis=1)

  p = train_metadata.shape[1]   # number of covariates
  G_data = train_metadata.values
  y_data = train_table.matrix_data.tocoo().T
  y_test = np.array(test_table.matrix_data.todense()).T
  N, D = y_data.shape
  save_path = opts.save_path
  learning_rate = opts.learning_rate
  batch_size = opts.batch_size
  gamma_mean, gamma_scale = opts.gamma_mean, opts.gamma_scale
  beta_mean, beta_scale = opts.beta_mean, opts.beta_scale
  num_neg = opts.num_neg_samples
  clipping_size=opts.clipping_size

  epoch = y_data.nnz // batch_size
  num_iter = int(opts.epochs_to_train * epoch)
  holdout_size = test_metadata.shape[0]
  checkpoint_interval = opts.checkpoint_interval

  # Model code
  with tf.Graph().as_default(), tf.Session() as session:

    Gpos_ph = tf.placeholder(tf.float32, [batch_size, p], name='G_pos')
    Gneg_ph = tf.placeholder(tf.float32, [num_neg, p], name='G_neg')
    G_holdout = tf.placeholder(tf.float32, [holdout_size, p], name='G_holdout')
    Y_holdout = tf.placeholder(tf.float32, [holdout_size, D], name='Y_holdout')

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
    qgamma = tf.Variable(tf.random_normal([1, D]), name='qgamma')
    # sample bias (for overdispersion)
    theta = tf.Variable(tf.random_normal([N, 1]), name='theta')
    qbeta = tf.Variable(tf.random_normal([p, D]), name='qB')

    # Distributions
    # species bias
    gamma = Normal(loc=tf.zeros([1, D]) + gamma_mean,
                 scale=tf.ones([1, D]) * gamma_scale,
                 name='gamma')
    # regression coefficents distribution
    beta = Normal(loc=tf.zeros([p, D]) + beta_mean,
                scale=tf.ones([p, D]) * beta_scale,
                name='B')

    V = tf.concat([qgamma, qbeta], axis=0)

    # add bias terms for samples
    Gpos = tf.concat([tf.ones([batch_size, 1]), Gpos_ph], axis=1)
    Gneg = tf.concat([tf.ones([num_neg, 1]), Gneg_ph], axis=1)

    # sparse matrix multiplication for positive samples
    pos_prime = tf.reduce_sum(
      tf.multiply(
          Gpos, tf.transpose(
              tf.gather(V, pos_col, axis=1))),
      axis=1)
    pos_phi = tf.reshape(tf.gather(theta, pos_row), shape=[batch_size]) + pos_prime

    Y = Poisson(log_rate=pos_phi, name='Y')

    # sparse matrix multiplication for negative samples
    neg_prime = tf.reduce_sum(
      tf.multiply(
          Gneg, tf.transpose(
              tf.gather(V, neg_col, axis=1))),
      axis=1)
    neg_phi = tf.reshape(tf.gather(theta, neg_row), shape=[num_neg]) + neg_prime
    neg_poisson = Poisson(log_rate=neg_phi, name='neg_counts')

    loss = -(
        tf.reduce_sum(gamma.log_prob(qgamma)) + \
        tf.reduce_sum(beta.log_prob(qbeta)) + \
        tf.reduce_sum(Y.log_prob(Y_ph)) * (total_nonzero / batch_size) + \
        tf.reduce_sum(neg_poisson.log_prob(neg_data)) * (total_zero / num_neg)
    )

    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, clipping_size)
    train = optimizer.apply_gradients(zip(gradients, variables))

    with tf.name_scope('accuracy'):
      holdout_count = tf.reduce_sum(Y_holdout, axis=1)
      pred =  tf.reshape(holdout_count, [-1, 1]) * tf.nn.softmax(
        tf.matmul(G_holdout, qbeta) + qgamma)
      mse = tf.reduce_mean(tf.squeeze(tf.abs(pred - Y_holdout)))
      tf.summary.scalar('mean_absolute_error', mse)

    tf.summary.scalar('loss', loss)
    tf.summary.histogram('qbeta', qbeta)
    tf.summary.histogram('qgamma', qgamma)
    tf.summary.histogram('theta', theta)
    merged = tf.summary.merge_all()

    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter(save_path, session.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    losses = np.array([0.] * num_iter)
    idx = np.arange(train_metadata.shape[0])
    log_handle = open(os.path.join(save_path, 'run.log'), 'w')
    gen = subsampler(y_data,
                     num_pos=batch_size,
                     num_neg=num_neg)
    start_time = time.time()
    last_checkpoint_time = 0
    saver = tf.train.Saver()
    for i in range(num_iter):
        batch_idx = np.random.choice(idx, size=batch_size)
        batch = next(gen)
        (positive_row, positive_col, positive_data,
         negative_row, negative_col) = batch
        feed_dict={
            Y_ph: positive_data,
            Y_holdout: y_test.astype(np.float32),
            G_holdout: test_metadata.values.astype(np.float32),
            Gpos_ph: G_data[positive_row, :],
            Gneg_ph: G_data[negative_row, :],
            pos_row: positive_row,
            pos_col: positive_col,
            neg_row: negative_row,
            neg_col: negative_col
        }
        if i % 1000 == 0:
          _, summary, train_loss, grads = session.run(
              [train, merged, loss, gradients],
              feed_dict=feed_dict,
              options=run_options,
              run_metadata=run_metadata
          )
        elif i % 5000 == 0:
          _, summary, err, train_loss, grads = session.run(
            [train, mse, merged, loss, gradients],
            feed_dict=feed_dict
          )
          writer.add_summary(summary, i)
        else:
          _, summary, train_loss, grads = session.run(
              [train, merged, loss, gradients],
              feed_dict=feed_dict
          )
          writer.add_summary(summary, i)

        now = time.time()
        if now - last_checkpoint_time > checkpoint_interval:
          saver.save(session,
                     os.path.join(opts.save_path, "model.ckpt"),
                     global_step=i)
          last_checkpoint_time = now

        losses[i] = train_loss

    elapsed_time = time.time() - start_time
    print('Elapsed Time: %f seconds' % elapsed_time)

    # Cross validation
    pred_beta = qbeta.eval()
    pred_gamma = qgamma.eval()
    mse, mrc = cross_validation(test_metadata.values, pred_beta, pred_gamma, y_test)
    print("MSE: %f, MRC: %f" % (mse, mrc))


if __name__ == "__main__":
  tf.app.run()
