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
from util import cross_validation
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
  """Options used by our Poisson Regression model."""

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

    try:
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

      self.formula = self.formula + '+0'

      if not os.path.exists(self.save_path):
        os.makedirs(self.save_path)

      if isinstance(self.tree_file, str):
        self.tree = TreeNode.read(self.tree_file)
      elif isinstance(self.tree_file, TreeNode):
        self.tree = self.tree_file
    except Exception as err:
      print(err)


class PoissonRegression(object):

  def __init__(self, options, session):
    """ Constructor for Poisson regression.

    Parameters
    ----------
    options : Options class
      Options specifying file paths.
    session : tf.Session
      Tensorflow session.
    """
    self.opts = options
    self.sess = session

  def initialize(self):
    # preprocessing (i.e. biom table, metadata, ...)
    (y_data, y_test, G_data, G_test) = self.preprocess(
      opts.formula,
      opts.train_table, opts.train_metadata,
      opts.test_table, opts.test_metadata,
      opts.min_sample_counts, opts.min_feature_count
    )

    batch = self.sample(
      y_data, num_pos=opts.batch_size, num_neg=opts.num_neg_samples
    )

    self.log_loss = self.loss(G_data, y_data, batch)
    self.train = self.optimize(self.log_loss)
    self.mse = self.evaluation_graph(G_test, y_test)

  def preprocess(self, formula,
                 train_table, train_metadata,
                 test_table, test_metadata,
                 min_sample_counts=10, min_feature_count=10):
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

    biom_data = train_table.matrix_data.tocoo().T
    y_data = tf.SparseTensorValue(
      indices=np.array([biom_data.row,  biom_data.col]).T,
      values=biom_data.data,
      dense_shape=coo_matrix.shape)
    G_data = tf.constant(train_metadata.values, dtype=tf.float32)

    G_test = tf.constant(test_metadata.values, dtype=tf.float32)
    y_test = tf.constant(np.array(test_table.matrix_data.todense()).T,
                         dtype=tf.float32)

    # D = number of features.  N = number of samples
    # num_nonzero = number of nonzero entries in the table.
    self.D, self.N = train_table.shape
    self.p = G_data.shape[1]
    self.num_nonzero = train_table.nnz

    return y_data, y_test, G_data, G_test


  def sample(self, y_data, num_pos=50, num_neg=10):
    """ Creates a minibatch made up of positive and negative examples.

    Parameters
    ----------
    y_data : tf.SparseTensor
       Sparse tensor to sample
    num_pos : int
       Number of positive examples in a batch
    num_neg : int
       Number of negative examples in a batch.

    Returns
    -------
    positive_batch
       Sparse tensor of positive examples
    negative_batch
       Sparse tensor of negative examples
    accident_batch
       Sparse tensor of accidental positive examples.
       These are examples that are claimed to be negative,
       but are actually positive.  This is corrected downstream
       in the `inference` module.  These are added to
       to the negative batch to correct the accident.
       Since Poisson(0) + Poisson(k) = Poisson(k), this should
       be equivalent.  Blame Google for this ugly hack.
    num_exp_pos
       Number of expected positive hits.  This is useful for
       scaling the minibatches appropriately.
    num_exp_neg
       Number of expected negative hits. This is useful for
       scaling the minibatches appropriately.
    """
    opts = self.opts

    # Create a negative sampling scheme
    # First create all of the true classes using row major order
    # All of the entries in the sparse matrix is a possible class
    # But only the non-zero entries are true classes
    N, D = y_data.dense_shape
    rows = tf.gather(y_data.indices, 0, axis=1)
    cols = tf.gather(y_data.indices, 1, axis=1)
    rc_major = tf.cast(rows * D + cols, dtype=tf.int32)

    nnz = y_data.indices.shape[0]

    # Collect samples from y_data
    true_batch_ids = tf.random_uniform(
      [opts.batch_size], maxval=nnz, dtype=tf.int32)
    true_ids = tf.gather(rc_major, true_batch_ids, axis=0)

    # cast the sampled results back to row, col, data tuples
    # and return the batch
    true_rows = tf.cast(tf.floordiv(true_ids, D), dtype=tf.int64,
                        name='positive_rows')
    true_cols = tf.cast(tf.floormod(true_ids, D), dtype=tf.int64,
                        name='positive_cols')
    true_data = tf.gather(y_data.values, true_batch_ids,
                          name='positive_data')

    true_labels = tf.cast(tf.reshape(true_ids, [opts.batch_size, 1]),
                          dtype=tf.int64)

    # Then run tf.nn.uniform_candidate_sampler to sample the negative samples
    sampled_ids, num_exp_pos, num_exp_neg = tf.nn.uniform_candidate_sampler(
      true_classes=true_labels,
      num_true=1,
      num_sampled=opts.num_neg_samples,
      range_max=N*D,
      unique=True
    )

    true_acc, samp_acc, w = tf.nn.compute_accidental_hits(
      true_labels, sampled_ids, num_true=1)

    # basically this will be added to the negative samples
    # to double count, this way, any positive examples that
    # were sampled as negative samples will be removed.
    hit_ids = tf.gather(rc_major, true_acc)
    hit_rows = tf.cast(tf.floordiv(hit_ids, D), dtype=tf.int64,
                       name='hit_rows')
    hit_cols = tf.cast(tf.floormod(hit_ids, D), dtype=tf.int64,
                       name='hit_cols')
    hit_data = tf.gather(y_data.values, true_acc,
                         name='hit_data')

    samp_rows = tf.cast(tf.floordiv(sampled_ids, D), dtype=tf.int64,
                        name='negative_rows')
    samp_cols = tf.cast(tf.floormod(sampled_ids, D), dtype=tf.int64,
                        name='negative_cols')
    samp_data = tf.zeros([sampled_ids.shape[0]], dtype=tf.int32,
                        name='negative_data')

    positive_batch = tf.SparseTensor(
      indices=tf.concat([tf.reshape(true_rows, [-1, 1]),
                         tf.reshape(true_cols, [-1, 1])], axis=1),
      values=true_data, dense_shape=[N, D])

    accident_batch = tf.SparseTensor(
      indices=tf.concat([tf.reshape(hit_rows, [-1, 1]),
                         tf.reshape(hit_cols, [-1, 1])], axis=1),
      values=hit_data, dense_shape=[N, D])

    negative_batch = tf.SparseTensor(
      indices=tf.concat([tf.reshape(samp_rows, [-1, 1]),
                         tf.reshape(samp_cols, [-1, 1])], axis=1),
      values=samp_data, dense_shape=[N, D])

    return (positive_batch, negative_batch, accident_batch,
            num_exp_pos, num_exp_neg)


  def inference(self, G_data, obs_id):
    """ Builds computation graph for the model.

    Parameters
    ----------
    G_data : tf.Tensor
       Design matrix of covariates over samples.
       Covariates are columns and rows are samples.
    obs_id : tf.Tensor
       Observation id.  This also corresponds to the
       column id in the sparse matrix.

    Returns
    -------
    y_pred : tf.Tensor
       Predicted counts
    """
    # unpack batches
    opts = self.opts

    # more preprocessing
    # how to actually get this???
    p = self.p
    N, D = self.N, self.D

    pos_col = obs_id

    batch_size = opts.batch_size
    gamma_mean, gamma_scale = opts.gamma_mean, opts.gamma_scale
    beta_mean, beta_scale = opts.beta_mean, opts.beta_scale

    # Regression coefficients
    qgamma = tf.Variable(tf.random_normal([1, D]), name='qgamma')
    qbeta = tf.Variable(tf.random_normal([p, D]), name='qbeta')

    # sample bias (for overdispersion)
    theta = tf.Variable(tf.random_normal([N, 1]), name='theta')
    self.V = tf.concat([qgamma, qbeta], axis=0, name='V')

    # add bias terms for samples
    Gpos = tf.concat(
      [tf.ones([batch_size, 1]), G_data],
      axis=1, name='Gpos')

    Vprime = tf.transpose(
      tf.gather(self.V, pos_col, axis=1), name='Vprime')
    # sparse matrix multiplication for positive samples
    pos_prime = tf.reduce_sum(
      tf.multiply(
          Gpos, Vprime),
      axis=1)
    y_pred = pos_prime

    # save parameters to model
    self.qbeta = qbeta
    self.qgamma = qgamma
    self.theta = theta

    return y_pred


  def loss(self, G_data, y_data, batch):
    """ Computes the loss

    Parameters
    ----------
    G_data : tf.Tensor
       Design matrix
    y_data : tf.SparseTensor
       Sparse tensor of counts
    batch : tuple of results tf.Tensor
       The output from sample().  The tuple is decomposed as follows

       positive_batch : tf.SparseTensor
          Sparse tensor of positive examples
       negative_batch : tf.SparseTensor
          Sparse tensor of negative examples
       accident_batch : tf.SparseTensor
          Sparse tensor of accidental positive examples.
          These are examples that are claimed to be negative,
          but are actually positive.  This is corrected downstream
          in the `inference` module.  These are added to
          to the negative batch to correct the accident.
          Since Poisson(0) + Poisson(k) = Poisson(k), this should
          be equivalent.  Blame Google for this ugly hack.
       num_exp_pos : int
          Number of expected positive hits.  This is useful for
          scaling the minibatches appropriately.
       num_exp_neg : int
          Number of expected negative hits. This is useful for
          scaling the minibatches appropriately.
    """
    opts = self.opts

    (positive_batch, negative_batch, accident_batch,
     num_exp_pos, num_exp_neg) = batch
    gamma_mean, gamma_scale = opts.gamma_mean, opts.gamma_scale
    beta_mean, beta_scale = opts.beta_mean, opts.beta_scale
    N, D, p = self.N, self.D, self.p
    num_nonzero = self.num_nonzero
    # unpack sparse tensors
    pos_data = positive_batch.values  # nonzero examples
    pos_row = tf.gather(positive_batch.indices, 0, axis=1)
    pos_col = tf.gather(positive_batch.indices, 1, axis=1)
    neg_data = negative_batch.values  # zero examples
    neg_row = tf.gather(negative_batch.indices, 0, axis=1)
    neg_col = tf.gather(negative_batch.indices, 1, axis=1)
    acc_data = accident_batch.values  # accident examples
    acc_row = tf.gather(accident_batch.indices, 0, axis=1)
    acc_col = tf.gather(accident_batch.indices, 1, axis=1)

    batch_size, num_neg = pos_row.shape[0], neg_row.shape[0]

    # obtain prediction to then calculate loss
    Gpos = tf.gather(G_data, pos_row, axis=0)
    y_pred = self.inference(Gpos, pos_col)
    qbeta, qgamma, theta = self.qbeta, self.qgamma, self.theta

    # Actual calculation of loss is below.
    # add sample bias
    y_pred += tf.reshape(tf.gather(theta, pos_row), shape=[batch_size])

    total_zero = tf.constant(N*D - num_nonzero,
                             dtype=tf.float32)
    total_nonzero = tf.constant(num_nonzero, dtype=tf.float32)

    pos_poisson = Poisson(log_rate=y_pred, name='Y')

    # Distributions species bias
    gamma = Normal(loc=tf.zeros([1, D]) + gamma_mean,
                   scale=tf.ones([1, D]) * gamma_scale,
                   name='gamma')
    # regression coefficents distribution
    beta = Normal(loc=tf.zeros([p, D]) + beta_mean,
                  scale=tf.ones([p, D]) * beta_scale,
                  name='B')

    # sparse matrix multiplication for negative samples
    Gneg = tf.gather(G_data, neg_row, axis=0)
    Gneg = tf.concat([tf.ones([num_neg, 1]), Gneg], axis=1)
    neg_prime = tf.reduce_sum(
      tf.multiply(
          Gneg, tf.transpose(
              tf.gather(self.V, neg_col, axis=1))),
      axis=1)
    neg_phi = tf.reshape(tf.gather(theta, neg_row), shape=[num_neg]) + neg_prime
    neg_poisson = Poisson(log_rate=neg_phi, name='neg_counts')

    # accident samples
    num_acc = tf.shape(accident_batch.indices)[0]
    Gacc = tf.gather(G_data, acc_row, axis=0)
    Gacc = tf.concat([tf.ones([num_acc, 1]), Gacc], axis=1)
    acc_prime = tf.reduce_sum(
      tf.multiply(
          Gacc, tf.transpose(
              tf.gather(self.V, acc_col, axis=1))),
      axis=1)
    #acc_phi = tf.gather(theta, acc_row) + acc_prime
    acc_phi = tf.reshape(tf.gather(theta, acc_row), shape=[num_acc]) + acc_prime
    acc_poisson = Poisson(log_rate=acc_phi, name='acc_counts')

    pos_data = tf.cast(pos_data, dtype=tf.float32)
    neg_data = tf.cast(neg_data, dtype=tf.float32)
    acc_data = tf.cast(acc_data, dtype=tf.float32)

    log_loss = -(
      tf.reduce_sum(gamma.log_prob(qgamma)) + \
      tf.reduce_sum(beta.log_prob(qbeta)) + \
      tf.reduce_sum(pos_poisson.log_prob(y_pred)) * (total_nonzero / num_exp_pos) + \
      (tf.reduce_sum(neg_poisson.log_prob(neg_data)) + \
       tf.reduce_sum(acc_poisson.log_prob(acc_data))) * (total_zero / num_exp_neg)
    )
    return log_loss

  def optimize(self, log_loss):
    opts = self.options

    learning_rate = opts.learning_rate
    clipping_size=opts.clipping_size

    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(log_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, clipping_size)
    train = optimizer.apply_gradients(zip(gradients, variables))
    return train

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
    qbeta = self.qbeta
    qgamma = self.qgamma

    # evaluate the accuracy
    holdout_count = tf.reduce_sum(Y_holdout, axis=1)
    pred =  tf.reshape(holdout_count, [-1, 1]) * tf.nn.softmax(
      tf.matmul(G_holdout, qbeta) + qgamma)

    mse = tf.reduce_mean(tf.squeeze(tf.abs(pred - Y_holdout)))
    tf.summary.scalar('mean_absolute_error', mse)
    return mse

  def train(self):
    """ Trains a single batch """
    opts = self.opts
    batch_size = opts.batch_size
    checkpoint_interval = opts.checkpoint_interval

    (positive_batch, negative_batch, accident_batch,
     num_exp_pos, num_exp_neg) = batch(self.y_data)

    if i % 1000 == 0:
      # store runtime information
      _, summary, train_loss = session.run(
          [self.train, merged, loss],
          feed_dict=feed_dict,
          options=run_options,
          run_metadata=run_metadata
      )
    elif i % 5000 == 0:
      # store loss information and cross-validation information
      _, summary, err, train_loss = session.run(
        [train, mse, merged, loss],
        feed_dict=feed_dict
      )
      writer.add_summary(summary, i)
    else:
      _ = session.run(
          [train],
          feed_dict=feed_dict
      )
      writer.add_summary(summary, i)

    now = time.time()
    if now - self.last_checkpoint_time > checkpoint_interval:
      saver.save(session,
                 os.path.join(opts.save_path, "model.ckpt"),
                 global_step=i)
      self.last_checkpoint_time = now


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
  save_path = opts.save_path
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
    model = PoissonRegression(options, session)

    # where to put this?
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('qbeta', qbeta)
    tf.summary.histogram('qgamma', qgamma)
    tf.summary.histogram('theta', theta)
    merged = tf.summary.merge_all()

    tf.global_variables_initializer().run()

    # where to put this?
    writer = tf.summary.FileWriter(save_path, session.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    losses = np.array([0.] * num_iter)
    idx = np.arange(train_metadata.shape[0])
    log_handle = open(os.path.join(save_path, 'run.log'), 'w')
    gen = get_batch(batch_size,
                    N, D,
                    y_data.data,
                    y_data.row,
                    y_data.col,
                    num_neg=num_neg)
    start_time = time.time()
    last_checkpoint_time = 0
    saver = tf.train.Saver()
    for i in range(num_iter):
      model.train()

    elapsed_time = time.time() - start_time
    print('Elapsed Time: %f seconds' % elapsed_time)

    # Cross validation
    pred_beta = qbeta.eval()
    pred_gamma = qgamma.eval()
    mse, mrc = cross_validation(test_metadata.values, pred_beta, pred_gamma, y_test)
    print("MSE: %f, MRC: %f" % (mse, mrc))


if __name__ == "__main__":
  tf.app.run()

# Two parts to tensorflow (look at convolutional.py)
# __init__ = instance of class
#            (set the configuration parameters)
# __call__ = invoke the actual network
#            (pass in the session + runtime parameters)
#
