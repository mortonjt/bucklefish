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
# TODO: Add in block size parameter
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training sample.")
flags.DEFINE_integer("batch_size", 512,
                     "Number of nonzero entries processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("block_size", 10,
                     "Number of training samples processed per step ")
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


class Options(object):
  """Options used by our Poisson Regression model."""

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

    try:

      if not os.path.exists(self.save_path):
        os.makedirs(self.save_path)

      self.formula = self.formula + '+0'

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

    except Exception as err:
      print(err)

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

  def __init__(self, options, session):
    """ Constructor for Poisson regression.

    Parameters
    ----------
    options : Options class
      Options specifying file paths.
    session : tf.Session
      Tensorflow session.

    Notes
    -----
    TODO: Separate this constructor into two constructors, i.e.

    __init__(options)
        ...

    __call__(session)
        ...
    """
    self.opts = options
    self.sess = session


  def initialize(self):
    """ Configures model with specified parameters. """
    options = self.opts
    # D = number of features.  N = number of samples
    # num_nonzero = number of nonzero entries in the table.
    self.D, self.N = options.train_table.shape
    self.M = options.block_size

  def retrieve(self, train_table, train_metadata):
    """ Subsample blocks for training.

    This is not exactly like mini-batching, but it will retreive a block
    in the event that the block cannot fit in memory.

    Parameters
    ----------
    train_table : biom.Table
       Biom table containing the feature counts within the training dataset.
    train_metadata : pd.DataFrame
       Sample metadata table containing all of the measured covariates in
       the training dataset.

    Yields
    ------
    y_feed : tf.SparseTensorValue
       Sparse Tensor representing small minibatch
    G_feed : tf.placeholder
    """
    opts = self.opts
    # partition the biom table into multiple chunks and yield
    # the next chunk at retreival
    k = opts.block_size
    while True:
      table = train_table.subsample(opts.block_size, by_id=True)
      md = train_metadata.loc[table.ids(axis='sample')]

      ids = table.ids(axis='sample')
      table = table.matrix_data.tocoo().T

      y_feed = tf.SparseTensorValue(
        indices=np.array([table.row,  table.col]).T,
        values=table.data,
        dense_shape=table.shape)
      G_feed = md.values.astype(np.float32)
      yield y_feed, G_feed


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
    with tf.name_scope('sample'):
      opts = self.opts
      D, M = self.D, self.M
      nnz = tf.size(y_data.values, out_type=tf.int32)

      # Create a negative sampling scheme
      # First create all of the true classes using row major order
      # All of the entries in the sparse matrix is a possible class
      # But only the non-zero entries are true classes
      rows = tf.cast(tf.gather(y_data.indices, 0, axis=1), dtype=tf.int32)
      cols = tf.cast(tf.gather(y_data.indices, 1, axis=1), dtype=tf.int32)

      rows = tf.Print(rows, [rows])
      #cols = tf.Print(cols, [cols])
      rowD = rows * D
      rowD = tf.Print(rowD, [rowD])
      rc_major = tf.cast(rowD + cols, dtype=tf.int32)
      #rc_major = tf.Print(rc_major, [rc_major])
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
      # TODO: This line breaks because of overflow.
      sampled_ids, num_exp_pos, num_exp_neg = tf.nn.uniform_candidate_sampler(
        true_classes=true_labels,
        num_true=1,
        num_sampled=opts.num_neg_samples,
        range_max=M * D,
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
      yvals = y_data.values
      hit_data = tf.gather(yvals, true_acc,
                           name='hit_data')

      samp_rows = tf.cast(tf.floordiv(sampled_ids, D), dtype=tf.int64,
                          name='negative_rows')
      samp_cols = tf.cast(tf.floormod(sampled_ids, D), dtype=tf.int64,
                          name='negative_cols')
      samp_data = tf.zeros([sampled_ids.shape[0]], dtype=tf.int64,
                          name='negative_data')

      positive_batch = tf.SparseTensor(
        indices=tf.concat([tf.reshape(true_rows, [-1, 1]),
                           tf.reshape(true_cols, [-1, 1])], axis=1),
        values=true_data, dense_shape=[M, D])

      accident_batch = tf.SparseTensor(
        indices=tf.concat([tf.reshape(hit_rows, [-1, 1]),
                           tf.reshape(hit_cols, [-1, 1])], axis=1),
        values=hit_data, dense_shape=[M, D])

      negative_batch = tf.SparseTensor(
        indices=tf.concat([tf.reshape(samp_rows, [-1, 1]),
                           tf.reshape(samp_cols, [-1, 1])], axis=1),
        values=samp_data, dense_shape=[M, D])

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
    with tf.name_scope('inference'):
      # unpack batches
      opts = self.opts

      # more preprocessing
      p = self.p
      N, D = self.N, self.D
      pos_col = obs_id

      # Regression coefficients
      qgamma = tf.Variable(tf.random_normal([1, D]), name='qgamma')
      qbeta = tf.Variable(tf.random_normal([p, D]), name='qbeta')

      # sample bias (for overdispersion)
      self.V = tf.concat([qgamma, qbeta], axis=0, name='V')

      # add bias terms for samples
      Gpos = tf.concat(
        [tf.ones([G_data.shape[0], 1]), G_data],
        axis=1, name='Gpos')

      Vprime = tf.transpose(
        tf.gather(self.V, pos_col, axis=1), name='Vprime')
      # sparse matrix multiplication for positive samples
      pos_prime = tf.reduce_sum(
        tf.multiply(Gpos, Vprime), axis=1)
      y_pred = pos_prime

      # save parameters to model
      self.qbeta = qbeta
      self.qgamma = qgamma

      return y_pred


  def loss(self, G_data, y_data, batch):
    """ Computes the loss.

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
    with tf.name_scope('loss'):
      opts = self.opts
      (positive_batch, negative_batch, accident_batch,
       num_exp_pos, num_exp_neg) = batch
      gamma_mean, gamma_scale = opts.gamma_mean, opts.gamma_scale
      beta_mean, beta_scale = opts.beta_mean, opts.beta_scale
      N, D, p = self.N, self.D, self.p
      num_nonzero = tf.size(y_data.values, out_type=tf.float32)

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
      batch_size, num_sampled = opts.batch_size, opts.num_neg_samples

      # obtain prediction to then calculate loss
      Gpos = tf.gather(G_data, pos_row, axis=0)
      y_pred = self.inference(Gpos, pos_col)
      theta = tf.log(
        tf.cast(tf.sparse_reduce_sum(y_data, axis=1), dtype=tf.float32))
      qbeta, qgamma = self.qbeta, self.qgamma

      # Actual calculation of loss is below.
      # Adding sample bias
      y_pred += tf.reshape(tf.gather(theta, pos_row), shape=[batch_size])
      total_zero = tf.constant(N*D, dtype=tf.float32) - num_nonzero
      total_nonzero = num_nonzero
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
      Gneg = tf.concat([tf.ones([num_sampled, 1]), Gneg], axis=1)
      neg_prime = tf.reduce_sum(
        tf.multiply(
            Gneg, tf.transpose(
                tf.gather(self.V, neg_col, axis=1))),
        axis=1)
      neg_phi = tf.reshape(tf.gather(theta, neg_row),
                           shape=[num_sampled]) + neg_prime
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
      acc_phi = tf.reshape(tf.gather(theta, acc_row),
                           shape=[num_acc]) + acc_prime
      acc_poisson = Poisson(log_rate=acc_phi, name='acc_counts')

      pos_data = tf.cast(pos_data, dtype=tf.float32)
      neg_data = tf.cast(neg_data, dtype=tf.float32)
      acc_data = tf.cast(acc_data, dtype=tf.float32)

      num_acc = tf.cast(tf.size(acc_data), tf.float32)
      num_pos = batch_size + num_acc
      num_neg = num_sampled - num_acc

      pos_prob = pos_poisson.log_prob(pos_data)
      neg_prob = neg_poisson.log_prob(neg_data)
      acc_prob = acc_poisson.log_prob(acc_data)

      total_pos = tf.reduce_sum(pos_prob)
      total_acc = tf.reduce_sum(acc_prob)
      total_neg = tf.reduce_sum(neg_prob)
      total_gamma = tf.reduce_sum(gamma.log_prob(qgamma))
      total_beta = tf.reduce_sum(beta.log_prob(qbeta))

      log_loss = - ( total_gamma + total_beta + \
        (total_pos + total_acc) * (total_nonzero / num_pos) + \
        (total_neg - total_acc) * (total_zero / num_neg)
      )
      return log_loss

  def optimize(self, log_loss):
    """ Perform optimization (via Gradient Descent)"""
    with tf.name_scope('optimize'):
      opts = self.opts
      learning_rate = opts.learning_rate
      clipping_size = opts.clipping_size

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

  options = Options(
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
    block_size=FLAGS.block_size,
    min_sample_count=FLAGS.min_sample_count,
    min_feature_count=FLAGS.min_feature_count,
    statistics_interval=FLAGS.statistics_interval,
    summary_interval=FLAGS.summary_interval,
    checkpoint_interval=FLAGS.checkpoint_interval
  )

  # preprocessing (i.e. biom table, metadata, ...)
  (train_table, test_biom,
   train_metadata, test_metadata) = preprocess(
     options.formula,
     options.train_table, options.train_metadata,
     options.test_table, options.test_metadata,
     options.min_sample_count, options.min_feature_count
   )
  samp_ids = train_table.ids(axis='sample')

  obs_ids = train_table.ids(axis='observation')
  md_ids = np.array(train_metadata.columns)
  biom_train = train_table.matrix_data.tocoo().T
  biom_test = test_biom.matrix_data.tocoo().T

  # Model code
  with tf.Graph().as_default(), tf.Session() as session:
    model = PoissonRegression(options, session)
    model.initialize()
    gen = model.retrieve(train_table, train_metadata)
    y_feed, G_feed = next(gen)
    y_data = tf.sparse_placeholder(
      dtype=tf.int32, shape=(model.M, model.D), name='y_data_ph')
    G_data = tf.placeholder(
      tf.float32, shape=(model.M, model.p), name='G_data_ph')

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
       model.qbeta, model.qgamma],
      feed_dict={y_data: y_feed, G_data: G_feed}
    )

    epoch = model.num_nonzero // options.batch_size
    num_iter = int(options.epochs_to_train * epoch)
    saver = tf.train.Saver()

    writer = tf.summary.FileWriter(options.save_path, session.graph)

    start_time = time.time()
    k = 0
    for i in tqdm(range(1, num_iter)):
      now = time.time()
      # grab the next block
      if i % options.block_size == 0:
        y_feed, G_feed = next(gen)
        train_, loss, err, beta, gamma = session.run(
          [train_step, log_loss, mean_err,
            model.qbeta, model.qgamma],
          feed_dict={y_data: y_feed, G_data: G_feed}
        )
        k = k % options.block_size
      # check for summary
      elif now - last_summary_time > options.summary_interval:
        run_metadata = tf.RunMetadata()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        _, summary  = session.run(
          [train_step, merged],
          options=run_options,
          run_metadata=run_metadata,
          feed_dict={y_data: y_feed, G_data: G_feed}
        )
        writer.add_summary(summary, i)
        writer.add_run_metadata(run_metadata, 'step%d' % i)
        last_summary_time = now
      elif now - last_checkpoint_time > options.checkpoint_interval:
        saver.save(
          session, os.path.join(options.save_path, "model.ckpt"),
          global_step=i)
        last_checkpoint_time = now
      else:
        train_, loss, beta, gamma = session.run(
          [train_step, log_loss,
           model.qbeta, model.qgamma],
          feed_dict={y_data: y_feed, G_data: G_feed}
        )

    elapsed_time = time.time() - start_time
    print('Elapsed Time: %f seconds' % elapsed_time)

    # save all parameters to the save path
    train_, loss, beta, gamma, theta = session.run(
      [train_step, log_loss,
       model.qbeta, model.qgamma]
    )
    pd.DataFrame(
      beta, index=md_ids, columns=obs_ids,
    ).to_csv(os.path.join(options.save_path, 'beta.csv'))
    pd.DataFrame(
      gamma, index=['intercept'], columns=obs_ids,
    ).to_csv(os.path.join(options.save_path, 'gamma.csv'))
    pd.DataFrame(
      theta, index=samp_ids, columns=['theta'],
    ).to_csv(os.path.join(options.save_path, 'theta.csv'))

    # Run final round of cross validation
    y_test = np.array(model.y_test.todense())
    G_test = model.G_test
    # Cross validation
    pred_beta = model.qbeta.eval()
    pred_gamma = model.qgamma.eval()

    mse, mrc = cross_validation(
      G_test, pred_beta, pred_gamma, y_test)
    print("MSE: %f, MRC: %f" % (mse, mrc))


if __name__ == "__main__":
  tf.app.run()

# TODO: Future refactor
# Two parts to tensorflow (look at convolutional.py)
# __init__ = instance of class
#            (set the configuration parameters)
# __call__ = invoke the actual network
#            (pass in the session + runtime parameters)
