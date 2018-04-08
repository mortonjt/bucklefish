import tensorflow as tf
import numpy as np
import pandas as pd
from biom import load_table, Table

from gneiss.balances import _balance_basis
from gneiss.composition import ilr_transform
from gneiss.util import match, match_tips, rename_internal_nodes

from tensorflow.contrib.distributions import Poisson, Normal
from patsy import dmatrix
from skbio import TreeNode


flags = tf.app.flags
flags.DEFINE_string("save_path", None, "Directory to write the model and "
                    "training summaries.")
flags.DEFINE_string("biom_data", None, "Input biom table. "
                    "i.e. input.biom")
flags.DEFINE_string("sample_metadata", None, "Input sample metadata. "
                    "i.e. metadata.txt")
flags.DEFINE_string("tree", None, "Input tree. "
                    "i.e. tree.nwk")
flags.DEFINE_string("formula", None, "Statistical formula for "
                    "specifying covariates.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")

flags.DEFINE_float("beta_mean", 0,
                   'Mean of prior distribution for covariates')
flags.DEFINE_float("beta_scale", 1.0,
                   'Scale of prior distribution for covariates')
flags.DEFINE_float("theta_mean", 0,
                   'Mean of prior distribution for sample bias')
flags.DEFINE_float("theta_scale", 1.0,
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

FLAGS = flags.FLAGS

class Options(object):
  """Options used by our Poisson Niche model."""

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)


    if isinstance(self.table_file, str):
      self.table = load_table(self.table_file)
    elif isinstance(self.table_file, Table):
      self.table = self.table_file

    if isinstance(self.tree_file, str):
      self.tree = TreeNode.read(self.tree_file)
    elif isinstance(self.tree_file, Table):
      self.tree = self.tree_file

    if isinstance(self.metadata, str):
      self.metadata = pd.read_table(metadata_file, index_col=0)
    elif isinstance(self.metadata_file, Table):
      self.metadata = self.metadata_file

    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)


def match_tips(table, tree):
    """ Returns the contingency table and tree with matched tips.
    Sorts the columns of the contingency table to match the tips in
    the tree.  The ordering of the tips is in post-traversal order.
    If the tree is multi-furcating, then the tree is reduced to a
    bifurcating tree by randomly inserting internal nodes.
    The intersection of samples in the contingency table and the
    tree will returned.

    Parameters
    ----------
    table : biom.Table
        Contingency table where samples correspond to rows and
        features correspond to columns.
    tree : skbio.TreeNode
        Tree object where the leafs correspond to the features.

    Returns
    -------
    biom.Table :
        Subset of the original contingency table with the common features.
    skbio.TreeNode :
        Sub-tree with the common features.
    """
    tips = [x.name for x in tree.tips()]
    common_tips = set(tips) & set(table.ids(axis='observation'))


    _tree = tree.shear(names=list(common_tips))

    def filter_uncommon(val, id_, md):
        return id_ in common_tips
    _table = table.filter(filter_uncommon, axis='observation', inplace=False)

    _tree.bifurcate()
    _tree.prune()
    sort_f = lambda x: [n.name for n in _tree.tips()]
    _table = _table.sort(sort_f=sort_f, axis='observation')
    return _table, _tree


class PoissonNiche():
  def __init__(self, options, session):
    self._options = options
    self._session = session

  def build_graph(self):
    pass

  def preprocess(self):
    opts = self._options
    table, tree, metadata = self.table, self.tree, self.metadata
    metadata.columns = [x.replace('-', '_') for x in metadata.columns]
    metadata = metadata.loc[table.ids(axis='sample')]

    sample_filter = lambda val, id_, md: (
      (id_ in metadata.index) and np.sum(val) > opts.min_sample_count)
c    read_filter = lambda val, id_, md: np.sum(val) > opts.min_feature_count

    table = table.filter(sample_filter, axis='sample')
    table = table.filter(read_filter, axis='observation')
    table, tree = match_tips(table, tree)
    sort_f = lambda x: list(metadata.index)
    table = table.sort(sort_f=sort_f, axis='sample')
    tree = rename_internal_nodes(tree)

    self.table = table
    self.metadata = dmatrix(formula, metadata, return_type='dataframe')
    self.tree = tree

  def forward(self):
    pass

  def cross_validation(self, hold_out_data):
    """ Computes two cross validation metrics

    1) Rank difference
    2) MSE

    """
    pass

  def optimize(self, loss):
    pass

  def train(self, train_data):
    pass


def main(_):
    pass


if __name__ == "__main__":
  tf.app.run()
