from biom import Table
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from scipy.sparse import coo_matrix
from skbio.stats.composition import closure


def random_poisson_model(num_samples, num_features,
                         tree=None,
                         reps=1,
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
    theta = state.normal(theta_mean, theta_scale, size=(num_samples, 1))
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
                mu[i, :] + theta[i] + eps[i, :] + alpha
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


def random_multinomial_model(num_samples, num_features,
                             tree=None,
                             reps=1,
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
    theta = state.normal(theta_mean, theta_scale, size=(num_samples, 1))
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
    depth = np.exp(alpha).astype(np.int32)
    table = np.vstack(
        state.multinomial(depth,
            closure(np.exp(
                mu[i, :] + eps[i, :]
            ))
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


def sparse_balance_basis(tree):
    """ Calculates sparse representation of an ilr basis from a tree.

    This computes an orthonormal basis specified from a bifurcating tree.

    Parameters
    ----------
    tree : skbio.TreeNode
        Input bifurcating tree.  Must be strictly bifurcating
        (i.e. every internal node needs to have exactly 2 children).
        This is used to specify the ilr basis.

    Returns
    -------
    scipy.sparse.coo_matrix
       The ilr basis required to perform the ilr_inv transform.
       This is also known as the sequential binary partition.
       Note that this matrix is represented in clr coordinates.
    nodes : list, str
        List of tree nodes indicating the ordering in the basis.

    Raises
    ------
    ValueError
        The tree doesn't contain two branches.

    """
    NUMERATOR=1
    DENOMINATOR=0
    # This is inspired by @wasade in
    # https://github.com/biocore/gneiss/pull/8
    t = tree.copy()
    D = len(list(tree.tips()))
    # calculate number of tips under each node
    for n in t.postorder(include_self=True):
        if n.is_tip():
            n._tip_count = 1
        else:
           try:
               left, right = n.children[NUMERATOR], n.children[DENOMINATOR],
           except:
               raise ValueError("Not a strictly bifurcating tree.")
           n._tip_count = left._tip_count + right._tip_count

    # calculate k, r, s, t coordinate for each node
    left, right = t.children[NUMERATOR], t.children[DENOMINATOR],
    t._k, t._r, t._s, t._t = 0, left._tip_count, right._tip_count, 0
    for n in t.preorder(include_self=False):
        if n.is_tip():
            n._k, n._r, n._s, n._t = 0, 0, 0, 0

        elif n == n.parent.children[NUMERATOR]:
            n._k = n.parent._k
            n._r = n.children[NUMERATOR]._tip_count
            n._s = n.children[DENOMINATOR]._tip_count
            n._t = n.parent._s + n.parent._t
        elif n == n.parent.children[DENOMINATOR]:
            n._k = n.parent._r + n.parent._k
            n._r = n.children[NUMERATOR]._tip_count
            n._s = n.children[DENOMINATOR]._tip_count
            n._t = n.parent._t
        else:
            raise ValueError("Tree topology is not correct.")

    # navigate through tree to build the basis in a sparse matrix form
    value = []
    row, col = [], []
    nodes = []
    i = 0

    for n in t.levelorder(include_self=True):

        if n.is_tip():
            continue

        for j in range(n._k, n._k + n._r):
            row.append(i)
            col.append(D-1-j)
            A = np.sqrt(n._s / (n._r * (n._s + n._r)))

            value.append(A)

        for j in range(n._k + n._r, n._k + n._r + n._s):
            row.append(i)
            col.append(D-1-j)
            B = -np.sqrt(n._r / (n._s * (n._s + n._r)))

            value.append(B)
        i += 1
        nodes.append(n.name)

    basis = coo_matrix((value, (row, col)), shape=(D-1, D), dtype=np.float32)

    return basis, nodes


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
