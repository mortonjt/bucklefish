import click
import pandas as pd
import numpy as np
import os
from util import random_multinomial_model, random_poisson_model
from gneiss.cluster import rank_linkage
from biom.util import biom_open
from biom.table import Table

from biom import load_table
from scipy.sparse import coo_matrix


@click.group()
def generate():
    pass


@generate.command()
@click.option('--output_dir', help='output directory')
@click.option('--num_samples', type=int)
@click.option('--num_features', type=int)
@click.option('--reps', default=1, help='replicates')
@click.option('--low', default=2, help='Lower bound of gradient')
@click.option('--high',default=10,  help='Upper bound of gradient')
@click.option('--alpha_mean',  default=0., help='Mean prior for global bias')
@click.option('--alpha_scale', default=1., help='Scale prior for global bias')
@click.option('--theta_mean',  default=0., help='Mean prior for sample bias')
@click.option('--theta_scale', default=1., help='Scale prior for sample bias')
@click.option('--gamma_mean',  default=0., help='Mean prior for feature bias')
@click.option('--gamma_scale', default=1., help='Scale prior for feature bias')
@click.option('--kappa_mean',  default=0., help='Mean prior for dispersion')
@click.option('--kappa_scale', default=1., help='Scale prior for dispersion')
@click.option('--beta_mean',   default=0., help='Mean prior for covariate')
@click.option('--beta_scale',  default=1., help='Scale prior for covariate')
@click.option('--seed', default=0, help='Random number generator')
def simulate_poisson(
        output_dir,
        num_samples, num_features,
        reps,
        low, high,
        alpha_mean,
        alpha_scale,
        theta_mean,
        theta_scale,
        gamma_mean,
        gamma_scale,
        kappa_mean,
        kappa_scale,
        beta_mean,
        beta_scale,
        seed):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    table_file = os.path.join(output_dir, 'table.biom')
    metadata_file = os.path.join(output_dir, 'metadata.txt')
    tree_file = os.path.join(output_dir, 'tree.nwk')
    beta_file = os.path.join(output_dir, 'beta.csv')
    theta_file = os.path.join(output_dir, 'theta.csv')
    gamma_file = os.path.join(output_dir, 'gamma.csv')
    kappa_file = os.path.join(output_dir, 'kappa.csv')
    eps_file= os.path.join(output_dir, 'eps.csv')

    feat_ids = ['F%d' % i for i in range(num_features)]
    ranks = pd.Series(np.arange(num_features), index=feat_ids)
    tree = rank_linkage(ranks, method='average')

    gen = random_poisson_model(num_samples, num_features,
                               tree, reps,
                               low, high,
                               alpha_mean,
                               alpha_scale,
                               theta_mean,
                               theta_scale,
                               gamma_mean,
                               gamma_scale,
                               kappa_mean,
                               kappa_scale,
                               beta_mean,
                               beta_scale,
                               seed)
    (table, metadata, basis, sim_alpha,
     sim_beta, sim_theta, sim_gamma, sim_kappa, sim_eps) = gen

    tree.write(tree_file)
    with biom_open(table_file, 'w') as f:
        table.to_hdf5(f, "simulation")
    metadata.to_csv(metadata_file, sep='\t')
    sim_beta.to_csv(beta_file)
    sim_gamma.to_csv(gamma_file)
    sim_theta.to_csv(theta_file)
    sim_kappa.to_csv(kappa_file)


@generate.command()
@click.option('--output_dir', help='output directory')
@click.option('--num_samples', type=int)
@click.option('--num_features', type=int)
@click.option('--reps', default=1, help='replicates')
@click.option('--low', default=2, help='Lower bound of gradient')
@click.option('--high',default=10,  help='Upper bound of gradient')
@click.option('--alpha_mean',  default=0, help='Mean prior for global bias')
@click.option('--alpha_scale', default=1, help='Scale prior for global bias')
@click.option('--theta_mean',  default=0, help='Mean prior for sample bias')
@click.option('--theta_scale', default=1, help='Scale prior for sample bias')
@click.option('--gamma_mean',  default=0, help='Mean prior for feature bias')
@click.option('--gamma_scale', default=1, help='Scale prior for feature bias')
@click.option('--kappa_mean',  default=0, help='Mean prior for dispersion')
@click.option('--kappa_scale', default=1, help='Scale prior for dispersion')
@click.option('--beta_mean',   default=0, help='Mean prior for covariate')
@click.option('--beta_scale',  default=1, help='Scale prior for covariate')
@click.option('--seed', default=0, help='Random number generator')
def simulate_multinomial(
        output_dir,
        num_samples, num_features,
        reps,
        low, high,
        alpha_mean,
        alpha_scale,
        theta_mean,
        theta_scale,
        gamma_mean,
        gamma_scale,
        kappa_mean,
        kappa_scale,
        beta_mean,
        beta_scale,
        seed):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    table_file = os.path.join(output_dir, 'table.biom')
    metadata_file = os.path.join(output_dir, 'metadata.txt')
    tree_file = os.path.join(output_dir, 'tree.nwk')
    beta_file = os.path.join(output_dir, 'beta.csv')
    theta_file = os.path.join(output_dir, 'theta.csv')
    gamma_file = os.path.join(output_dir, 'gamma.csv')
    kappa_file = os.path.join(output_dir, 'kappa.csv')
    eps_file= os.path.join(output_dir, 'eps.csv')

    feat_ids = ['F%d' % i for i in range(num_features)]
    ranks = pd.Series(np.arange(num_features), index=feat_ids)
    tree = rank_linkage(ranks, method='average')

    gen = random_multinomial_model(num_samples, num_features,
                                   tree,
                                   reps,
                                   low, high,
                                   alpha_mean,
                                   alpha_scale,
                                   theta_mean,
                                   theta_scale,
                                   gamma_mean,
                                   gamma_scale,
                                   kappa_mean,
                                   kappa_scale,
                                   beta_mean,
                                   beta_scale,
                                   seed)
    (table, metadata, basis, sim_alpha,
     sim_beta, sim_theta, sim_gamma, sim_kappa, sim_eps) = gen

    tree.write(tree_file)

    with biom_open(table_file, 'w') as f:
        table.to_hdf5(f, "simulation")
    metadata.to_csv(metadata_file, sep='\t')
    sim_beta.to_csv(beta_file)
    sim_gamma.to_csv(gamma_file)
    sim_theta.to_csv(theta_file)
    sim_kappa.to_csv(kappa_file)


@generate.command()
@click.option('--input_biom', help='Input biom table')
@click.option('--input_metadata', help='Input metadata')
@click.option('--split_ratio', default=0.75,
              help='Number of training vs test examples')
@click.option('--output_dir', help='output directory')
def split_dataset(input_biom, input_metadata, split_ratio, output_dir):
    table = load_table(input_biom)
    metadata = pd.read_table(input_metadata, index_col=0)
    metadata.columns = [x.replace('-', '_') for x in metadata.columns]
    sample_ids = metadata.index
    D, N = table.shape
    samples = pd.Series(np.arange(N), index=sample_ids)
    train_size = int(N * split_ratio)
    test_size = N - train_size

    test_samples = set(np.random.choice(sample_ids, size=test_size))

    test_idx =  np.array([(x in test_samples) for x in metadata.index])
    train_idx = ~test_idx
    f = lambda id_, md: id_ in test_samples
    gen = table.partition(f)

    _, train_table = next(gen)
    _, test_table = next(gen)

    train_metadata = metadata.iloc[train_idx]
    test_metadata = metadata.iloc[test_idx]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    test_metadata_path = os.path.join(output_dir, 'test_' + os.path.basename(input_metadata))
    train_metadata_path = os.path.join(output_dir, 'train_' + os.path.basename(input_metadata))

    test_biom_path = os.path.join(output_dir, 'test_' + os.path.basename(input_biom))
    train_biom_path = os.path.join(output_dir, 'train_' + os.path.basename(input_biom))

    print(train_metadata_path)
    train_metadata.to_csv(train_metadata_path, sep='\t')
    test_metadata.to_csv(test_metadata_path, sep='\t')

    with biom_open(train_biom_path, 'w') as f:
        train_table.to_hdf5(f, "train")

    with biom_open(test_biom_path, 'w') as f:
        test_table.to_hdf5(f, "test")


if __name__ == "__main__":
    generate()
