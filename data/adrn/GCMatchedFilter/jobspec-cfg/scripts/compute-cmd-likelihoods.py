""" Find stuff around NGC 5897 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import numpy as np
import filelock
import h5py

# Project
from globber.core import likelihood_worker

# HACK:
# from globber.ngc5897 import mixing_matrix as W
W = None

def initialize_dataset(dset_name, group_path, XCov_filename, lock_filename):
    # only one process should modify the file to add the dataset if it doesn't exist
    with h5py.File(XCov_filename, mode='r') as f:
        make_group = False
        make_dataset = True

        try:
            group = f[group_path]
            logger.debug("Group already exists")
        except KeyError:
            make_group = True
            logger.debug("Group doesn't exist...")

        if not make_group and dset_name in group:
            make_dataset = False

    if make_group or make_dataset:
        lock = filelock.FileLock(lock_filename)
        try:
            with lock.acquire(timeout=90):
                logger.debug("File lock acquired: creating dataset for log-likelihoods")
                with h5py.File(XCov_filename, mode='r+') as f:
                    if make_group and group_path not in f:
                        group = f.create_group(group_path)
                    else:
                        group = f[group_path]

                    if dset_name not in group: # double checking!
                        ll_shape = (f['search']['X'].shape[0],)
                        ll_dset = group.create_dataset(dset_name, ll_shape, dtype='f')
                        ll_dset[:] = np.nan

        except filelock.Timeout:
            logger.error("Timed out trying to acquire file lock to create dataset.")
            sys.exit(1)

def main(XCov_filename, chunk_index, n_per_chunk, ll_name, overwrite=False,
         n_compare=None, smooth=None, dm=None):

    if not os.path.exists(XCov_filename):
        raise IOError("XCov file '{}' does not exist! Run photometry-to-xcov.py first."
                      .format(XCov_filename))
    lock_filename = "{}.lock".format(os.path.splitext(XCov_filename)[0])

    # define a slice object for this chunk to process
    slc = slice(chunk_index*n_per_chunk, (chunk_index+1)*n_per_chunk)

    # name of the log-likelihood dataset
    if ll_name == 'isochrone':
        if dm is None:
            raise ValueError("If isochrone, must specify distance modulus (--dm=...)")

        dset_name = "{:.2f}".format(dm)
        group_path = 'log_likelihood/isochrone'
    else:
        dset_name = ll_name
        group_path = 'log_likelihood'

    dset_path = os.path.join(group_path, dset_name)
    initialize_dataset(dset_name, group_path, XCov_filename, lock_filename)

    with h5py.File(XCov_filename, mode='r') as f:
        ll = f[dset_path][slc]

        if np.isfinite(ll).all() and not overwrite:
            logger.debug("All log-likelihoods already computed for Chunk {} ({}:{})"
                         .format(chunk_index,slc.start,slc.stop))
            return

        if not np.isfinite(ll).all() and not overwrite:
            some_unfinished = True
            unfinished_idx = np.isnan(ll)
            logger.debug("{} log-likelihoods already computed -- will fill unfinished values."
                         .format(len(ll) - unfinished_idx.sum()))
        else:
            some_unfinished = False

        # slice out this chunk
        X = f['search']['X'][slc]
        Cov = f['search']['Cov'][slc]
        if some_unfinished:
            X = X[unfinished_idx]
            Cov = Cov[unfinished_idx]

        X_compare = f[ll_name]['X']
        if 'Cov' not in f[ll_name]:
            Cov_compare = None
        else:
            Cov_compare = f[ll_name]['Cov']

        if n_compare is not None and n_compare < X_compare.shape[0]:
            # Note: can't use randint here because non-unique lists cause an OSError,
            #   using np.random.choice on an int array uses a bit of memory
            idx = []
            iterations = 0
            while len(idx) < n_compare and iterations < 1E8:
                s = np.random.randint(X_compare.shape[0])
                if s not in idx:
                    idx.append(s)
                iterations += 1
            idx = sorted(idx)
            X_compare = X_compare[idx]
            if Cov_compare is not None:
                Cov_compare = Cov_compare[idx]

        else:
            X_compare = X_compare[:]
            if Cov_compare is not None:
                Cov_compare = Cov_compare[:]

        if ll_name == 'isochrone':
            X_compare[:,0] += dm # add distance modulus

        logger.debug("{} total stars, {} comparison stars, {} chunk stars"
                     .format(f['search']['X'].shape[0], X_compare.shape[0], X.shape[0]))

        logger.debug("Computing likelihood for Chunk {} ({}:{})..."
                     .format(chunk_index,slc.start,slc.stop))
        ll = likelihood_worker(X, Cov, X_compare, Cov_compare, smooth=smooth, W=W)
        logger.debug("...finished computing log-likelihoods (nan/inf: {})"
                     .format(np.logical_not(np.isfinite(ll)).sum()))

    lock = filelock.FileLock(lock_filename)
    try:
        with lock.acquire(timeout=300):
            logger.debug("File lock acquired - writing to results")
            with h5py.File(XCov_filename, mode='r+') as f:
                f[dset_path][slc] = ll

    except filelock.Timeout:
        logger.error("Timed out trying to acquire file lock to write results.")
        sys.exit(1)

def status(XCov_filename, ll_name, dm=None):
    if ll_name == 'isochrone':
        if dm is None:
            raise ValueError("If isochrone, must specify distance modulus (--dm=...)")

        dset_name = "{:.2f}".format(dm)
        group_path = 'log_likelihood/isochrone'
    else:
        dset_name = ll_name
        group_path = 'log_likelihood'
    dset_path = os.path.join(group_path, dset_name)

    with h5py.File(XCov_filename, mode='r') as f:
        if dset_path not in f:
            logger.info("0 done for '{}'".format(ll_name))
            return

        ll = f[dset_path]
        ndone = np.isfinite(ll).sum()
        nnot = np.isnan(ll).sum()
        logger.info("{} done, {} not done".format(ndone, nnot))

        # check what blocks are unfinished
        if nnot != 0:
            idx, = np.where(np.isnan(ll))
            diff = idx[1:]-idx[:-1]
            derp, = np.where(diff > 1)
            if 0 not in derp:
                derp = np.concatenate(([0], derp, [len(idx)-1]))

            logger.debug("Unfinished blocks:")
            blocks = []
            for d1,d2 in zip(derp[:-1],derp[1:]):
                if d1 == 0:
                    blocks.append("{}-{}".format(idx[d1], idx[d2]))
                else:
                    blocks.append("{}-{}".format(idx[d1+1], idx[d2]))
            logger.debug(", ".join(blocks))

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="DESTROY OLD VALUES.")

    parser.add_argument("--status", dest="status", action="store_true", default=False,
                        help="Check status of results file.")

    parser.add_argument("-f", "--xcov-filename", dest="XCov_filename", required=True,
                        type=str, help="Full path to XCov file")
    parser.add_argument("--name", dest="name", required=True,
                        type=str, help="name for log-likelihood calc. (cluster, control, isochrone)")
    parser.add_argument("-n", "--nperchunk", dest="n_per_chunk", default=1000,
                        type=int, help="Number of stars per chunk.")
    parser.add_argument("-i", "--chunk-index", dest="index", default=None,
                        type=int, help="Index of the chunk to process.")
    parser.add_argument("--ncompare", dest="n_compare", default=None,
                        type=int, help="Number of points (stars for cluster or noncluster) "
                                       "to compare to.")
    parser.add_argument("--smooth", dest="smooth", default=None,
                        type=float, help="Smooth comparison by this amount (units: mag)")
    parser.add_argument("--dm", dest="distance_modulus", default=None,
                        type=float, help="Distance modulus for isochrone.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    if args.status:
        status(args.XCov_filename, args.name, dm=args.distance_modulus)
        sys.exit(0)

    if args.index is None:
        raise ValueError("You must supply a chunk index to process! (-i or --chunk-index)")

    main(args.XCov_filename, chunk_index=args.index, n_per_chunk=args.n_per_chunk,
         overwrite=args.overwrite, ll_name=args.name, n_compare=args.n_compare,
         smooth=args.smooth, dm=args.distance_modulus)
