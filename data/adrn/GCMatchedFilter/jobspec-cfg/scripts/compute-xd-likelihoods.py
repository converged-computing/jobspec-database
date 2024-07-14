""" Find stuff around NGC 5897 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys
import pickle

# Third-party
from astropy import log as logger
import numpy as np
import filelock
import h5py
from scipy.misc import logsumexp

def main(XCov_filename, xd_filename, chunk_index, n_per_chunk, overwrite=False):

    if not os.path.exists(XCov_filename):
        raise IOError("XCov file '{}' does not exist! Run photometry-to-xcov.py first."
                      .format(XCov_filename))
    lock_filename = "{}.lock".format(os.path.splitext(XCov_filename)[0])

    # load the classifier
    with open(xd_filename, 'rb') as f:
        clf = pickle.load(f)

    # define a slice object for this chunk to process
    slc = slice(chunk_index*n_per_chunk, (chunk_index+1)*n_per_chunk)

    ll_name = "xd_log_likelihood"

    # only one process should modify the file to add the dataset if it doesn't exist
    with h5py.File(XCov_filename, mode='r') as f:
        if ll_name not in f['search']:
            make_ll_dataset = True
        else:
            make_ll_dataset = False

    if make_ll_dataset:
        lock = filelock.FileLock(lock_filename)
        try:
            with lock.acquire(timeout=90):
                logger.debug("File lock acquired: creating dataset for log-likelihoods")
                with h5py.File(XCov_filename, mode='r+') as f:
                    if ll_name not in f['search']:
                        # file has no _log_likelihood dataset -- make one!
                        ll_shape = (f['search']['X'].shape[0],)
                        ll_dset = f['search'].create_dataset(ll_name, ll_shape, dtype='f')
                        ll_dset[:] = np.nan
                        ll = ll_dset[slc]
        except filelock.Timeout:
            logger.error("Timed out trying to acquire file lock to create dataset.")
            sys.exit(1)

    with h5py.File(XCov_filename, mode='r') as f:
        ll = f['search'][ll_name][slc]

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

        logger.debug("Computing likelihood for Chunk {} ({}:{})..."
                     .format(chunk_index,slc.start,slc.stop))

        ll = logsumexp(clf.logprob_a(X, Cov), axis=-1)
        logger.debug("...finished computing log-likelihoods")

    lock = filelock.FileLock(lock_filename)
    try:
        with lock.acquire(timeout=300):
            logger.debug("File lock acquired - writing to results")
            with h5py.File(XCov_filename, mode='r+') as f:
                f['search'][ll_name][slc] = ll

    except filelock.Timeout:
        logger.error("Timed out trying to acquire file lock to write results.")
        sys.exit(1)

def status(XCov_filename):
    ll_name = "xd_log_likelihood"
    with h5py.File(XCov_filename, mode='r') as f:
        if ll_name not in f['search']:
            logger.info("0 done")
            return

        ll = f['search'][ll_name]
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
    parser.add_argument("-x", "--xd-filename", dest="xd_filename", required=True,
                        type=str, help="Full path to extreme deconv. pickle file")
    parser.add_argument("-n", "--nperchunk", dest="n_per_chunk", default=1000,
                        type=int, help="Number of stars per chunk.")
    parser.add_argument("-i", "--chunk-index", dest="index", default=None,
                        type=int, help="Index of the chunk to process.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    if args.status:
        status(args.XCov_filename)
        sys.exit(0)

    if args.index is None:
        raise ValueError("You must supply a chunk index to process! (-i or --chunk-index)")

    main(args.XCov_filename, args.xd_filename, chunk_index=args.index,
         n_per_chunk=args.n_per_chunk, overwrite=args.overwrite)
