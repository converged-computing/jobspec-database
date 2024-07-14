import gp_lens
import numpy as np
import matplotlib.pyplot as plt
import emcee
import argparse
default_color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

# parse arguments
parser = argparse.ArgumentParser(description='Process some arguments!')
parser.add_argument('-o', '--output', dest='filename', action='store',
                    default='chain.dat',
                    help='output chain path. default: chain.dat, current dir')
parser.add_argument('-d', '--data', dest='data', action='store',
                    default=None, required=True,
                    help='PS or Peaks')
parser.add_argument('-binmin', dest='binmin', action='store',
                    default=None, type=float, required=True,
                    help='minimum bin center')
parser.add_argument('-binmax', dest='binmax', action='store',
                    default=None, type=float, required=True,
                    help='maximum bin center')
parser.add_argument('-cn', '--cov_noise', dest='cov_noise', action='store',
                    default='KN', required=True,
                    help='Covariance: Noisy or not noisy, K or KN.')
parser.add_argument('-z', '--redshift', nargs='*', dest='redshift',
                    action='store', default=['05', '10', '15', '20', '25'],
                    help='redshifts, i.e. 05 10 15 20 25')
parser.add_argument('--bin_center_row', type=int, dest='bin_center_row',
                    action='store', default=0,
                    help='0 or 1, 0 for PS, 0 for kappa, 1 for S/N for peaks')
parser.add_argument('--binscale', dest='binscale', action='store',
                    default='',
                    help='empty str (default), log or lin')
parser.add_argument('-s', '--smooth', dest='smoothing_scale', action='store',
                    default='1.00',
                    help='Smoothing Scale')
args = parser.parse_args()

print(f'filename: {args.filename}')
print(f'data source: {args.data}')
print(f'redshifts: {args.redshift}')
print(f'bin min, binmax: {args.binmin}, {args.binmax}')
print(f'mean noise: cov noise: {args.cov_noise}')
print(f'smoothing scale: {args.smoothing_scale}, binscale: {args.binscale}')
print(f'bin center row: {args.bin_center_row}')

# load in data and compute covariance
lens_obs = gp_lens.LensingPSorPeaks(args.binmin,args.binmax, 
                                    args.cov_noise, 
                                    redshifts=args.redshift, 
                                    observable_name=args.data, 
                                    bin_center_row=args.bin_center_row, 
                                    binscale=args.binscale, 
                                    smoothing=args.smoothing_scale )
x, y = lens_obs.get_realizations(model_index=1, verbose=True)
lens_obs.fid = np.mean(y,axis=0)
invcov_to_plot = lens_obs.compute_cov(0, verbose=True)

# remove fiducial model A and fit
index_list = np.arange(len(lens_obs.params.T))
index_list = np.delete(index_list, 0)
modified_y = [lens_obs.get_realizations(i) for i in index_list]
modified_X = (lens_obs.params.T[index_list]).T
lens_obs.fit(X=modified_X, real_list=modified_y)

# set up some constants for prior evaluation
m_nu_min = 0.06  # minimum from oscillation experiments
m_nu_max = 2*np.max(lens_obs.table['M_nu(eV)'])
om_m_min = np.min(lens_obs.table['Omega_m'])
om_m_max = 2*np.max(lens_obs.table['Omega_m'])
A_s_min = np.min(lens_obs.table['10^9*A_s'])
A_s_max = 2*np.max(lens_obs.table['10^9*A_s'])


# define emcee function calls, prior, likelihood,
def lnprior(theta):
    """Ensure the sampler stays near computed simulations."""
    m_nu, om_m, A_s = theta
    if (m_nu_min < m_nu < m_nu_max and
            om_m_min < om_m < om_m_max and
            A_s_min < A_s < A_s_max):
        return 0.0
    return -np.inf

def lnlike(theta):
    """Compute the log likelihood based on multivariate Gaussian."""
    return lens_obs.likelihood(theta)

def lnprob(theta):
    """Combine the likelihood and prior."""
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)



# set up emcee
ndim, nwalkers = 3, 120
p0 = [lens_obs.params.T[1] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

# begin sampling, use incremental chain
filename = args.filename
f = open(filename, "w")
f.close()

for result in sampler.sample(p0, iterations=5000, storechain=False):
    position = result[0]
    f = open(filename, "a")
    for k in range(position.shape[0]):
        out_str = "{0:4d} {1:s}\n".format(k, ' '.join(map(str, position[k])))
        f.write(out_str)
    f.close()

