import hoomd
import hoomd.md
from hoomd.deprecated import *

hoomd.context.initialize("");
import math
# parameters
phi_P = 0.25
n_poly = 24
T = 1.2
polymer1 = dict(bond_len=1.2, type=['A']*3000 + ['B']*4000 + ['A']*3000,bond="linear", count=n_poly)
# perform some simple math to find the length of the box
N = len(polymer1['type']) * polymer1['count']
# generate the polymer system
init.create_random_polymers(box=hoomd.data.boxdim(volume=10*math.pi * N / (6.0 * phi_P)), polymers=[polymer1],separation=dict(A=0.35, B=0.35),seed=12)
# force field setup
harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('polymer', k=330.0, r0=0.84)
nl = hoomd.md.nlist.cell();
lj = hoomd.md.pair.lj(r_cut=3.0,nlist=nl)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=0.0,r_cut=2**(1.0/6.0))
lj.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0, alpha=0.0,r_cut=2**(1.0/6.0))
lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, alpha=1.0)
all = hoomd.group.all()
hoomd.dump.gsd("/home/mferguson/scratch/hum_gen_24x10000.gsd", period=1000, group=all, overwrite=True);
#hoomd.analyze.imd(port=54321, period=10000)
# integrate NVT for a bunch of time steps
hoomd.md.integrate.mode_standard(dt=0.005)
hoomd.md.integrate.nvt(group=all, kT=1.2, tau=0.5)
#run(2000)
# uncomment the next run() command if you have a few hours to spare
# running this on a GPU the resulting dump files should show the
# polymers self-assembling into the hex phase
hoomd.run(1e5)
