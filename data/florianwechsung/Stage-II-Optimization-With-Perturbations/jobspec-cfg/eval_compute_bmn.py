#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.curve import curves_to_vtk
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curveobjectives import CurveLength, CoshCurveCurvature
from simsopt.geo.curveobjectives import MinimumDistance
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import boozer_surface_residual, ToroidalFlux, Area
from simsopt.geo.qfmsurface import QfmSurface
from simsopt.geo.surfaceobjectives import QfmResidual, ToroidalFlux, Area, Volume
from objective import create_curves, add_correction_to_coils, get_outdir
from scipy.optimize import minimize
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, default=1e-3)
parser.add_argument("--sym", dest="sym", default=False, action="store_true")
parser.add_argument("--sampleidx", type=int, default=-1)
parser.add_argument("--outdiridx", type=int, default=0)
parser.add_argument("--well", dest="well", default=False, action="store_true")
parser.add_argument("--correctionlevel", type=int, default=1)
args = parser.parse_args()

print(args)
if args.sampleidx == -1:
    sampleidx = None
else:
    sampleidx = args.sampleidx

if not args.well:
    filename = 'input.LandremanPaul2021_QA'
else:
    filename = "input.20210728-01-010_QA_nfp2_A6_magwell_weight_1.00e+01_rel_step_3.00e-06_centered"

outdir = get_outdir(args.well, args.outdiridx)

nsamples = 0 if sampleidx is None else sampleidx + 1
fil = 0

base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
    fil=fil, ig=0, nsamples=nsamples, stoch_seed=1, sigma=args.sigma, order=16, sym=args.sym,
    zero_mean=False)

BiotSavart(coils_fil).x = np.loadtxt(outdir + "xmin.txt")

if sampleidx is None:
    coils_bmn = coils_fil
    nfp = 2
    stellsym = True
    mpol = 10
    ntor = 16
    nphi = ntor + 1
    ntheta = 2*mpol + 1
    phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
    thetas = np.linspace(0, 1., ntheta, endpoint=False)
else:
    coils_bmn = coils_fil_pert[sampleidx]
    nfp = 1
    stellsym = False

    mpol = 10
    ntor = 32
    nphi = int(1.5*(2*ntor + 1))
    ntheta = int(1.5*(2*mpol + 1))
    phis = np.linspace(0, 1., nphi, endpoint=False)
    thetas = np.linspace(0, 1., ntheta, endpoint=False)


clampy = not stellsym
clampy = False
s = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas,
    clamped_dims=[False, clampy, False]
)

s.least_squares_fit(
    SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=-phis, quadpoints_theta=-thetas).gamma()
)

sbmn = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, nphi=101, ntheta=101, range="full torus",
    clamped_dims=[False, clampy, False]
)

print("ls fit finished", flush=True)

def compute_non_quasisymmetry_L2(s, bs):
    x = s.gamma()
    B = bs.set_points(x.reshape((-1, 3))).B().reshape(x.shape)
    mod_B = np.linalg.norm(B, axis=2)
    n = np.linalg.norm(s.normal(), axis=2)
    mean_phi_mod_B = np.mean(mod_B*n, axis=0)/np.mean(n, axis=0)
    mod_B_QS = mean_phi_mod_B[None, :]
    mod_B_non_QS = mod_B - mod_B_QS
    non_qs = np.mean(mod_B_non_QS**2 * n)**0.5
    qs = np.mean(mod_B_QS**2 * n)**0.5
    return non_qs, qs



bs_tf = BiotSavart(coils_bmn)
# bs_tf.x = x
bs = BiotSavart(coils_bmn)
# bs.x = x
if (sampleidx is not None) and args.correctionlevel > 0:
    coils_bmn = add_correction_to_coils(coils_bmn, args.correctionlevel)
    bs = BiotSavart(coils_bmn)
    bs_tf = BiotSavart(coils_bmn)
    corrname = "corrections/" \
        + outdir.replace("/", "_")[:-1] \
        + f"_correction_sigma_{args.sigma}_sampleidx_{sampleidx}_correctionlevel_{args.correctionlevel}"
    y = np.loadtxt(corrname + ".txt")
    bs.x = y


current_sum = sum(abs(c.current.get_value()) for c in coils_bmn)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
iota = 0.416

tf = ToroidalFlux(s, bs_tf)
tf_target = tf.J()
# tf_ratios = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
tf_ratios = np.linspace(0.001, 1.0, 50, endpoint=True)[::-1]
tf_targets = [ratio*tf_target for ratio in tf_ratios]


boozer_surface = BoozerSurface(bs, s, tf, tf_target)
res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
    tol=1e-10, maxiter=200, constraint_weight=100., iota=iota, G=G0)
print(f"After LBFGS:   iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")

bmns = []
non_qss = []
b00s = []
for i, tf_target in enumerate(tf_targets):
    boozer_surface = BoozerSurface(bs, s, tf, tf_target)
    if i > 0:
        s.scale(tf_target/tf_targets[i-1])
        res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10, maxiter=20, constraint_weight=100., iota=res['iota'], G=res['G'])
        print(f"After LBFGS:   iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
    res = boozer_surface.minimize_boozer_penalty_constraints_ls(
        tol=1e-9, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')
    print(f"After Lev-Mar: iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
    # res = boozer_surface.solve_residual_equation_exactly_newton(
    #     tol=1e-11, maxiter=100, iota=res['iota'], G=res['G'])
    # print(f"After Exact:   iota={res['iota']:.3f}, tf={tf.J():.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}")
    sbmn.x = s.x

    bs.set_points(sbmn.gamma().reshape((-1, 3)))
    AbsB = bs.AbsB().reshape(sbmn.gamma().shape[:2])

    coeffs = np.fft.fft2(AbsB)
    coeffs *= 1./coeffs.size
    rcmn = np.real(coeffs).copy()
    rsmn = -np.imag(coeffs).copy()
    ms = np.zeros_like(rcmn)
    ns = np.zeros_like(rcmn)
    nphi, ntheta = AbsB.shape
    for j in range(ms.shape[1]):
        ms[:, j] = np.fft.fftfreq(ms.shape[0])
    for i in range(ns.shape[0]):
        ns[i, :] = np.fft.fftfreq(ns.shape[1])

    i, j = 3, 4
    assert abs(AbsB[i, j] -  np.real(np.sum(coeffs * np.exp(1j * 2 * np.pi * (i*ms + j*ns))))) < 1e-13
    assert abs(AbsB[i, j] -  np.sum(rcmn * np.cos(2*np.pi * (i*ms + j*ns))) - np.sum(rsmn * np.sin(2*np.pi * (i*ms + j*ns)))) < 1e-13
    rcmn[:, ntheta//2+1:] = 0
    rcmn[:, 1:ntheta//2+1] *= 2
    rcmn[nphi//2+1:, 0] = 0
    rcmn[1:nphi//2+1, 0] *= 2

    rsmn[:, ntheta//2+1:] = 0
    rsmn[:, 1:ntheta//2+1] *= 2
    rsmn[nphi//2+1:, 0] = 0
    rsmn[1:nphi//2+1, 0] *= 2

    ii = np.arange(nphi)
    jj = np.arange(ntheta)
    angle = 2*np.pi * (ii[None, None, :, None]*ms[:, :, None, None] + jj[None, None, None, :] *ns[:, :, None, None])
    assert np.linalg.norm(AbsB - np.sum(rcmn[:, :, None, None] * np.cos(angle) + rsmn[:, :, None, None] * np.sin(angle), axis=(0, 1))) < 1e-12

    bmn = max(np.max(np.abs(rcmn[1:, :])), np.max(np.abs(rsmn[1:, :])))

    bmns.append(bmn)
    b00s.append(rcmn[0, 0])
    non_qs, qs = compute_non_quasisymmetry_L2(sbmn, bs)
    non_qss.append(non_qs/sbmn.area()**0.5)



bmns = np.asarray(bmns)
bmns = bmns/b00s[-1]

non_qss = np.asarray(non_qss)
non_qss = non_qss/b00s[-1]
print("bmns", bmns)
print("non_qss", non_qss)

outname = "qsmeasures/" + outdir.replace("/", "_") + f"qsmeasures"
if sampleidx is not None:
    outname += f"_sampleidx_{sampleidx}_correctionlevel_{args.correctionlevel}_ls_10_32"
import os
os.makedirs("qsmeasures", exist_ok=True)
np.savetxt(outname + ".txt", np.asarray([tf_ratios, bmns, non_qss]).T, delimiter=",", header="flux,maxbmn,nonqsnorm", comments="")
# import IPython; IPython.embed()
import sys; sys.exit()
import matplotlib.pyplot as plt
plt.semilogy(tf_ratios, bmns)
plt.semilogy(tf_ratios, non_qss)
plt.hlines(0.00005, 0, 1.)
plt.ylim((1.5e-5, 4e-2))
plt.show()
