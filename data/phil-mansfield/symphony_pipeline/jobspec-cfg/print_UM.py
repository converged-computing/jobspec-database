import symlib
import sys
import numpy as np
import os.path as path
import os

um_dir = "/sdf/home/p/phil1/code/src/bitbucket.org/pbehroozi/universemachine"

print_sm_catalog = path.join(um_dir, "print_sm_catalog")

def sf_fmt_to_bin_fmt(sf_fmt):
    return sf_fmt[:-4] + ".bin", sf_fmt[:-4] + ".bin.txt"

def config_file(param):
    return"""INBASE = /dev/null
OUTBASE = /dev/null
NUM_BLOCKS = 144

BOX_SIZE = 125
Om = %f
Ol = %f
h0 = %f
fb = %f

NUM_NODES = 48
BLOCKS_PER_NODE = 24

CALC_ICL = 1
""" % (param["Om0"], 1-param["Om0"], param["H0"]/100, param["Ob0"]/param["Om0"])

def parse_sim_dir(sim_dir):
    if sim_dir[-1] == "/": sim_dir = sim_dir[:-1]
    suite_dir, halo = path.split(sim_dir)
    base_dir, suite = path.split(suite_dir)
    return base_dir, suite, halo

def get_scale_str(fname):
    return fname[-12:-4]

def get_base_str(path):
    return path.basename(path)[-12:]

def main():
    cfg_name, idx = sys.argv[1], sys.argv[2]
    idx = int(idx)
    sim_dirs, sf_fmts = np.loadtxt(cfg_name, dtype=str, usecols=(7,9)).T
    
    base_dir, suite, halo = parse_sim_dir(sim_dirs[0])
    
    param = symlib.simulation_parameters(suite)
    cfg = config_file(param)
    
    n_host = symlib.n_hosts(suite)
    for i_host in range(n_host):
        if i_host != idx and idx != -1: continue
        print("%d/%d" % (i_host+1, n_host))
        sim_dir, sf_fmt = sim_dirs[i_host], sf_fmts[i_host]
        sf_bin_fmt, sf_bin_txt_fmt = sf_fmt_to_bin_fmt(sf_fmt)

        a = symlib.scale_factors(sim_dir)

        # Horrible hack because UM rounds its scale factors incorrectly, ahhh
        names = [path.join(path.dirname(sf_fmt), name) for name in
                 sorted(os.listdir(path.dirname(sf_fmt))) if
                 "sfr_catalog" in name and name[-4:] == ".bin"]

        um_dir = path.join(sim_dir, "um")
        if not path.exists(um_dir):
            os.system("mkdir %s" % um_dir)
        um_cfg_file = path.join(um_dir, "um.cfg")

        with open(um_cfg_file, "w+") as fp:
            print(cfg, file=fp)

        if not path.exists(um_dir):
            os.mkdir(um_dir)
        out_names = [path.join(um_dir, "um_%d.txt" % i) for i in
                     range(len(a))]
        for i in range(len(a)):
            os.system("%s %s %s > %s" % (print_sm_catalog, names[i],
                                         um_cfg_file, out_names[i]))

                

if __name__ == "__main__": main()
