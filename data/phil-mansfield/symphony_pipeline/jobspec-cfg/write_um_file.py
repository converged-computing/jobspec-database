import numpy as np
import symlib
import sys
import os.path as path
import convert_core_catalogue
    
def write_um(sim_dir, um):
    fname = path.join(sim_dir, "um", "um.dat")
    n_snap, n_halo = um.shape
    n = n_snap*n_halo

    with open(fname, "w+") as fp:
        um["m_star"].tofile(fp)
        um["m_in_situ"].tofile(fp)
        um["x"].tofile(fp)
        um["v"].tofile(fp)
        um["sfr"].tofile(fp)
        um["rank"].tofile(fp)
        um["mvir"].tofile(fp)
        um["vmax"].tofile(fp)
        um["is_orphan"].tofile(fp)
        um["ok"].tofile(fp)

def get_first_snap(h):
    out = np.ones(len(h), dtype=np.int64)*-1
    snap = np.arange(h.shape[1], dtype=int)
    for i_sub in range(len(h)):
        ok = h["ok"][i_sub,:]
        out[i_sub] = np.min(snap[ok])
    return out

def get_target_ids(h, snap, first_snap, next_id, is_err):
    out = next_id
    use_symlib_id = ((snap == first_snap) | is_err) & h["ok"][:,snap]
    out[use_symlib_id] = h["id"][use_symlib_id,snap]
    return out

def get_um_data(sim_dir):
    if sim_dir[-1] == "/": sim_dir = sim_dir[:-1]

    base_dir,suite_name,halo_name=convert_core_catalogue.parse_sim_dir(sim_dir)
    param = symlib.parameter_table[suite_name]
    h, hist = symlib.read_subhalos(sim_dir)
    h_cmov, _ = symlib.read_subhalos(sim_dir, comoving=True)
    a = symlib.scale_factors(sim_dir)

    um = np.zeros(h.shape, dtype=symlib.UM_DTYPE)

    first_snap = get_first_snap(h)
    next_id = np.ones(len(h), dtype=np.int64)*-1
    prev_id = np.ones(len(h), dtype=np.int64)*-1
    is_err = np.zeros(len(h), dtype=np.bool)

    print(halo_name)
    for snap in range(len(a)):
        print("   ", snap)
        um_txt_name = path.join(sim_dir, "um", "um_%d.txt" % snap)
        id, desc_id = np.loadtxt(um_txt_name, usecols=(0,1), dtype=np.int64).T
        x, y, z, vx, vy, vz, mvir, vmax, rank, mstar, micl, sfr = np.loadtxt(
            um_txt_name, usecols=(5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 21, 22)
        ).T
        x = np.array([x, y, z]).T
        v = np.array([vx, vy, vz]).T
        order = np.argsort(id)

        id, desc_id = id[order], desc_id[order]
        x, v, rank = x[order], v[order], rank[order]
        mvir, vmax = mvir[order], vmax[order]
        mstar, micl, sfr = mstar[order], micl[order], sfr[order]
        already_used = np.zeros(len(id), dtype=bool)
        
        target_ids = get_target_ids(h, snap, first_snap, next_id, is_err)
        print(target_ids)
        idx = np.ones(len(target_ids), dtype=np.int64)*-1
        for i_sub in range(len(h)):
            if target_ids[i_sub] == -1: continue
            idx[i_sub] = np.searchsorted(id, target_ids[i_sub])
            if id[idx[i_sub]] != target_ids[i_sub]:
                idx[i_sub] = -1
                is_err[i_sub] = True
                #print("        Error at sub %d, snap %d" % (i_sub, snap))
            elif already_used[idx[i_sub]]:
                idx[i_sub] = -1
                is_err[i_sub] = False
                next_id[i_sub] = -1
            else:
                is_err[i_sub] = False
                already_used[idx[i_sub]] = True
                next_id[i_sub] = -1

        ok = idx > 0
        print(np.sum(ok & h["ok"][:,snap]), np.sum((~ok) & h["ok"][:,snap]))
        um["m_star"][ok,snap] = mstar[idx[ok]]
        um["m_icl"][ok,snap] = micl[idx[ok]]
        um["sfr"][ok,snap] = sfr[idx[ok]]
        um["x"][ok,snap] = x[idx[ok]]
        um["v"][ok,snap] = v[idx[ok]]
        um["rank"][ok,snap] = rank[idx[ok]]
        um["mvir"][ok,snap] = mvir[idx[ok]]
        um["vmax"][ok,snap] = vmax[idx[ok]]
        um["is_orphan"][:,snap] = False
        um["ok"][:,snap] = ok
        
        next_id[ok] = desc_id[idx[ok]]
        next_id[~ok] = -1

        um["x"][:,snap] -= h_cmov["x"][0, snap]
        um["v"][:,snap] -= h_cmov["v"][0, snap]
        um["x"][:,snap] *= a[snap]

        if snap == 0: exit(1)
    return um

def main():
    config_name, idx_str = sys.argv[1], sys.argv[2]
    target_idx = int(idx_str)

    sim_dirs = convert_core_catalogue.get_sim_dirs(config_name)
    n_host = len(sim_dirs)
    
    for i_host in range(n_host):
        if target_idx == -1 or i_host == target_idx:
            um = get_um_data(sim_dirs[i_host])
            write_um(sim_dirs[i_host], um)

            um = symlib.read_um(sim_dirs[i_host])

if __name__ == "__main__": main()
