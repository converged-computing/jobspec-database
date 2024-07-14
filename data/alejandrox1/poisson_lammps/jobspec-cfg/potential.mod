# NOTE: This script can be modified for different pair styles 
# See in.elastic for more info.

# Choose potential
pair_style     rebo
pair_coeff     * * CH.airebo C

#pair_style     lcbop
#pair_coeff     C.lcbop
#pair_style     tersoff
#pair_coeff     BNC.tersoff


# Setup neighbor style
neighbor 1.0 nsq
neigh_modify once no every 1 delay 0 check yes

# Setup minimization style
min_style	     cg
min_modify	     dmax ${dmax} line quadratic

# Setup output
thermo		500
thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
thermo_modify norm no


# Output
dump            1 all xtc 100 traj.xtc
dump            2 all xyz 2000 relaxed.xyz
#dump            4 all atom 10 dump.atom
