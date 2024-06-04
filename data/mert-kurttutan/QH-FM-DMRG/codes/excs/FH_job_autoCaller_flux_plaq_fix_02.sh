Lx_arr=(12 6 8 16)
Ly_arr=(4 5)
U_arr=(8)
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for Lx in "${Lx_arr[@]}"; do
    for Ly in "${Ly_arr[@]}"; do
        Nend_t=$(echo "scale=0; $Lx*$Ly/2" | bc -l)        #quarter-filling
        Nend=${Nend_t%.*}      #typecasting to integer
        N0_t=$(echo "scale=0; 2" | bc -l)          #a 2-particles
        N0=${N0_t%.*}

        for ((n=$N0; n<=$Nend; n+=2)); do             #array to interpolate between these two
            Smax_t=$(echo "scale=0; $n*0.5" | bc -l)       #max Spin sector
            Smax=${Smax_t%.*}
            S2_t=$(echo "scale=0; $n*0.5*0.5" | bc -l)       #half Spin sector
            S2=${S2_t%.*}
            S_arr=(0 $Smax)

            for S in "${S_arr[@]}"; do                 #all the spin sectors
                for U in "${U_arr[@]}"; do
                        nPhi=$(echo "scale=8; ($Lx-1)*$Ly*0.2" | bc -l)      #flux per plaquette=0.2
                    source ${__dir}/FHH-SU2-job_01.sh $Lx $Ly $nPhi $U $n $S 1.0
                    #echo $Lx, $Ly, $S, $n, $U, $nPhi
                done
            done
        done
    done
done