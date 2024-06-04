Lx_arr=(31 36 )
Ly_arr=(5 )

U_arr=(8 )
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
y0=0
e=1
for Lx in "${Lx_arr[@]}"; do
    for Ly in "${Ly_arr[@]}"; do
        Nend_t=$(echo "scale=0; $Lx*$Ly*0.1" | bc -l)        
        Nend=${Nend_t%.*}     #typecasting to integer
        Nend=$((2*Nend + 4))
        N0_t=$(echo "scale=0; $Lx*$Ly*0.1" | bc -l)         
        N0=${N0_t%.*}
        N0=$((2*N0 - 4))
        x0=$(echo "scale=0; $Lx*0.5" | bc -l)
        nPhi=$(echo "scale=8; ($Lx-1)*$Ly*0.2" | bc -l)      #flux per plaquette=0.2
        for ((n=$N0; n<=$Nend; n+=2)); do             #array to interpolate between these two
            Smax_t=$(echo "scale=0; $n*0.5" | bc -l)       #max Spin sector
            Smax=${Smax_t%.*}
            S2_t=$(echo "scale=0; $n*0.5*0.5" | bc -l)       #half Spin sector
            S2=${S2_t%.*}
            S_arr=(0)
            for S in "${S_arr[@]}"; do                 #all the spin sectors
                for U in "${U_arr[@]}"; do
                    if (( $(echo "$nPhi > $n" |bc -l) )); then
                        g_arr=(0.0 0.1 0.5 1.0 4.0 )

                    elif (( $(echo "$nPhi == $n" |bc -l) )); then
                        g_arr=(0.0)
                    else
                        g_arr=(0.0 -0.1 -0.5 -1.0 -4.0 )
                    fi  
                    
                    for g in "${g_arr[@]}"; do
                        sbatch ${__dir}/FHH-SU2-job_01.sh $Lx $Ly $nPhi $U $n $S 1.0 $g $x0 $y0
                        #echo $Lx, $Ly, $S, $n, $U, $nPhi, $g $x0
                        #echo $e
                        #e=$((e + 1))
                    done
                done
            done
        done
    done
done