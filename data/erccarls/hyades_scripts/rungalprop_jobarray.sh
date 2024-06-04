    #!/bin/bash
    #PBS -N galprop_mod_p_
    #PBS -l nodes=1:ppn=32
    #PBS -l pmem=6gb
    #PBS -l walltime=12:00:00
    #PBS -t 0-56%32
    #PBS -q hyper

#    /pfs/carlson/galprop/GALDEF/run_script_mod_k_"$PBS_ARRAYID".sh


export OMP_NUM_THREADS=32
/pfs/carlson/galprop/bin/galprop -r mod_s2_"$PBS_ARRAYID" -o /pfs/carlson/galprop/output/

cd /pfs/carlson/GCE_sys/
    python /pfs/carlson/GCE_sys/GenDiffuseModel_X_CO_3zone_P8R2.py /pfs/carlson/galprop/output mod_s2_"$PBS_ARRAYID" /pfs/carlson/galprop/GALDEF    

python RunAnalysis_P8.py /pfs/carlson/galprop/output/ mod_s_"$PBS_ARRAYID"_XCO_P8 0
