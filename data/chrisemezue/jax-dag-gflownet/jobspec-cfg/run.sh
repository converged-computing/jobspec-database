#!/bin/bash

#for baseline in "bcdnets" "bootstrap_ges" "bootstrap_pc" "dag_gflownet" "dibs" "gadget" "mc3"
#for baseline in "dibs" "bootstrap_pc"
for case in "treatment_effect_with_common_child" "treatment_effect_with_mediator" "treatment_effect_with_cofounder_no_effect" "treatment_effect_with_cofounder"
do
    for baseline in "bcdnets" "bootstrap_ges" "bootstrap_pc" "dibs" "gadget" "mc3"

    do    
        sbatch job_ci.sh $baseline $case

    done
done

