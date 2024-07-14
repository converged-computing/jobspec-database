#!/usr/bin/env python3
# coding: utf-8


import h5py
import numpy as np
from scipy.stats.qmc import LatinHypercube
from tqdm.auto import trange

import an_cockrell
from an_cockrell import EpiType

# # An-Cockrell model reimplementation

# constants
init_inoculum = 100
num_sims = 10_000
num_steps = 2016  # <- full run value

default_params = dict(
    GRID_WIDTH=51,
    GRID_HEIGHT=51,
    is_bat=False,
    init_inoculum=100,
    init_dcs=50,
    init_nks=25,
    init_macros=50,
    macro_phago_recovery=0.5,
    macro_phago_limit=1_000,
    inflammasome_activation_threshold=10,  # default 50 for bats
    inflammasome_priming_threshold=1.0,  # default 5.0 for bats
    viral_carrying_capacity=500,
    susceptibility_to_infection=77,
    human_endo_activation=5,
    bat_endo_activation=10,
    bat_metabolic_byproduct=2.0,
    human_metabolic_byproduct=0.2,
    viral_incubation_threshold=60,
    epi_apoptosis_threshold_lower=450,
    epi_apoptosis_threshold_range=100,
    epi_apoptosis_threshold_lower_regrow=475,
    epi_apoptosis_threshold_range_regrow=51,
    epi_regrowth_counter_threshold=432,
    epi_cell_membrane_init_lower=975,
    epi_cell_membrane_init_range=51,
    infected_epithelium_ros_damage_counter_threshold=10,
    epithelium_ros_damage_counter_threshold=2,
    epithelium_pdamps_secretion_on_death=10.0,
    dead_epithelium_pdamps_burst_secretion=10.0,
    dead_epithelium_pdamps_secretion=1.0,
    epi_max_tnf_uptake=0.1,
    epi_max_il1_uptake=0.1,
    epi_t1ifn_secretion=0.75,
    epi_t1ifn_secretion_prob=0.01,
    epi_pdamps_secretion_prob=0.01,
    infected_epi_t1ifn_secretion=1.0,
    infected_epi_il18_secretion=0.11,
    infected_epi_il6_secretion=0.10,
    activated_endo_death_threshold=0.5,
    activated_endo_adhesion_threshold=36.0,
    activated_endo_pmn_spawn_prob=0.1,
    activated_endo_pmn_spawn_dist=5.0,
    extracellular_virus_init_amount_lower=80,
    extracellular_virus_init_amount_range=40,
    human_viral_lower_bound=0.0,
    human_t1ifn_effect_scale=0.01,
    pmn_max_age=36,
    pmn_ros_secretion_on_death=10.0,
    pmn_il1_secretion_on_death=1.0,
    nk_ifng_secretion=1.0,
    macro_max_virus_uptake=10.0,
    macro_activation_threshold=5.0,
    activated_macro_il8_secretion=1.0,
    activated_macro_il12_secretion=0.5,
    activated_macro_tnf_secretion=1.0,
    activated_macro_il6_secretion=0.4,
    activated_macro_il10_secretion=1.0,
    macro_antiactivation_threshold=5.0,
    antiactivated_macro_il10_secretion=0.5,
    inflammasome_il1_secretion=1.0,
    inflammasome_macro_pre_il1_secretion=5.0,
    inflammasome_il18_secretion=1.0,
    inflammasome_macro_pre_il18_secretion=0.5,
    pyroptosis_macro_pdamps_secretion=10.0,
    dc_t1ifn_activation_threshold=1.0,
    dc_il12_secretion=0.5,
    dc_ifng_secretion=0.5,
    dc_il6_secretion=0.4,
    dc_il6_max_uptake=0.1,
    extracellular_virus_diffusion_const=0.05,
    T1IFN_diffusion_const=0.1,
    PAF_diffusion_const=0.1,
    ROS_diffusion_const=0.1,
    P_DAMPS_diffusion_const=0.1,
    IFNg_diffusion_const=0.2,
    TNF_diffusion_const=0.2,
    IL6_diffusion_const=0.2,
    IL1_diffusion_const=0.2,
    IL10_diffusion_const=0.2,
    IL12_diffusion_const=0.2,
    IL18_diffusion_const=0.2,
    IL8_diffusion_const=0.3,
    extracellular_virus_cleanup_threshold=0.05,
    cleanup_threshold=0.1,
    evap_const_1=0.99,
    evap_const_2=0.9,
)

variational_params = [
    "init_inoculum",
    "init_dcs",
    "init_nks",
    "init_macros",
    "macro_phago_recovery",
    "macro_phago_limit",
    "inflammasome_activation_threshold",
    "inflammasome_priming_threshold",
    "viral_carrying_capacity",
    "susceptibility_to_infection",
    "human_endo_activation",
    "human_metabolic_byproduct",
    "viral_incubation_threshold",
    "epi_apoptosis_threshold_lower",
    "epi_apoptosis_threshold_range",
    "epi_apoptosis_threshold_lower_regrow",
    "epi_apoptosis_threshold_range_regrow",
    "epi_regrowth_counter_threshold",
    "epi_cell_membrane_init_lower",
    "epi_cell_membrane_init_range",
    "infected_epithelium_ros_damage_counter_threshold",
    "epithelium_ros_damage_counter_threshold",
    "epithelium_pdamps_secretion_on_death",
    "dead_epithelium_pdamps_burst_secretion",
    "dead_epithelium_pdamps_secretion",
    "epi_max_tnf_uptake",
    "epi_max_il1_uptake",
    "epi_t1ifn_secretion",
    "epi_t1ifn_secretion_prob",
    "epi_pdamps_secretion_prob",
    "infected_epi_t1ifn_secretion",
    "infected_epi_il18_secretion",
    "infected_epi_il6_secretion",
    "activated_endo_death_threshold",
    "activated_endo_adhesion_threshold",
    "activated_endo_pmn_spawn_prob",
    "activated_endo_pmn_spawn_dist",
    "extracellular_virus_init_amount_lower",
    "extracellular_virus_init_amount_range",
    "human_t1ifn_effect_scale",
    "pmn_max_age",
    "pmn_ros_secretion_on_death",
    "pmn_il1_secretion_on_death",
    "nk_ifng_secretion",
    "macro_max_virus_uptake",
    "macro_activation_threshold",
    "activated_macro_il8_secretion",
    "activated_macro_il12_secretion",
    "activated_macro_tnf_secretion",
    "activated_macro_il6_secretion",
    "activated_macro_il10_secretion",
    "macro_antiactivation_threshold",
    "antiactivated_macro_il10_secretion",
    "inflammasome_il1_secretion",
    "inflammasome_macro_pre_il1_secretion",
    "inflammasome_il18_secretion",
    "inflammasome_macro_pre_il18_secretion",
    "pyroptosis_macro_pdamps_secretion",
    "dc_t1ifn_activation_threshold",
    "dc_il12_secretion",
    "dc_ifng_secretion",
    "dc_il6_secretion",
    "dc_il6_max_uptake",
    # # ACK's Executive Judgement: These are physics-like parameters and won't vary between individuals.
    # "human_viral_lower_bound", 0.0
    # "extracellular_virus_diffusion_const",
    # "T1IFN_diffusion_const",
    # "PAF_diffusion_const",
    # "ROS_diffusion_const",
    # "P_DAMPS_diffusion_const",
    # "IFNg_diffusion_const",
    # "TNF_diffusion_const",
    # "IL6_diffusion_const",
    # "IL1_diffusion_const",
    # "IL10_diffusion_const",
    # "IL12_diffusion_const",
    # "IL18_diffusion_const",
    # "IL8_diffusion_const",
    # "extracellular_virus_cleanup_threshold",
    # "cleanup_threshold",
    # "evap_const_1",
    # "evap_const_2",
]

param_list = np.full((num_sims, len(variational_params)), -1, dtype=np.float64)
total_P_DAMPS = np.full((num_sims, num_steps), -1, dtype=np.float64)
total_T1IFN = np.full((num_sims, num_steps), -1, dtype=np.float64)
total_TNF = np.full((num_sims, num_steps), -1, dtype=np.float64)
total_IFNg = np.full((num_sims, num_steps), -1, dtype=np.float64)
total_IL6 = np.full((num_sims, num_steps), -1, dtype=np.float64)
total_IL1 = np.full((num_sims, num_steps), -1, dtype=np.float64)
total_IL8 = np.full((num_sims, num_steps), -1, dtype=np.float64)
total_IL10 = np.full((num_sims, num_steps), -1, dtype=np.float64)
total_IL12 = np.full((num_sims, num_steps), -1, dtype=np.float64)
total_IL18 = np.full((num_sims, num_steps), -1, dtype=np.float64)
total_extracellular_virus = np.full((num_sims, num_steps), -1, dtype=np.float64)
total_intracellular_virus = np.full((num_sims, num_steps), -1, dtype=np.float64)
apoptosis_eaten_counter = np.full((num_sims, num_steps), -1, dtype=np.float64)
infected_epis = np.full((num_sims, num_steps), -1, dtype=np.float64)
dead_epis = np.full((num_sims, num_steps), -1, dtype=np.float64)
apoptosed_epis = np.full((num_sims, num_steps), -1, dtype=np.float64)
system_health = np.full((num_sims, num_steps), -1, dtype=np.float64)

with h5py.File("simulation-statistics.hdf5", "w") as f:
    f.create_dataset(
        "param_list",
        (num_sims, len(variational_params)),
        dtype=np.float64,
        data=param_list,
        chunks=(100, len(variational_params)),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "total_P_DAMPS",
        (num_sims, num_steps),
        dtype=np.float64,
        data=total_P_DAMPS,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "total_T1IFN",
        (num_sims, num_steps),
        dtype=np.float64,
        data=total_T1IFN,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "total_TNF",
        (num_sims, num_steps),
        dtype=np.float64,
        data=total_TNF,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "total_IFNg",
        (num_sims, num_steps),
        dtype=np.float64,
        data=total_IFNg,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "total_IL6",
        (num_sims, num_steps),
        dtype=np.float64,
        data=total_IL6,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "total_IL1",
        (num_sims, num_steps),
        dtype=np.float64,
        data=total_IL1,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "total_IL8",
        (num_sims, num_steps),
        dtype=np.float64,
        data=total_IL8,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "total_IL10",
        (num_sims, num_steps),
        dtype=np.float64,
        data=total_IL10,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "total_IL12",
        (num_sims, num_steps),
        dtype=np.float64,
        data=total_IL12,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "total_IL18",
        (num_sims, num_steps),
        dtype=np.float64,
        data=total_IL18,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "total_extracellular_virus",
        (num_sims, num_steps),
        dtype=np.float64,
        data=total_extracellular_virus,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "total_intracellular_virus",
        (num_sims, num_steps),
        dtype=np.float64,
        data=total_intracellular_virus,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "apoptosis_eaten_counter",
        (num_sims, num_steps),
        dtype=np.float64,
        data=apoptosis_eaten_counter,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "infected_epis",
        (num_sims, num_steps),
        dtype=np.float64,
        data=infected_epis,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "dead_epis",
        (num_sims, num_steps),
        dtype=np.float64,
        data=dead_epis,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "apoptosed_epis",
        (num_sims, num_steps),
        dtype=np.float64,
        data=apoptosed_epis,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )
    f.create_dataset(
        "system_health",
        (num_sims, num_steps),
        dtype=np.float64,
        data=system_health,
        chunks=(100, num_steps),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
        fletcher32=True,
    )


lhc = LatinHypercube(len(variational_params))
sample = 1.0 + 0.5 * (lhc.random(n=num_sims) - 0.5)  # between 75% and 125%

# noinspection PyTypeChecker
for sim_idx in trange(num_sims, desc="simulation"):
    # generate a perturbation of the default parameters
    params = default_params.copy()
    pct_perturbation = sample[sim_idx]
    for pert_idx, param in enumerate(variational_params):
        if isinstance(params[param], int):
            params[param] = int(round(float(pct_perturbation[pert_idx] * params[param])))
        else:
            params[param] = float(pct_perturbation[pert_idx] * params[param])

    param_list[sim_idx, :] = np.array([params[param] for param in variational_params])

    model = an_cockrell.AnCockrellModel(**params)

    # noinspection PyTypeChecker
    for step_idx in trange(num_steps):
        model.time_step()

        total_P_DAMPS[sim_idx, step_idx] = model.total_P_DAMPS
        total_T1IFN[sim_idx, step_idx] = model.total_T1IFN
        total_TNF[sim_idx, step_idx] = model.total_TNF
        total_IFNg[sim_idx, step_idx] = model.total_IFNg
        total_IL6[sim_idx, step_idx] = model.total_IL6
        total_IL1[sim_idx, step_idx] = model.total_IL1
        total_IL8[sim_idx, step_idx] = model.total_IL8
        total_IL10[sim_idx, step_idx] = model.total_IL10
        total_IL12[sim_idx, step_idx] = model.total_IL12
        total_IL18[sim_idx, step_idx] = model.total_IL18
        total_extracellular_virus[sim_idx, step_idx] = model.total_extracellular_virus
        total_intracellular_virus[sim_idx, step_idx] = model.total_intracellular_virus
        apoptosis_eaten_counter[sim_idx, step_idx] = model.apoptosis_eaten_counter
        infected_epis[sim_idx, step_idx] = np.sum(model.epithelium == EpiType.Infected)
        dead_epis[sim_idx, step_idx] = np.sum(model.epithelium == EpiType.Dead)
        apoptosed_epis[sim_idx, step_idx] = np.sum(model.epithelium == EpiType.Apoptosed)
        system_health[sim_idx, step_idx] = model.system_health

    with h5py.File("simulation-statistics.hdf5", "r+") as f:
        f["param_list"][sim_idx, :] = param_list[sim_idx, :]
        f["total_P_DAMPS"][sim_idx, :] = total_P_DAMPS[sim_idx, :]
        f["total_T1IFN"][sim_idx, :] = total_T1IFN[sim_idx, :]
        f["total_TNF"][sim_idx, :] = total_TNF[sim_idx, :]
        f["total_IFNg"][sim_idx, :] = total_IFNg[sim_idx, :]
        f["total_IL6"][sim_idx, :] = total_IL6[sim_idx, :]
        f["total_IL1"][sim_idx, :] = total_IL1[sim_idx, :]
        f["total_IL8"][sim_idx, :] = total_IL8[sim_idx, :]
        f["total_IL10"][sim_idx, :] = total_IL10[sim_idx, :]
        f["total_IL12"][sim_idx, :] = total_IL12[sim_idx, :]
        f["total_IL18"][sim_idx, :] = total_IL18[sim_idx, :]
        f["total_extracellular_virus"][sim_idx, :] = total_extracellular_virus[sim_idx, :]
        f["total_intracellular_virus"][sim_idx, :] = total_intracellular_virus[sim_idx, :]
        f["apoptosis_eaten_counter"][sim_idx, :] = apoptosis_eaten_counter[sim_idx, :]
        f["infected_epis"][sim_idx, :] = infected_epis[sim_idx, :]
        f["dead_epis"][sim_idx, :] = dead_epis[sim_idx, :]
        f["apoptosed_epis"][sim_idx, :] = apoptosed_epis[sim_idx, :]
        f["system_health"][sim_idx, :] = system_health[sim_idx, :]
