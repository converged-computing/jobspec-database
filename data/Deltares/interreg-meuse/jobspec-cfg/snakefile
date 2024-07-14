### Import some useful python library
import os
import shutil
import itertools
import numpy as np
from datetime import datetime
from snakemake.io import Wildcards

# Parsing the Snakemake config file (options)
f_proj = config["proj_dir"]
f_setup = config["data_dir"]
f_input = config['data_source']
f_data = config["input_folder"]
f_wflow = config["wflow_dir"]

#Data preprocessing directories
f_unzipped = f_setup + "/" + f_input + "/a_raw"
f_modif = f_setup + "/" + f_input + "/b_preprocess"
f_wflow_input = f_setup + "/" + f_input + "/c_wflow"
f_figures = f_proj + "/Figures/" +  f_input

#wflow model specifics
exp_name = config['data_source']
model = config['wflow_model']
year_start = int(datetime.strptime(config['wflow_params']['starttime'], '%Y-%m-%dT%H:%M:%S').year)
year_end = int(datetime.strptime(config['wflow_params']['endtime'], '%Y-%m-%dT%H:%M:%S').year)

# def get_member_name(wildcards):
#     return config["members"][wildcards.member_nb]["name"]
# def get_zip_name(wildcards):
#     return config["dts"][wildcards.dts]["name_zip"]
def get_zip_main_fn_name(wildcards):
    return config["dts"][wildcards.dt]["name_main"]
def get_extension(wildcards):
    return config["dts"][wildcards.dt]["ext"]

#We create the folder we need to log the output from the snakemake pipeline
# onstart:
#     print("##### Creating profile pipeline #####\n") 
#     print("\t Creating jobs output subfolders...\n")
#     shell("mkdir -p jobs/all jobs/unzip jobs/cdo_regrid jobs/ds_convert_merge jobs/figure_forcing jobs/update_toml_wflow jobs/run_wflow")


rule all:
    input:
        expand((f"{f_wflow_input}"+"/{dt}"+"/{member_nb}/"+"ds_merged_{year}.nc"), dt = config["dts"], member_nb = config["members"], year= np.arange(year_start,year_end+1)),   
        expand(f"{f_wflow}"+f"/{model}"+f"/{exp_name}"+"_{dt}"+"/{member_nb}"+"/output.csv", dt = config["dts"], member_nb = config["members"]),
        expand(f"{f_figures}"+"/{dt}"+"/{member_nb}"+"/precip_sum.png", dt = config["dts"], member_nb = config["members"]),
        expand(f"{f_unzipped}"+"/{dt}"+"/full_ds"+"/{member_nb}/"+"{var}"+"/{var}"+".KNMI-{year}.{member_nb}"+".nc", dt = config["dts"], member_nb = config["members"], var = config["variables"], year= np.arange(year_start,year_end))


rule unzip:
    input:
        f_zip = os.path.join(f_setup, f_input, f_data, "{dt}",'data.zip')
    params:
        member_number = "{member_nb}",
        main_folder =  get_zip_main_fn_name,
        ext =  get_extension,
        year_name = "{year}",
        var_name = "{var}",
        dt_name = "{dt}",
        extract_to = f_unzipped
    output:
        #expand((f"{f_unzipped}"+"/{dt}"+"/full_ds"+"/{member_nb}/"+"{var}"+"/{var}"+".KNMI-{year}.{member_nb}"+".nc"), dt = config["dts"], member_nb = config["members"], var = config["variables"], year= np.arange(year_start,year_end))
        (f"{f_unzipped}"+"/{dt}"+"/full_ds"+"/{member_nb}/"+"{var}"+"/{var}"+".KNMI-{year}.{member_nb}"+".nc")
    resources:
        mem_mb=1,
        disk_mb=1
    conda:
        "envs/env_cdo.yaml"
    script:
        "src/preprocess/unzip_knmi.py"

rule cdo_regrid:
    input:
        (f"{f_unzipped}"+"/{dt}"+"/full_ds"+"/{member_nb}/"+"{var}"+"/{var}"+".KNMI-{year}.{member_nb}"+".nc")
    params:
        f_src = f_unzipped,
        f_dst = f_modif,
        grid_fn = config["cdo_grid"],
        dt_step = "{dt}",
        var_name = "{var}"
    output:
        fn_out = f"{f_modif}"+"/{dt}"+"/{member_nb}/"+"{var}"+"/{var}"+".KNMI-{year}.{member_nb}_regrid_meuse"+".nc"
    resources:
        mem_mb=1,
        disk_mb=1
    conda:
        "envs/env_cdo.yaml"
    script:
        "src/preprocess/cdo_regrid_script.py"

rule ds_convert_merge:
    input:
        fn_temp = f"{f_modif}"+"/{dt}"+"/{member_nb}/"+"t2m"+"/t2m.KNMI-{year}.{member_nb}_regrid_meuse"+".nc",
        fn_pet = f"{f_modif}"+"/{dt}"+"/{member_nb}/"+"pet"+"/pet.KNMI-{year}.{member_nb}_regrid_meuse"+".nc",
        fn_precip = f"{f_modif}"+"/{dt}"+"/{member_nb}/"+"precip"+"/precip.KNMI-{year}.{member_nb}_regrid_meuse"+".nc"
    params:
        conv_params = config["variables"],
        dt_step = "{dt}",
        year_name = "{year}", 
    output:
        fn_out = f"{f_wflow_input}"+"/{dt}"+"/{member_nb}/"+"ds_merged_{year}.nc"
    resources:
        mem_mb=1,
        disk_mb=1
    conda:
        "envs/env_hydromt_wflow.yaml"
    script:
        "src/preprocess/convert_nc.py"

rule figure_forcing:
    input:
        fn_forcing = [(f"{f_wflow_input}"+"/{dt}"+"/{member_nb}/"+f"ds_merged_{years}.nc") for years in np.arange(year_start,year_end+1)] 
    params:
        year_random = np.random.randint(year_start,year_end+1)
    output:
        f"{f_figures}"+"/{dt}"+"/{member_nb}"+"/precip_sum.png",
        f"{f_figures}"+"/{dt}"+"/{member_nb}"+"/pet_sum.png",   
        f"{f_figures}"+"/{dt}"+"/{member_nb}"+"/temp_max.png", 
    conda:
        "envs/env_hydromt_wflow.yaml"
    script:
        "src/preprocess/convert_nc_figures.py"  

rule update_toml_wflow:
    input:
        fn_in = f"{f_wflow_input}"+"/{dt}"+"/{member_nb}/"+f"ds_merged_{year_start}.nc"
    params:
        wflow_params = config["wflow_params"], #Change this to have the size of the years if they are missing
        wflow_base_toml = config["wflow_base_toml"],
        timestep = "{dt}",
        exp_name = config['data_source'],
        model = config['wflow_model'], 
        fn_wflow = f"{f_wflow}",
        fn_forcing = f"{f_wflow_input}"+"/{dt}"+"/{member_nb}",
        member_nb = "{member_nb}",
        start_path = f"{f_wflow}"+f"/{model}"+f"/{exp_name}"+"_{dt}"+"/{member_nb}",
        fn_path_input = config['wflow_path_input']
    output:
        fn_out = f"{f_wflow}"+f"/{model}"+f"/{exp_name}"+"_{dt}"+"/{member_nb}"+f"/{exp_name}"+"_{dt}"+"_{member_nb}.toml"
    conda:
        "envs/env_hydromt_wflow.yaml"
    script:
        "src/model_building/update_toml_wflow.py"
    
rule run_wflow:
    input: #We need the tomls!
        fn_toml = f"{f_wflow}"+f"/{model}"+f"/{exp_name}"+"_{dt}"+"/{member_nb}"+f"/{exp_name}"+"_{dt}"+"_{member_nb}.toml",
        fn_in = [(f"{f_wflow_input}"+"/{dt}"+"/{member_nb}/"+f"ds_merged_{years}.nc") for years in np.arange(year_start,year_end+1)]
    params:
        wd = f"{f_wflow}"+f"/{model}"+f"/{exp_name}"+"_{dt}"+"/{member_nb}",
        julia_env_fn = config["julia_env_fn"]
    output:
        csv_file = f"{f_wflow}"+f"/{model}"+f"/{exp_name}"+"_{dt}"+"/{member_nb}"+"/output.csv"
    shell:
        """
        julia --threads 4 --project={params.julia_env_fn} -e "using Wflow; Wflow.run()" "{input.fn_toml}" 
        """

#rule eva: #to be added



#read also; https://taylorreiter.github.io/2020-02-03-How-to-use-snakemake-checkpoints-to-extract-files-from-an-archive/