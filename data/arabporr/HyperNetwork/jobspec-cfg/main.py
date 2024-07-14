import projection
import MLP_Handler
import HN_Handler
import Test_and_plot
import os
import shutil

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data_index", type=int, default=0)
parser.add_argument("--Mem_Optimize", type=int, default=0)

args = parser.parse_args()
data_index = args.data_index
Mem_Optimize = args.Mem_Optimize

projection.OU_Projection_Problem(data_index)
MLP_Handler.Run(data_index)
HN_Handler.Run(data_index)
Test_and_plot.Run(data_index)


try:
    destination_path = "/h/rezad/Run_Results_1/" + "res_" + str(data_index) + "/"
    os.makedirs(destination_path, exist_ok=True)
    source_path_data = "problem_instance_" + str(data_index) + ".pt"
    source_path_mlp_params = (
        "MLP_Log_problem_" + str(data_index) + "/MLPs_parameters.pt"
    )
    source_path_hn_model = "HN_Log_problem_" + str(data_index) + "/HN_model.pt"
    source_path_results = "Results_" + str(data_index) + "/"
    sources = [
        source_path_data,
        source_path_mlp_params,
        source_path_hn_model,
        source_path_results,
    ]
    for item in sources:
        shutil.move(item, destination_path)
except:
    print("error in moving files is data_index: ", data_index)


if Mem_Optimize:
    shutil.rmtree("MLP_Log_problem_" + str(data_index) + "/")
    shutil.rmtree("HN_Log_problem_" + str(data_index) + "/")
    shutil.rmtree("Testing_Results_problem_" + str(data_index) + "/")
