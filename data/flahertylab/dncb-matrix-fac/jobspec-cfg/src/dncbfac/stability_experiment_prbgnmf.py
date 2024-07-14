import os
import click
import numpy as np
import mlflow
import json

from prbgnmf import PRBGNMF, initialize_PRBGNMF_with_BGNMF

@click.command()
@click.argument('json_string')
def main(json_string, test = False, verbose=0):

    json_dict = json.loads(json_string)

    # if these params not in json_dict then throw an exception
    data_path = json_dict['data_path']
    output_path = json_dict['output_path']
    K = json_dict['K']
    seed = json_dict['seed']
    uuid = json_dict['uuid']

    data_dict = np.load(data_path)
    data_IJ = np.ascontiguousarray(data_dict['Beta_IJ'])
    I,J = data_IJ.shape

    # Set experiment for run
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", default="stability_experiment_prbgnmf")
    experiment_id = mlflow.create_experiment(experiment_name, artifact_location=output_path)

    # Log parameters C, K
    mlflow.start_run(experiment_id = experiment_id)
    mlflow.log_params({"K": K,
                       "data": data_path})

    # Instantiate the Model
    dncbmf_param_names = set(['K','shp_a', 'rte_a', 'shp_h', 'rte_h','seed','n_threads'])
    dncbmf_param_dict = {x:json_dict[x] for x in json_dict
                              if x in dncbmf_param_names}
    model = PRBGNMF(I = I,J = J,**dncbmf_param_dict)

    # Initialize the Model
    initialize_PRBGNMF_with_BGNMF(model, data_IJ,  verbose=verbose, n_itns=5)

    # Create the output path
    # samples_path = os.path.join(output_path, 'samples')
    # os.makedirs(samples_path, exist_ok = True)

    n_burnin = 1000
    n_epochs = 50
    n_itns = 20

    if test:
        n_burnin = 4
        n_epochs = 3
        n_itns = 2
        
    # Fit the Model
    for epoch in range(n_epochs+2):
        if epoch > 0:
            model.fit(Beta_IJ = data_IJ, 
                           n_itns=n_itns if epoch > 1 else n_burnin,
                           verbose=verbose,
                           initialize=False,
                           schedule={},
                           fix_state={}
                           )

        state = dict(model.get_state())
        A_IK = state['A_IK']
        B_IK = state['B_IK']
        H_KJ = state['H_KJ']

    # Write the Model parameters
    state_name = f"uuid_{uuid}_state_{model.total_itns}.npz"
    np.savez_compressed(os.path.join(output_path,state_name), A_IK = A_IK, B_IK = B_IK, H_KJ = H_KJ)

    # Log results
    mlflow.log_artifact(os.path.join(output_path,state_name))
    mlflow.end_run()
    
if __name__ == '__main__':
    main()