import os
import click
import numpy as np
import mlflow
import json

from bgnmf import BGNMF
from dncbtd import DNCBTD, initialize_DNCBTD_with_BGNMF

@click.command()
@click.argument('json_string')
def main(json_string, n_burnin=4, n_epochs=3, n_itns=2, verbose=0):

    json_dict = json.loads(json_string)

    # if these params not in json_dict then throw an exception
    data_path = json_dict['data_path']
    output_path = json_dict['output_path']
    C = json_dict['C']
    K = json_dict['K']
    uuid = json_dict['uuid']

    data_dict = np.load(data_path)
    data_IJ = np.ascontiguousarray(data_dict['Beta_IJ'])
    I,J = data_IJ.shape

    # Set experiment for run
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", default="experiment_1")
    experiment_id = mlflow.create_experiment(experiment_name, artifact_location=output_path)

    # Log parameters C, K
    mlflow.start_run(experiment_id = experiment_id)
    mlflow.log_params({"C": C, 
                       "K": K,
                       "data": data_path})

    # Instantiate the Model
    dncbtd_param_names = set(['C','K','bm','bu','pi_prior','phi_prior','theta_prior','shp_delta','rte_delta','shp_theta',
                     'shp_phi','shp_pi','debug','seed','n_threads'])
    dncbtd_param_dict = {x:json_dict[x] for x in json_dict
                              if x in dncbtd_param_names}
    prbf_model = DNCBTD(I=I, J=J, **dncbtd_param_dict)

    # Initialize the Model
    initialize_DNCBTD_with_BGNMF(prbf_model, data_IJ,  verbose=verbose, n_itns=5)

    # Create the output path
    samples_path = os.path.join(output_path, 'samples')
    os.makedirs(samples_path, exist_ok = True)

    # Fit the Model
    for epoch in range(n_epochs+2):
        if epoch > 0:
            prbf_model.fit(C = C,
                           K = K,
                           data_IJ = data_IJ, 
                           n_itns=n_itns if epoch > 1 else n_burnin,
                           verbose=verbose,
                           initialize=False,
                           schedule={},
                           fix_state={}
                           )

        state = dict(prbf_model.get_state())
        Theta_IC = state['Theta_IC']
        Phi_KJ = state['Phi_KJ']
        Pi_2CK = state['Pi_2CK']
        Pi_CK = Pi_2CK[0, :, :]       # clusters (C) x pathways (K) core matrix
        Y_IC = state['Y_IC']
        Y_KJ = state['Y_KJ']
        Y_2CK = state['Y_2CK']
        Y_2IJ = state['Y_2IJ']

    # Write the Model parameters
    state_name = f"uuid_{uuid}_state_{prbf_model.total_itns}.npz"
    np.savez_compressed(os.path.join(samples_path,state_name), Theta_IC = Theta_IC, Phi_KJ = Phi_KJ, Pi_CK = Pi_CK, Y_IC = Y_IC, Y_KJ = Y_KJ, Y_2CK = Y_2CK, Y_2IJ = Y_2IJ)
    
    # Log results
    mlflow.log_artifact(os.path.join(samples_path,state_name))
    mlflow.end_run()
if __name__ == '__main__':
    main()