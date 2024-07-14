import os
import click
import numpy as np
import mlflow
import json
import numpy.random as rn

from bgnmf import BGNMF

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
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", default="stability_experiment_bgnmf")
    experiment_id = mlflow.create_experiment(experiment_name, artifact_location=output_path)

    # Log parameters C, K
    mlflow.start_run(experiment_id = experiment_id)
    mlflow.log_params({"K": K,
                       "data": data_path,
                       "seed": seed})

    # Instantiate the Model
    rn.seed(seed)
    bgnmf_model = BGNMF(n_components = K,
                         tol=1e-2, 
                         max_iter=200)
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
            bgnmf_model.fit(data_IJ)

        A_IK = bgnmf_model.A_IK
        B_IK = bgnmf_model.B_IK
        H_KJ = bgnmf_model.H_KJ

    # Write the Model parameters
    state_name = f"uuid_{uuid}_state_bgnmf.npz"
    np.savez_compressed(os.path.join(output_path,state_name), A_IK=A_IK, B_IK=B_IK, H_KJ=H_KJ)
    print("File path:", os.path.join(output_path,state_name))
    print("A_IK shape:", A_IK.shape, "B_IK shape:", B_IK.shape, "H_KJ shape:", H_KJ.shape)
    
    # Log results
    mlflow.log_artifact(os.path.join(output_path,state_name))
    mlflow.end_run()
if __name__ == '__main__':
    main()