import os
import click
import numpy as np
import numpy.random as rn
import mlflow
import json
import numpy.random as rn
import scipy.stats as st

from bgnmf import BGNMF

@click.command()
@click.argument('json_string')
def main(json_string, test = False, verbose=0):

    json_dict = json.loads(json_string)

    # if these params not in json_dict then throw an exception
    data_path = json_dict['data_path']
    output_path = json_dict['output_path']
    K = json_dict['K']
    mask_seed = json_dict['mask_seed']
    seed = json_dict['seed']
    uuid = json_dict['uuid']

    data_dict = np.load(data_path)
    data_IJ = np.ascontiguousarray(data_dict['Beta_IJ'])
    I,J = data_IJ.shape

    # Set experiment for run
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", default="heldout_experiment_prbgnmf")
    experiment_id = mlflow.create_experiment(experiment_name, artifact_location=output_path)

    # Log parameters C, K
    mlflow.start_run(experiment_id = experiment_id)
    mlflow.log_params({"K": K,
                       "data": data_path,
                       "mask_seed": mask_seed,
                       "model_seed": seed,
                       "uuid": uuid})

    # Create masked data
    rn.seed(mask_seed)
    mask_p = 0.1
    mask_IJ = (rn.random(size=(I, J)) < mask_p).astype(int) 
    masked_data_IJ = np.ma.core.MaskedArray(data_IJ, mask=mask_IJ)

    # These are the subscripts (subs) of the heldout data entries.
    heldout_subs = mask_IJ.nonzero()
    # These are the values (vals) of the heldout data entries.
    heldout_vals = data_IJ[heldout_subs]

    # Instantiate the Model
    rn.seed(seed)
    bgnmf_model = BGNMF(n_components= K, 
                        tol=1e-2, 
                        max_iter=200)

    n_burnin = 1000
    n_epochs = 50
    n_itns = 20

    if test:
        n_burnin = 4
        n_epochs = 3
        n_itns = 2
    
    for epoch in range(n_epochs+2):
        if epoch > 0:
            bgnmf_model.fit(masked_data_IJ)
            
        A_IK = bgnmf_model.A_IK
        B_IK = bgnmf_model.B_IK
        H_KJ = bgnmf_model.H_KJ
        
    heldout_likelihood_N = st.beta.pdf(heldout_vals,
                                        A_IK.dot(H_KJ)[heldout_subs], 
                                        B_IK.dot(H_KJ)[heldout_subs])
    
    ppd = np.exp(np.mean(np.log(heldout_likelihood_N)))

    # Write the Model parameters
    state_name = f"uuid_{uuid}_state_bgnmf.npz"
    np.savez_compressed(os.path.join(output_path,state_name), A_IK=A_IK, B_IK=B_IK, H_KJ=H_KJ, ppd=ppd)
    
    # Log results
    mlflow.log_artifact(os.path.join(output_path,state_name))
    mlflow.end_run()
if __name__ == '__main__':
    main()