import os
import click
import numpy as np
import numpy.random as rn
import mlflow
import json

from bgnmf import BGNMF
from dncbtd import DNCBTD, initialize_DNCBTD_with_BGNMF

@click.command()
@click.argument('json_string')
def main(json_string, test = False, verbose=0):

    json_dict = json.loads(json_string)

    # if these params not in json_dict then throw an exception
    data_path = json_dict['data_path']
    output_path = json_dict['output_path']
    C = json_dict['C']
    K = json_dict['K']
    mask_seed = json_dict['mask_seed']
    model_seed = json_dict['model_seed']
    uuid = json_dict['uuid']

    data_dict = np.load(data_path)
    data_IJ = np.ascontiguousarray(data_dict['Beta_IJ'])
    I,J = data_IJ.shape

    # Set experiment for run
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", default="heldout_experiment")
    experiment_id = mlflow.create_experiment(experiment_name, artifact_location=output_path)

    # Log parameters C, K
    mlflow.start_run(experiment_id = experiment_id)
    mlflow.log_params({"C": C, 
                       "K": K,
                       "data": data_path,
                       "mask_seed": mask_seed,
                       "model_seed": model_seed,
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
    dncbtd_param_names = set(['C','K','bm','bu','pi_prior','phi_prior','theta_prior','shp_delta','rte_delta','shp_theta',
                     'shp_phi','shp_pi','debug','seed','n_threads'])
    dncbtd_param_dict = {x:json_dict[x] for x in json_dict
                              if x in dncbtd_param_names}
    prbf_model = DNCBTD(I=I, J=J, **dncbtd_param_dict)

    # Initialize the Model
    initialize_DNCBTD_with_BGNMF(prbf_model, masked_data_IJ,  verbose=verbose, n_itns=5)

    # Create the output path
    #samples_path = os.path.join(output_path, 'samples')
    #os.makedirs(samples_path, exist_ok = True)

    n_samples = 0
    avg_heldout_likelihood_N = np.zeros_like(heldout_vals)

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
            prbf_model.fit(data_IJ = masked_data_IJ, 
                           n_itns=n_itns if epoch > 1 else n_burnin,
                           verbose=verbose,
                           initialize=False,
                           schedule={},
                           fix_state={}
                           )
        heldout_likelihood_N = prbf_model.marginal_likelihood(subs=heldout_subs,
                                                         missing_vals=heldout_vals)
        if epoch > 0:
            avg_heldout_likelihood_N += heldout_likelihood_N
            n_samples += 1
            
            # curr_pppd = np.exp(np.log(avg_heldout_likelihood_N / n_samples).mean())
        
        state = dict(prbf_model.get_state())
        state['heldout_ppd'] = np.exp(np.log(heldout_likelihood_N).mean())
        state['heldout_likelihood_N'] = heldout_likelihood_N

    # Write the Model parameters
    state_name = f"uuid_{uuid}_state_{prbf_model.total_itns}.npz"
    # np.savez_compressed(os.path.join(output_path,state_name), **state)
    
    # Log results
    mlflow.log_artifact(os.path.join(output_path,state_name))
    mlflow.end_run()
if __name__ == '__main__':
    main()