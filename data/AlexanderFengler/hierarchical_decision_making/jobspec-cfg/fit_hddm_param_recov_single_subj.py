import hddm
import argparse
import uuid
import pickle
import os

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--model",
                    type = str,
                    default = 'ddm_par2_no_bias')
    CLI.add_argument("--n_trials_per_subject",
                    type = int,
                    default = 800)
    CLI.add_argument("--nmcmc",
                     type = int,
                     default = 4000)
    CLI.add_argument("--nburn",
                    type = int,
                    default = 1000)
    CLI.add_argument("--nchains",
                    type = int,
                    default = 2)
    CLI.add_argument("--out_folder",
                    type = str,
                    default = "data/param_recov/single_subj/")
    
    # Process supplied arguments:
    args = CLI.parse_args()

    print('Loaded argument: ')
    print(args)

    # Set fixed hexadecimal code for our model
    model_id =  uuid.uuid1().hex

    # Generate simulated data:
    data, full_parameter_dict = hddm.simulators.hddm_dataset_generators.simulator_h_c(data = None,
                                                                                      n_subjects = 1,
                                                                                      n_trials_per_subject = args.n_trials_per_subject,
                                                                                      model = args.model,
                                                                                      p_outlier = 0.00,
                                                                                      conditions = None,
                                                                                      depends_on = None,
                                                                                      regression_models = None,
                                                                                      regression_covariates = None,
                                                                                      group_only_regressors = False,
                                                                                      group_only = None,
                                                                                      fixed_at_default = None) #['z'])

    # Check if target folder exists:
    # Check in two steps
    # 1
    if os.path.isdir(args.out_folder):
        pass
    else:
        os.mkdir(args.out_folder)
    
    # 2
    if os.path.isdir(args.out_folder + args.model):
        pass
    else:
        os.mkdir(args.out_folder + args.model)

    # Save simulated data
    save_data = {'data': data, 'param_dict': full_parameter_dict}
    pickle.dump(save_data, open(args.out_folder + args.model + '/' + \
                                'data_{}_uuid_{}.pickle'.format(str(args.model), model_id), 'wb'))
    
    # Main loop across chains:
    for chain in range(args.nchains):

        # Specify HDDM model
        hddm_model_ = hddm.HDDMnn(data,
                                  model = args.model,
                                  informative = False,
                                  include = hddm.simulators.model_config[args.model]['hddm_include'],
                                  is_group_model = False,
                                  p_outlier = 0.00,
                                  network_type='torch_mlp')

        # Sample from model              
        hddm_model_.sample(args.nmcmc, 
                           burn = args.nburn, 
                           dbname = args.out_folder + args.model + '/' + \
                                    'db_{}_chain_{}_uuid_{}.db'.format(str(args.model), str(chain), model_id),
                           db = 'pickle')
        print("\n FINISHED SAMPLING CHAIN " + str(chain))
        
        # Store model
        hddm_model_.save(args.out_folder + args.model + '/' + \
                        'model_{}_chain_{}_uuid_{}.pickle'.format(str(args.model), str(chain), model_id))
        
        print("\n FINISHED FITTING HDDM MODEL CHAIN " + str(chain))

    print("\n FINISHED ALL CHAINS")