import hddm
import argparse
import uuid

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data_path",
                    type = str,
                    default = "data/chong_data_hddm_ready.csv")
    CLI.add_argument("--model",
                    type = str,
                    default = 'ddm_par2_no_bias')
    CLI.add_argument("--dep_on_task",
                    type = int,
                    default = 0)
    CLI.add_argument("--dep_on_coh",
                    type = int,
                    default = 0)
    CLI.add_argument("--is_group_model",
                    type = int,
                    default = 0)
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
                    default = "data/hddm_models/")
    
    # Process supplied arguments:
    args = CLI.parse_args()

    print('Loaded argument: ')
    print(args)

    # Read in data:
    chong_data = hddm.load_csv(args.data_path)

    # Specify model depends_on arguments, as per supplied arguments ----
    depends_on = {'vh': [],'vl1': [],'vl2': []}

    # Should drifts depend on task?
    if args.dep_on_task:
        depends_on['vh'].append('highDim')
        depends_on['vl1'].append('irrDim')
        depends_on['vl2'].append('lowDim')
    
    # Should drifts depend on coherence levels?
    if args.dep_on_coh:
            depends_on['vh'].append('highDimCoh')
            depends_on['vl1'].append('irrDimCoh')
            depends_on['vl2'].append('lowDimCoh')
    
    # In case no dependencies specified --> set to None
    if len(depends_on['vh'])==0:
        depends_on = None
    # --------------------------------------------------------------------

    # Main loop across chains:
    # model_id =  uuid.uuid1().hex

    for chain in range(args.nchains):

        # Specify HDDM model
        #depends_on = {'vh': ['highDimCoh','highDim'], 'vl1': ['irrDimCoh','irrDim'], 'vl2': ['lowDimCoh','lowDim']}
        hddm_model_ = hddm.HDDMnn(chong_data,
                                  model = args.model,
                                  informative = False,
                                  include = hddm.simulators.model_config[args.model]['hddm_include'],
                                  is_group_model = bool(args.is_group_model),
                                  depends_on = depends_on,
                                  p_outlier = 0.05,
                                  network_type='torch_mlp')

        # Sample from model              
        hddm_model_.sample(args.nmcmc, 
                           burn = args.nburn, 
                           dbname = args.out_folder + \
                                    '{}_chong_task_{}_coh_{}_group_{}_chain_{}.db'.format(str(args.model),
                                                                                          str(args.dep_on_task),
                                                                                          str(args.dep_on_coh), 
                                                                                          str(args.is_group_model),
                                                                                          str(chain)),
                           db = 'pickle')
        print("\n FINISHED SAMPLING CHAIN " + str(chain))
        
        # Store model
        hddm_model_.save(args.out_folder + \
                        '{}_chong_task_{}_coh_{}_group_{}_chain_{}.pickle'.format(str(args.model),
                                                                                  str(args.dep_on_task),
                                                                                  str(args.dep_on_coh),
                                                                                  str(args.is_group_model),
                                                                                  str(chain)))
        print("\n FINISHED FITTING HDDM MODEL CHAIN " + str(chain))

    print("\n FINISHED ALL CHAINS")