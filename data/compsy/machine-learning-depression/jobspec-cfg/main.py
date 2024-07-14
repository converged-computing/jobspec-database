#!/usr/bin/python
from learner.driver import Driver
import sys, getopt

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"ht:pnf")
    except getopt.GetoptError:
        print('python main.py -t <createset|train|evaluate>  -p (use polynomial features) -n (bust cache) -f (feature selection)')
        sys.exit(2)
    use_polynomial = False
    use_force_no_caching = False
    use_feature_selection = False
    used_type = None
    print(opts)
    for opt, arg in opts:
        if opt == '-h':
            print('python main.py -t <createset|train|evaluate>  -p (use polynomial features) -n (bust cache) -f (feature selection)')
            sys.exit()
        elif opt in ("-t"):
            if not arg in ('createset','train','evaluate'):
                print('Choose one of createset, train, evaluate')
                sys.exit(2)
            used_type = arg
        elif opt in ("-p"):
            use_polynomial = True
        elif opt in ("-n"):
            use_force_no_caching = True
        elif opt in ("-f"):
            use_feature_selection = True
        else:
            print('python main.py -t <createset|train|evaluate>  -p (use polynomial features) -n (bust cache) -f (feature selection)')
            assert False, "unhandled option"

    return {
            'used_type': used_type,
            'use_polynomial': use_polynomial,
            'use_force_no_caching': use_force_no_caching,
            'use_feature_selection': use_feature_selection
            }

if __name__ == '__main__':
    params = main(sys.argv[1:])
    print(params)
    d = Driver(verbosity=0,
            polynomial_features=params['use_polynomial'],
            normalize=False,
            scale=True,
            force_no_caching=params['use_force_no_caching'],
            feature_selection=params['use_feature_selection'])

    # Run the create dataset operation
    if params['used_type'] == 'createset':
        if not params['use_feature_selection'] and params['use_polynomial']:
            warning('You have not done feature selection, but you do use polynomial features. This is will take some time.')
        d.run_setcreator()

    # Run the evaluation operation
    elif params['used_type'] == 'evaluate':
        d.run_evaluator()

    # Run the training operation
    elif params['used_type'] == 'train':
        if params['use_polynomial'] or params['use_feature_selection'] or params['use_force_no_caching']:
           raise ValueError('The extra options supplied DONT work with training') 
        d.run_trainer()
    else:
        print('Specify -t and choose one of createset, train, evaluate')
        sys.exit(2)
