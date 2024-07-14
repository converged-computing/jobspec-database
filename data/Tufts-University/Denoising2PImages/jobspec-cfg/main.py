import os as os
import sys
import json
import jsonschema
import pywt

# Local dependencies
import train
import basics
import eval

#################################################################################

def make_config(model_name):
    return {
        'cwd': '',
        'nadh_data': '',
        'fad_data': '',
        'epochs': 300,
        'steps_per_epoch': {'srgan': None,'rcan': None, 'care': 100, 'resnet':None, 'wunet':100, 'UnetRCAN': None}[model_name],
        'input_shape': [256, 256],
        'initial_learning_rate': 1e-5,
        # Adjusting flags for loading training set and test set from seperate or same files
        'val_seed': 0, # Controls the validation split
        'val_split': 4, # Controls how many stacks to include in the validation set
        'test_flag' : 1, # Controls if a test set is generated or already exist (1: I have a test set 0:I need a test set)
        'train_mode': 0, # Controls if we want to load a test set after training or use all data for evaluation only
        'ssim_FSize': 11, # SSIM Filter Size
        'ssim_FSig': 1.5, # SSIM Filter Sigma 
        'batch_size': 50, # Default batch size
        # Metrics
        'loss': {'srgan': 'mse', 'care': 'ssimr2_loss', 'rcan': 'ssiml1_loss', 'resnet':'mse', 'wunet': 'ssimr2_loss','UnetRCAN':'mse'}[model_name],
        'metrics': ['psnr', 'ssim'],

        # Metric hyperparameters 
        'loss_alpha': 0.5, # How much different loss functions are weighted in the compound loss.

        # RCAN and Resnet config
        'num_channels': 32,
        'num_residual_blocks': 5,
        'num_residual_groups': 5,
        'channel_reduction': 4,

        # Unet config
        'unet_n_depth': 6,
        'unet_n_first': 32, 
        'unet_kern_size': 3,

        # Wavelet config
        'wavelet_function': '', # One of pywt.wavelist() or empty for non-wavelet.
    }


def apply_config_flags(config_flags, config):
    for config_flag in config_flags:
        components = config_flag.split('=')
        if len(components) != 2:
            raise ValueError(
                f'Invalid config flag: "{config_flag}"; expected key="string" or key=number')

        key, raw_value = components
        if key not in config:
            raise ValueError(
                f'Invalid config flag: "{config_flag}"; key "{key}" not found in config.')

        try:
            value = int(raw_value)
        except:
            try:
                value = float(raw_value)
            except:
                if raw_value == 'True':
                    value = True
                elif raw_value == 'False':
                    value = False
                else:
                    value = raw_value

        config[key] = value

    return config

def json_config(config):
    p = pywt.wavelist(kind='discrete')
    p.append('')
    schema = {
    'type': 'object',
    'properties': {
        'mode': {'type': 'string', 'enum': ['train','eval']},
        'model_name': {'type': 'string', 'enum': ['rcan', 'care', 'srgan', 'resnet', 'wunet', 'UnetRCAN']},
        'trial_name': {'type': 'string'},
        'cwd': {'type': 'string'},
        'nadh_data': {'type': 'string'},
        'fad_data': {'type': 'string'},
        'input_shape': {
            'type': 'array',
            'items': {'type': 'integer', 'minimum': 1},
            'minItems': 2,
            'maxItems': 3
            },
        'batch_size': {'type': 'integer', 'minimum': 1},
        'num_channels': {'type': 'integer', 'minimum': 1},
        'num_residual_blocks': {'type': 'integer', 'minimum': 1},
        'num_residual_groups': {'type': 'integer', 'minimum': 1},
        'channel_reduction': {'type': 'integer', 'minimum': 1},
        'epochs': {'type': 'integer', 'minimum': 1},
        'steps_per_epoch': {'type': ['integer', 'null'], 'minimum': 1},
        'val_seed': {'type':'integer'},
        'val_split': {'type':'integer', 'minimum': 1},
        'test_flag': {'type':['integer','boolean']},
        'train_mode': {'type':['integer','boolean']},
        'ssim_FSize': {'type': 'number', 'minimum': 1},
        'ssim_FSig': {'type': 'number', 'minimum': 0.1},
        'initial_learning_rate': {'type': 'number', 'minimum': 1e-6},
        'loss': {'type': 'string', 'enum': ['mae', 'mse', 'ssiml1_loss', 'ssiml2_loss', 'ssim_loss',
            'MSSSIM_loss','pcc_loss','ssimpcc_loss','ffloss','SSIMFFL','ssimr2_loss']},
        'metrics': {
            'type': 'array',
            'items': {'type': 'string', 'enum': ['psnr', 'ssim', 'pcc']}
            },
        'loss_alpha':{'type': 'number', 'minimum': 0, 'maximum':1},
        'unet_n_depth': {'type':'integer', 'minimum': 1},
        'unet_n_first':  {'type': 'integer', 'minimum': 1}, 
        'unet_kern_size': {'type': 'integer', 'minimum': 1}, 
        'wavelet_function': {'type': 'string','enum':p},
        },
    'additionalProperties': False,
    'anyOf': [
        {'required': ['mode']},
        {'required': ['model_name']},
        {'required': ['trial_name']}
        ]
    }

    jsonschema.validate(config, schema)

    config.setdefault('cwd', '')
    config.setdefault('nadh_data', '')
    config.setdefault('fad_data', '')
    config.setdefault('epochs', 300)
    config.setdefault('steps_per_epoch', {'srgan': None,'rcan': None, 'care': 100, 'resnet':None, 'wunet': 100, 'UnetRCAN': None}[config['model_name']])
    config.setdefault('input_shape', [256,256])
    config.setdefault('initial_learning_rate', 1e-5)
    config.setdefault('val_seed', 0)
    config.setdefault('val_split', 4)
    config.setdefault('test_flag', 1)
    config.setdefault('train_mode', 0)
    config.setdefault('ssim_FSize', 11)
    config.setdefault('ssim_FSig', 1.5)
    config.setdefault('batch_size', 50)
    config.setdefault('loss', {'srgan': 'mse', 'care': 'ssiml2_loss', 'rcan': 'ssiml1_loss', 'resnet':'mse','wunet': 'ssimr2_loss','UnetRCAN': 'mse'}[config['model_name']])
    config.setdefault('metrics', ['psnr','ssim'])
    config.setdefault('loss_alpha', 0.5)
    config.setdefault('num_channels', 32)
    config.setdefault('num_residual_blocks', 5)
    config.setdefault('num_residual_groups', 5)
    config.setdefault('channel_reduction', 4)
    config.setdefault('unet_n_depth', 6)
    config.setdefault('unet_n_first', 32)
    config.setdefault('unet_kern_size', 3)
    config.setdefault('wavelet_function', '')
    return config

def main():
    if ".json" in sys.argv[1]:
        print('Reading config from json file')
        with open(sys.argv[1]) as json_data:
            config = json.load(json_data)
        config = json_config(config)
        if len(sys.argv) > 1:
            config_flags = sys.argv[2:]
            config = apply_config_flags(config_flags, config)
        mode = config['mode']
        model_name = config['model_name']
        trial_name = config['trial_name']
        print(f'Using trial name: "{trial_name}"')
    else:
        if len(sys.argv) < 4:
            print('Usage: python main.py <mode: train | eval> <name: rcan | care | srgan | resnet | wunet | UnetRCAN> <trial_name> <config options...>')
            raise Exception('Invalid arguments.')

        # === Get arguments ===

        # We get the arguments in the form:
        # ['main.py', mode, model_name, config_options...]

        mode = sys.argv[1]
        if mode not in ['train', 'eval']:
            raise Exception(f'Invalid mode: "{mode}"')

        model_name = sys.argv[2]
        if model_name not in ['rcan', 'care', 'srgan', 'resnet', 'wunet', 'UnetRCAN']:
            raise Exception(f'Invalid model name: "{model_name}"')

        trial_name = sys.argv[3]
        print(f'Using trial name: "{trial_name}"')

        config_flags = sys.argv[4:] if len(sys.argv) > 4 else []
        config = make_config(model_name)
        config = apply_config_flags(config_flags, config)
        config['mode'] = mode
    print(f'Using config: {config}\n')

    main_path = config['cwd']
    if main_path == '':
        raise Exception(
            'Please set the "cwd" config flag. To use the current directory use: cwd=.')
    elif not os.path.isdir(main_path):
        raise Exception(
            f'Could not find current working directory (cwd): "{main_path}"')

    nadh_data_path = config['nadh_data']
    fad_data_path = config['fad_data']
    if nadh_data_path == '' and fad_data_path == '':
        raise Exception(
            'Please at least one of the two data-path flags "nadh_data" or "fad_data" config flag to specify where the data is in relation to the current directory.')

    # === Get right paths ===

    print(f'Changing to directory: {main_path}')
    os.chdir(main_path)
    print(f'Current directory: {os.getcwd()}')
    # Check data paths exist.
    if nadh_data_path != "" and not os.path.isfile(nadh_data_path):
        raise Exception(
            f'Could not find file at NADH data path: "{nadh_data_path}"')
    if fad_data_path != "" and not os.path.isfile(fad_data_path):
        raise Exception(
            f'Could not find file at FAD data path: "{fad_data_path}"')

    if not os.path.exists(os.getcwd() + '/'+ trial_name):
        os.mkdir(os.getcwd() + '/'+ trial_name)
    model_save_path = os.getcwd() + '/'+ trial_name

    # === Send out jobs ===

    basics.print_device_info()

    if mode == 'train':
        data_path = None
        if nadh_data_path != '' and fad_data_path == '':
            print(f'Using NADH data at: {nadh_data_path}')
            data_path = nadh_data_path
        elif fad_data_path != '' and nadh_data_path == '':
            print(f'Using FAD data at: {fad_data_path}')
            data_path = fad_data_path
        else:
            raise Exception(
                'Train expects just one data set; either "nadh_data" or "fad_data".')

        print('Running in "train" mode.\n')

        train.train(model_name,
                    config=config,
                    output_dir=model_save_path,
                    data_path=data_path)

        print('Successfully completed training.')
    elif mode == 'eval':
        if nadh_data_path != '':
            print(f'Using NADH data at: {nadh_data_path}')
        if fad_data_path != '':
            print(f'Using FAD data at: {fad_data_path}')

        print('Running in "eval" mode.\n')

        eval.eval(model_name,
                trial_name=trial_name,
                config=config,
                output_dir=model_save_path,
                # The above code checks that at least one is not empty.
                nadh_path=nadh_data_path if nadh_data_path != '' else None,
                fad_path=fad_data_path if fad_data_path != '' else None)

        print('Successfully completed evaluation.')

try:
    main()
except Exception as e:
    sys.stderr.write(f'Failed with error: {e}')
    raise e
