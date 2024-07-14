"""

Batch prediction with all datasets and all models

"""

import json
import os
from time import sleep
import subprocess

base_model_dir = '/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/Same_Training_Set_Data/Results/'

ensemble_g_details = {
    'architecture':'ensemble',
    'encoder':'resnet34',
    'encoder_weights':'imagenet',
    'active':'sigmoid',
    'target_type':'nonbinary',
    'in_channels':2,
    'ann_classes':'background,collagen'
}

ensemble_multi_details = {
    'architecture':'ensemble',
    'encoder':'resnet34',
    'encoder_weights':'imagenet',
    'active':'sigmoid',
    'target_type':'nonbinary',
    'in_channels':6,
    'ann_classes':'background,collagen'
}

concat_g_details = {
    'architecture':'Unet++',
    'encoder':'resnet34',
    'encoder_weights':'imagenet',
    'active':'sigmoid',
    'target_type':'nonbinary',
    'in_channels':2,
    'ann_classes':'background,collagen'
}

concat_multi_details = {
    'architecture':'Unet++',
    'encoder':'resnet34',
    'encoder_weights':'imagenet',
    'active':'sigmoid',
    'target_type':'nonbinary',
    'in_channels':6,
    'ann_classes':'background,collagen'
}

single_g_details = {
    'architecture':'Unet++',
    'encoder':'resnet34',
    'encoder_weights':'imagenet',
    'active':'sigmoid',
    'target_type':'nonbinary',
    'in_channels':1,
    'ann_classes':'background,collagen'
}

single_rgb_details = {
    'architecture':'Unet++',
    'encoder':'resnet34',
    'encoder_weights':'imagenet',
    'active':'sigmoid',
    'target_type':'nonbinary',
    'in_channels':3,
    'ann_classes':'background,collagen'
}

multi_mean_prep = {
    'image_size':'512,512,2',
    'mask_size':'512,512,1',
    'color_transform':'multi_input_mean'
}

multi_green_prep = {
    'image_size':'512,512,2',
    'mask_size':'512,512,1',
    'color_transform':'multi_input_green'
}

multi_rgb_prep = {
    'image_size':'512,512,6',
    'mask_size':'512,512,1',
    'color_transform':'None'
}

single_mean_prep = {
    'image_size':'512,512,1',
    'mask_size':'512,512,1',
    'color_transform':'mean'
}

single_green_prep = {
    'image_size':'512,512,1',
    'mask_size':'512,512,1',
    'color_transform':'green'
}

single_rgb_prep = {
    'image_size':'512,512,3',
    'mask_size':'512,512,1',
    'color_transform':'None'
}


model_dict_list = [
    {
        'model':'DEDU-ENRGB',
        'type':'multi',
        'tags':['Ensemble_RGB predictions'],
        'model_file':f'{base_model_dir}Ensemble_RGB/models/Collagen_Seg_Model_Latest.pth',
        'model_details': ensemble_multi_details,
        'preprocessing': multi_rgb_prep
    },
    {
        'model':'DEDU-ENRGBL',
        'type':'multi',
        'tags':['Ensemble_RGB_Long predictions'],
        'model_file':f'{base_model_dir}Ensemble_RGB_Long/models/Collagen_Seg_Model_Latest.pth',
        'model_details':ensemble_multi_details,
        'preprocessing':multi_rgb_prep
    },
    {
        'model':'DEDU-ENG',
        'type':'multi',
        'tags':['Ensemble_Green predictions'],
        'model_file':f'{base_model_dir}Ensemble_Green/models/Collagen_Seg_Model_Latest.pth',
        'model_details':ensemble_g_details,
        'preprocessing': multi_green_prep
    },
    {
        'model':'DEDU-ENGL',
        'type':'multi',
        'tags':['Ensemble_Green_Long predictions'],
        'model_file':f'{base_model_dir}Ensemble_Green_Long/models/Collagen_Seg_Model_Latest.pth',
        'model_details':ensemble_g_details,
        'preprocessing':multi_green_prep
    },
    {
        'model':'DEDU-ENM',
        'type':'multi',
        'tags':['Ensemble_Mean predictions'],
        'model_file':f'{base_model_dir}Ensemble_Mean/models/Collagen_Seg_Model_Latest.pth',
        'model_details': ensemble_g_details,
        'preprocessing': multi_mean_prep
    },
    {
        'model':'DEDU-ENML',
        'type':'multi',
        'tags':['Ensemble_Mean_Long predictions'],
        'model_file':f'{base_model_dir}Ensemble_Mean_Long/models/Collagen_Seg_Model_Latest.pth',
        'model_details':ensemble_g_details,
        'preprocessing': multi_mean_prep
    },
    {
        'model':'DEDU-MCRGB',
        'type':'multi',
        'tags':['Concatenated_RGB predictions'],
        'model_file':f'{base_model_dir}Concatenated_RGB/models/Collagen_Seg_Model_Latest.pth',
    },
    {
        'model':'DEDU-MCRGBL',
        'type':'multi',
        'tags':['Concatenated_RGB_Long predictions'],
        'model_file':f'{base_model_dir}Concatenated_RGB_Long/models/Collagen_Seg_Model_Latest.pth',
        'model_details':concat_multi_details,
        'preprocessing':multi_rgb_prep
    },
    {
        'model':'DEDU-MCG',
        'type':'multi',
        'tags':['Concatenated_Green predictions'],
        'model_file':f'{base_model_dir}Concatenated_Green/models/Collagen_Seg_Model_Latest.pth'
    },
    {
        'model':'DEDU-MCGL',
        'type':'multi',
        'tags':['Concatenated_Green_Long predictions'],
        'model_file':f'{base_model_dir}Concatenated_Green_Long/models/Collagen_Seg_Model_Latest.pth',
        'model_details':concat_g_details,
        'preprocessing':multi_green_prep
    },
    {
        'model':'DEDU-COM',
        'type':'multi',
        'tags':['Concatenated_Mean predictions'],
        'model_file':f'{base_model_dir}Concatenated_Mean/models/Collagen_Seg_Model_Latest.pth',
        'model_details':concat_g_details,
        'preprocessing': multi_mean_prep
    },
    {
        'model':'DEDU-COML',
        'type':'multi',
        'tags':['Concatenated_Mean_Long predictions'],
        'model_file':f'{base_model_dir}Concatenated_Mean_Long/models/Collagen_Seg_Model_Latest.pth',
        'model_details':concat_g_details,
        'preprocessing':multi_mean_prep
    },
    {
        'model':'DEDU-FRGB',
        'type':'single',
        'tags':['Fluorescence_RGB predictions'],
        'model_file':f'{base_model_dir}Fluorescence_RGB/models/Collagen_Seg_Model_Latest.pth'
    },
    {
        'model':'DEDU-FRGBL',
        'type':'single',
        'tags':['Fluorescence_RGB_Long predictions'],
        'model_file':f'{base_model_dir}Fluorescence_RGB_Long/models/Collagen_Seg_Model_Latest.pth',
        'model_details':single_rgb_details,
        'preprocessing':single_rgb_prep
    },
    {
        'model':'DEDU-FG',
        'type':'single',
        'tags':['Fluorescence_Green predictions'],
        'model_file':f'{base_model_dir}Fluorescence_Green/models/Collagen_Seg_Model_Latest.pth'
    },
    {
        'model':'DEDU-FGL',
        'type':'single',
        'tags':['Fluorescence_Green_Long predictions'],
        'model_file':f'{base_model_dir}Fluorescence_Green_Long/models/Collagen_Seg_Model_Latest.pth',
        'model_details':single_g_details,
        'preprocessing':single_green_prep
    },
    {
        'model':'DEDU-FM',
        'type':'single',
        'tags':['Fluorescence_Mean predictions'],
        'model_file':f'{base_model_dir}Fluorescence_Mean/models/Collagen_Seg_Model_Latest.pth',
        'model_details':single_g_details,
        'preprocessing':single_mean_prep
    },
    {
        'model':'DEDU-FML',
        'type':'single',
        'tags':['Fluorescence_Mean_Long predictions'],
        'model_file':f'{base_model_dir}Fluorescence_Mean_Long/models/Collagen_Seg_Model_Latest.pth',
        'model_details':single_g_details,
        'preprocessing':single_mean_prep
    },
    {
        'model':'DEDU-BFRGB',
        'type':'single',
        'tags':['Brightfield_RGB predictions'],
        'model_file':f'{base_model_dir}Brightfield_RGB/models/Collagen_Seg_Model_Latest.pth'
    },
    {
        'model':'DEDU-BFRGBL',
        'type':'single',
        'tags':['Brightfield_RGB_Long predictions'],
        'model_file':f'{base_model_dir}Brightfield_RGB_Long/models/Collagen_Seg_Model_Latest.pth',
        'model_details': single_rgb_details,
        'preprocessing': single_rgb_prep
    },
    {
        'model':'DEDU-BFG',
        'type':'single',
        'tags':['Brightfield_Green predictions'],
        'model_file':f'{base_model_dir}Brightfield_Green/models/Collagen_Seg_Model_Latest.pth'
    },
    {
        'model':'DEDU-BFGL',
        'type':'single',
        'tags':['Brightfield_Green_Long predictions'],
        'model_file':f'{base_model_dir}Brightfield_Green_Long/models/Collagen_Seg_Model_Latest.pth',
        'model_details':single_g_details,
        'preprocessing':single_green_prep
    },
    {
        'model':'DEDU-BFM',
        'type':'single',
        'tags':['Brightfield_Mean predictions'],
        'model_file':f'{base_model_dir}Brightfield_Mean/models/Collagen_Seg_Model_Latest.pth',
        'model_details':single_g_details,
        'preprocessing':single_mean_prep
    },
    {
        'model':'DEDU-BFML',
        'type':'single',
        'tags':['Brightfield_Mean_Long predictions'],
        'model_file':f'{base_model_dir}Brightfield_Mean_Long/models/Collagen_Seg_Model_Latest.pth',
        'model_details':single_g_details,
        'preprocessing':single_mean_prep
    }
]

base_data_dir = '/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/020524_DUET_Patches/'
dataset_list = [i for i in os.listdir(base_data_dir) if os.path.isdir(base_data_dir+i)]
#dataset_list = ['24H Part 1']

model_dict_list = [model_dict_list[0]]

f_dir = 'F'
bf_dir = 'B'
out_dir = 'Results'

# Setting to false to go through again (see check_image_bytes() in CollagenSegMain to adjust which images are predicted on)
skip_duplicates = False
print('-----------------------------------------------------------------')
print(f'Iterating through {len(model_dict_list)} models on {len(dataset_list)} datasets')
print('-----------------------------------------------------------')

test_inputs = {
    "input_parameters":{
        "phase":"test",
        "type":"",
        "image_dir":{},
        "output_dir":"",
        "model":"",
        "model_file":"",
        "neptune":{
            "project":"samborder/Deep-DUET",
            "source_files":["*.py","**/*.py"],
            "tags":[]
        }
    }
}

non_neptune_inputs = {
    "input_parameters":{
        "phase":"test",
        "type":"",
        "image_dir":{},
        "output_dir":"",
        "model":"",
        "model_file":"",
        "model_details":{},
        "preprocessing":{},
        "skip_duplicates": skip_duplicates
    }
}

inputs_file_path = './batch_inputs/test_inputs.json'

if not os.path.exists('./batch_inputs/'):
    os.makedirs('./batch_inputs/')

count = 0

for dataset in dataset_list:
    for model in model_dict_list:
        print(model)

        if 'model_details' not in model:
            test_iter_inputs = test_inputs.copy()
        else:
            test_iter_inputs = non_neptune_inputs.copy()

        # Generating new test_inputs
        test_iter_inputs['input_parameters']['type'] = model['type']
        if model['type']=='multi':
            test_iter_inputs['input_parameters']['image_dir'] = {
                "DUET":f'{base_data_dir}{dataset}/{f_dir}/',
                'Brightfield':f'{base_data_dir}{dataset}/{bf_dir}/'
            }
        elif model['type']=='single':
            # Need to check if this is a BF or F
            check_inputs = model['model'].split('-')[-1]
            # check_inputs will be either 'BFRGB', 'BFRGBL', 'BFG', 'BFGL', 'BFM', 'BFML', 'FRGB', 'FRGBL', 'FG', 'FGL', 'FM', 'FML'
            if check_inputs in ['BFG','BFGL','BFRGB','BFRGBL','BFM','BFML']:
                test_iter_inputs['input_parameters']['image_dir'] = {
                    'Brightfield':f'{base_data_dir}{dataset}/{bf_dir}/'
                }
            elif check_inputs in ['FG','FGL','FRGB','FRGBL','FM','FML']:
                test_iter_inputs['input_parameters']['image_dir'] = {
                    'DUET':f'{base_data_dir}{dataset}/{f_dir}/'
                }
        
        output_dir = f'{base_data_dir}{dataset}/{out_dir}/{model["tags"][0].split(" ")[0]}/'
        test_iter_inputs['input_parameters']['output_dir'] = output_dir
        test_iter_inputs['input_parameters']['model'] = model['model']
        test_iter_inputs['input_parameters']['model_file'] = model['model_file']

        if 'model_details' not in model:
            test_iter_inputs['input_parameters']['neptune']['tags'] = model['tags'][0].replace(' ',f' {dataset} ')
        else:
            test_iter_inputs['input_parameters']['model_details'] = model['model_details']
            test_iter_inputs['input_parameters']['preprocessing'] = model['preprocessing']

        with open(inputs_file_path.replace('.json',f'{count}.json'),'w') as f:
            json.dump(test_iter_inputs,f,ensure_ascii=False)
            f.close()
        
        #if not os.path.exists(output_dir):
        process = subprocess.Popen(['python3', 'Collagen_Segmentation/CollagenSegMain.py', f'./batch_inputs/test_inputs{count}.json'])
        process.wait()

        exit_code = process.returncode
        print(f'Return code of process was: {exit_code}')
        #else:
        #    print('Already run, skipping')

        count+=1


