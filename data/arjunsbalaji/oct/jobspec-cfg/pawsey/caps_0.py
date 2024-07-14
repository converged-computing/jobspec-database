# default_exp runs

import sys

sys.path.append('/workspace/oct_ca_seg/oct/')

from oct.startup import *
from model import CapsNet
import numpy as np
import mlflow
from fastai.vision import *
import mlflow.pytorch as MLPY
from fastai.utils.mem import gpu_mem_get_all

gpu_mem_get_all()

### Configuration Setup

name = 'DEEPCAP_09start_10e_001lr_fsmooth_deconv_in0'

config_dict = loadConfigJSONToDict('configCAPS_APPresnet18.json')
config_dict['LEARNER']['lr']= 0.001
config_dict['LEARNER']['bs'] = 24
config_dict['LEARNER']['pct_start'] = 0.9
config_dict['LEARNER']['epochs'] = 10
config_dict['LEARNER']['runsave_dir'] = '/workspace/oct_ca_seg/runsaves/'
config_dict['MODEL']['uptype'] = 'deconv'
config_dict['MODEL']['input_images'] = [0]
config_dict['MODEL']['inputchannels'] = len(config_dict['MODEL']['input_images'])
config_dict['MODEL']['maps1'] = 4
config_dict['MODEL']['dims1'] = 8
config_dict['MODEL']['maps2'] = 8
config_dict['MODEL']['dims2'] = 16
config_dict['MODEL']['maps3'] = 16
config_dict['MODEL']['dims3'] = 32
config = DeepConfig(config_dict)

metrics = [sens, spec, dice, my_Dice_Loss, acc]

def saveConfigRun(dictiontary, run_dir, name):
    with open(run_dir/name, 'w') as file:
        json.dump(dictiontary, file)

## Dataset

cocodata_path = Path('/workspace/oct_ca_seg/COCOdata/')
train_path = cocodata_path/'train/images'
valid_path = cocodata_path/'valid/images'
test_path = cocodata_path/'test/images'

#for input image selection

print('config done')
def _in_delta(x):
    return x[config.MODEL.input_images,:,:]
in_delta = TfmPixel(_in_delta, order=0)
in_delta.use_on_y = False

### For complete dataset

fn_get_y = lambda image_name: Path(image_name).parent.parent/('labels/'+Path(image_name).name)
codes = np.loadtxt(cocodata_path/'codes.txt', dtype=str)
tfms = get_transforms()
tfms[0].append(in_delta())
tfms[1].append(in_delta())
src = (SegCustomItemList
       .from_folder(cocodata_path, recurse=True, extensions='.jpg')
       .filter_by_func(lambda fname: Path(fname).parent.name == 'images', )
       .split_by_folder('train', 'valid')
       .label_from_func(fn_get_y, classes=codes))
src.transform(tfms, tfm_y=True, size=config.LEARNER.img_size)
data = src.databunch(cocodata_path,
                     bs=config.LEARNER.bs,
                     val_bs=2*config.LEARNER.bs,
                     num_workers = config.LEARNER.num_workers)
#stats = [torch.tensor([0.2190, 0.1984, 0.1928]), torch.tensor([0.0645, 0.0473, 0.0434])]
stats = data.batch_stats()
data.normalize(stats);
data.c_in, data.c_out = config.MODEL.inputchannels, 2

### seg_model Caps
print('data done')
run_dir = config.LEARNER.runsave_dir+'/'+name
os.makedirs(run_dir, exist_ok=True)
exp_name = 'seg_model_caps'
mlflow_CB = partial(MLFlowTracker,
                    exp_name=exp_name,
                    uri='file:/workspace/oct_ca_seg/runsaves/fastai_experiments/mlruns/',
                    params=config.config_dict,
                    log_model=True,
                    nb_path="/workspace/oct_ca_seg/oct/02_caps.ipynb")
deepCap = CapsNet(config.MODEL).cuda()

#print(config_dict)
#print('debugging', data.one_batch()[0].shape, deepCap(data.one_batch()[0].cuda()), config.MODEL.input_images)

learner = Learner(data = data,
                  model = deepCap,
                  metrics = metrics,
                  callback_fns=mlflow_CB)
print('starting run')
with mlflow.start_run():
    learner.fit_one_cycle(config.LEARNER.epochs, slice(config.LEARNER.lr), pct_start=config.LEARNER.pct_start)
    MLPY.save_model(learner.model, run_dir+'/model')
    learner.save(Path(run_dir)/'learner')
    save_all_results(learner, run_dir, exp_name)
    saveConfigRun(config.config_dict, run_dir=Path(run_dir), name = name)
