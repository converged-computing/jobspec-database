## Tensorflow
import tensorflow as tf

print('-----------------------------------------')
print('List the Device info if GPU is available')
print(tf.config.list_physical_devices('GPU'))
print('-----------------------------------------')

## SRWGAN Modules
from GANcore.SRWGAN import SRWGAN
from model.Generator import Generator
from model.Discriminator import Discriminator
from model.metrics import d_srloss, g_srloss

## Data
from data.preprocess import TrainDatasetFromFolder

## Miscellaneous
from utils.CallBack import EpochVisualizer
import IPython.display
from GANcore.SRWGAN import save_model

import argparse
import pickle

# Argument Parsing
parser = argparse.ArgumentParser(description = 'SRWGAN in Tensorflow')
parser.add_argument('--pretrained', type = str, default = 'resnet50', help = 'Pretrained Model for the ContentLoss')
parser.add_argument('--datapath', type = str, default = './Datasets', help = 'Base Path for the Datasets')
parser.add_argument('--trainnum', type = int, default = 20, help = 'Number of Image Pairs for Training')
parser.add_argument('--epochs', type = int, default = 2, help = 'Number of Epochs')
parser.add_argument('--batchsz', type = int, default = 4, help = 'Batch Size')
parser.add_argument('--dstep', type = int, default = 1, help = 'Number of Train Steps for Discriminator Every Batch')
parser.add_argument('--gstep', type = int, default = 1, help = 'Number of Train Steps for Generator Every Batch')
parser.add_argument('--gpweight', type = float, default = 10.0, help = 'Coefficient for the Gradient-Penalty Term in Discriminator Loss')
parser.add_argument('--cweight', type = float, default = 1e-3, help = 'Coefficient for the Content Loss of Generator')
parser.add_argument('--savemodel', type = bool, default = False, help = 'Whether to Save the Model')
parser.add_argument('--chkpt_path', type = str, default = './SRWGAN', help = 'Path to Save or Load the Model')

args = parser.parse_args()


'''
Sample Code Snippets to Train and Save a GAN model
'''


'''
Initialize the Network
'''


srwgan_model = SRWGAN(
    dis_model = Discriminator(),
    gen_model = Generator(),
    name = 'srwgan',
    z_dims = [None, None],  # Not using random samples for Generator's Input in SRWGAN 
    # Default Values
    pretrained = args.pretrained,
    hyperimg_ids = [2, 7, 10, 14], # index of intermediate features from a pretrained model for calculating the Content Loss; 
                                   # *** Check model/SRWGAN.py for more information ***
    lr_shape = [64, 64, 3],     # Low-Res Images as the input for Generator
    hr_shape = [256, 256, 3]    # High-Res Images as the ground Truth; 
)                               # Super-Res Images as the output for Generator 


srwgan_model.compile(
    optimizers = {
        'd_opt' : tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5, beta_2 = 0.9),
        'g_opt' : tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5, beta_2 = 0.9),
    },
    losses = {
        'd_loss' : d_srloss,
        'g_loss' : g_srloss,
    },
)


'''
Load the Data for Super Resolution Task
'''
lres, hres = TrainDatasetFromFolder(dataset_dir = args.datapath)

# Centralize the data to [-1, 1] to for better training (and ofc to avoid overhead and overflow in memory)
lres = (lres - 127.5) / 127.5
hres = (hres - 127.5) / 127.5


'''
Prepare the Samples for Callback Visualization - Centralized + Normalized
'''
true_sample = hres[args.trainnum-2 : args.trainnum]                         ## 2 High Resolution images
fake_sample = srwgan_model.gen_model(lres[args.trainnum-2 : args.trainnum])   ## 2 Generated Super Resolution images
print('high-res samples shape:', true_sample.shape)
print('generatered super-res samples shape:', fake_sample.shape)

viz_callback = EpochVisualizer(srwgan_model, [true_sample, fake_sample])

# Train the Model
history = srwgan_model.fit(
    lres[:args.trainnum], hres[:args.trainnum],
    dis_steps = args.dstep,
    gen_steps = args.gstep,
    gp_weight = args.gpweight,
    content_weight = args.cweight,
    epochs = args.epochs,
    batch_size = args.batchsz,
    callbacks = [viz_callback]
    #return_dict = True
)


## Either Save the model/Visualizer
## or directly visualize the CallBack

if args.savemodel:
    save_model(srwgan_model.gen_model, args.chkpt_path + '_Gen')
    save_model(srwgan_model.crt_model, args.chkpt_path + '_Dis')
else:
    print('The model was not saved by default')



'''
Calculate PSNR for Sample Images - default calculating PSNR for 4 samples
'''
srwgan_model.psnr([ lres[args.trainnum-4 : args.trainnum], hres[args.trainnum-4 : args.trainnum] ])

'''
Visualizing the Results
'''
viz_callback.save_gif('./generated_samples/generatedSuperRes')
IPython.display.Image(open('./generated_samples/generatedSuperRes.gif', 'rb').read())

'''
Save the metrics history as a dictionary
'''
with open('./trainHistoryDict', 'wb') as f:
    pickle.dump(history.history, f)
