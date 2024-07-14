# Single image super resolution using GANs

This is a python framwork for training, evaluating and comparing GAN models with different loss functions and hyperparameters. 

The goal of the repo and framework is to be able to reproduce the results and methods presented in SISR papers such as SRGAN and Best-Buddy GAN, and to try out how different loss functions affect a GANs performance.
***

## Usage

### Initialization
1. Clone the repo
```bash
git clone https://github.com/SebastianBitsch/SRGAN-ST.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Pepare data
    1. Training data\
    Download a training dataset containing high-res images. We recommend either the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) dataset for comparable results. Place the dataset in /data/
    
    2. Evaluation data\
    Download one or more validation datasets. Set5, Set14, Urban100 or BSDS100 are all good options. Place the dataset(s) in /data/

    3. Make train dataset\
    Create a dataset of smaller equal sized images from the given training dataset by running the following.

            python data-prep/prepare_dataset.py --input_dir=/data/original/ --output_dir=/data/train/ --output_size=96

        See ```prepare_dataset.py -h``` for more options.

    3. Config\
    Update ```config.py``` to reflect the locations of the dataset(s). In particular update ```DATA.TRAIN_GT_IMAGES_DIR```, ```DATA.TEST_GT_IMAGES_DIR``` and ```DATA.TEST_LR_IMAGES_DIR```

### Training
**Train RResnet**\
To train a resnet / warmup the generator in GAN, edit relevant training parameters in the ```config.py``` - or keep the default, and run:
    
    python warmup.py
    
**Train GAN**\
To train a GAN edit any relevant parameters in the ```config.py``` file. To initialize the generator with the weights after warmup, set ```MODEL.G_CONTINUE_FROM_WARMUP``` to true and set the path to the weights in ```MODEL.G_WARMUP_WEIGHTS```. To start training run:

    python train.py

    
### Evaluation
**Train RResnet**\
To evalute a trained model (GAN or resnet), set ```EXP.NAME``` in the config to the name of the model you want to evaluate, then run:

    python validate.py

See ```python validate.py -h``` for more options

***

### Acknowledgement
This repo is loosely based on the work of Github user Lornatangs [repo](https://github.com/Lornatang/SRGAN-PyTorch) for SISR using SRGAN.


### Notes
* Moduler: [link](https://www.hpc.dtu.dk/?page_id=282)
    - ```module load python3/3.10.7```
    - ```module load cuda/11.7```

* Launch tensorboard fra ssh i vsc:
    - ```tensorboard --logdir=tensorboard/ --host localhost --port 3000```
    - ```fuser -k 3000/tcp``` slå processen ned hvis den allerede kører

* Kopier filer til og fra hpc [link](https://www.gbar.dtu.dk/index.php/faq/78-home-directory)

* Find ud hvor meget space er brugt på hpc
    - ```getquota_zhome.sh```
    - ```getquota_work3.sh```
    - ```du -h --max-depth=1 --apparent $HOME```

* How to use venv in notebooks [link](https://anbasile.github.io/posts/2017-06-25-jupyter-venv/)
* Kopier filer fra hpc til pc
    - ```scp -r -i /Users/sebastianbitsch/.ssh/gbar s204163@transfer.gbar.dtu.dk:SRGAN-ST/samples/logs /Users/sebastianbitsch/Desktop/```
* Scratch ligger på ```/work3/```
