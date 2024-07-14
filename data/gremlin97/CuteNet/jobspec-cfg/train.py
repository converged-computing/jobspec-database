import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ControlFill.training.load_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

print("Running-1")

# Configs
resume_path = './ControlFill/models/control_sd15_ini.ckpt'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

print("Running-2")

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu() 
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(max_epochs=1, gpus=1, precision=32, callbacks=[logger])
# trainer = pl.Trainer(strategy="ddp", accelerator="gpu", devices=2, precision=16, callbacks=[logger])
checkpoint_callback = ModelCheckpoint(every_n_train_steps=5000, save_last=True, save_weights_only=False, filename='exp_sd21_{epoch:02d}_{step:06d}')

# Train!
trainer.fit(model, dataloader)