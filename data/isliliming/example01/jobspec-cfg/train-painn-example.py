import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import numpy as np

import argparse
####### Parser arguments
parser = argparse.ArgumentParser(description='PAiNN for hamiltonian and diabatization')
parser.add_argument('--cutoff', help='Set cutoff radius, default = 5', type=float, default=5.)
parser.add_argument('--features', help='Number of features, default = 32', type=int, default=32)
parser.add_argument('--max_epochs', help='Number of maximum epochs, default = 100', type=int, default=100)
#parser.add_argument('--interactions', help='Number of interactions, default = 3', type=int, default=3)
parser.add_argument('--layer' , help="Set number of layers, default = 3", type=int, default=3)
parser.add_argument('--split' , help="Which split to calculate, default = 0", type=int, default=0)

args = parser.parse_args()

max_epochs = args.max_epochs
n_atom_basis=args.features
set_interactions={32:3,64:4,128:5,256:6,512:7,1024:8}
interactions=set_interactions[n_atom_basis]
cutoff=args.cutoff
layer=args.layer
nsplit=args.split
print("Features: ",n_atom_basis)
print("Interactions: ",interactions)
print("Layers: ",layer) 
print("Max epochs: ", max_epochs)  
print("Cutoff: ",cutoff)
print("Processing Split ",nsplit)

use_schnet=False


forcetut = "./split{}".format(nsplit)
if not os.path.exists(forcetut):
    os.makedirs(forcetut)

torch.set_float32_matmul_precision('high')

seed=1
seed_everything(seed, workers=True)

data =spk.data.AtomsDataModule(
        os.path.join('/gpfs/home/chem/mstvtc/data/THz_orig_dataset_energy.db'),
        batch_size=10,
        distance_unit='Ang',
        property_units={'energy':'kcal/mol'},
        num_train=1752, # comment if split file
        num_val=438, # comment if split file
      #  split_file="/home/z/zkoczor/thz-painn-opt/hyperopt/split{}.npz".format(nsplit),
        transforms=[
        trn.ASENeighborList(cutoff=5.),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
        ],
        num_workers=16,
        pin_memory=True, # set to false, when not using a GPU
    )

data.prepare_data()
data.setup()


properties = data.dataset[0]
print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])

print('energy:\n', properties["energy"])
print('Shape:\n', properties["energy"].shape)

#cutoff = 5.
pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=50, cutoff=cutoff)
if use_schnet:
    repres = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=interactions,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )
else:
    repres =  spk.representation.PaiNN(
        n_atom_basis=n_atom_basis, n_interactions=interactions,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )
pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="energy",n_layers=layer)
#pred_forces = spk.atomistic.Forces(energy_key=MD17.energy, force_key=MD17.forces)

nnpot = spk.model.NeuralNetworkPotential(
    representation=repres,
    input_modules=[pairwise_distance],
    output_modules=[pred_energy],
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets("energy", add_mean=True, add_atomrefs=False)
    ]
)

output_energy = spk.task.ModelOutput(
    name="energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.0,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

#output_forces = spk.task.ModelOutput(
#    name=MD17.forces,
#    loss_fn=torch.nn.MSELoss(),
#    loss_weight=0.99,
#    metrics={
#        "MAE": torchmetrics.MeanAbsoluteError()
#    }
#)
task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_energy],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4},
    scheduler_cls=spk.train.ReduceLROnPlateau,
    scheduler_args={"patience": 15,
                    "cooldown": 10,
                    "min_lr": 1e-6,
                    "factor": 0.8,
                    "verbose": True},
    scheduler_monitor="val_loss"
)
logger = pl.loggers.TensorBoardLogger(save_dir=forcetut)
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(forcetut, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=forcetut,
    max_epochs=max_epochs, # for testing, we restrict the number of epochs
    devices=1,
    accelerator="gpu"
)
trainer.fit(task, datamodule=data)



from ase import Atoms
import ase.db


# set device
device = torch.device("cuda")

# load model
model_path = os.path.join(forcetut, "best_inference_model")
best_model = torch.load(model_path, map_location=device)

# set up converter
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=cutoff), dtype=torch.float32, device=device
)

# create atoms object from dataset
pred=[]
target=[]
target2=[]
#db=ase.db.connect(os.path.join(forcetut,'THz_orig_dataset_energy.db'))
db=ase.db.connect('/gpfs/home/chem/mstvtc/data/THz_orig_dataset_energy.db')
for i in range(546):
    idx=data.test_idx[i]
  #  print("idx ",idx)
    structure = data.test_dataset[i]
    atoms = Atoms(
    numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
    )

   # convert atoms to SchNetPack inputs and perform prediction
    inputs = converter(atoms)
    results = best_model(inputs)
    pred.append(results['energy'].cpu().detach().numpy())
# includes offset:    target.append(structure[spk.properties.energy].cpu().detach().numpy())
# original values:
    target2.append(db.get(int(idx)+1).data["energy"])

#print(pred,target)
print(np.mean(np.array(pred)), np.std(np.array(pred)))
#print(np.mean(np.array(target)), np.std(np.array(target)))
print("Mean predicted P: ",np.mean(np.array(target2)), " std: ",np.std(np.array(target2)))
print("MAE test set: ",np.mean(np.abs(np.array(target2)-np.array(pred))))
np.savez(forcetut+"/predictions.npz",pred)
np.savez(forcetut+"/reference.npz",target2)

