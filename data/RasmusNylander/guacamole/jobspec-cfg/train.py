if __name__ == "__main__":
	print("importing stuff...\n")

import os
from datetime import datetime
from typing import Optional

import torch
import torchvision
from torch import Tensor
from tqdm import tqdm, trange

import dataloader
import metrics
import networks
from networks import Architecture

import argparse
import sys
from assert_gpu import assert_gpu
device = assert_gpu()
from metrics import Metrics, get_metrics
from dataclasses import dataclass
from dataloader import TACO

@dataclass
class ClassificationEvalResult:
	loss: float
	precision: float
	recall: float
	f1: float
	accuracy: float

def evaluate_classification(
		model: torch.nn.Module,
		test_loader: torch.utils.data.DataLoader,
		loss_function,
		metrics: Metrics,
) -> ClassificationEvalResult:
	model.eval()
	with torch.no_grad():
		loss = torch.empty(len(test_loader))

		for batch_number, (data, target) in enumerate(tqdm(test_loader, leave=False, unit="batches", position=1, desc="Evaluating")):
			data, target = data.to(device).float(), target.to(device).float()

			output = model(data)
			loss[batch_number] = loss_function(output, target).cpu().item()
			prediction = torch.argmax(output, dim=1)
			prediction = torch.nn.functional.one_hot(prediction, num_classes=len(TACO.LABELS)).float()
			for metric in metrics:
				metric(prediction, target)

	loss = loss.mean().item()
	precision, recall, f1, accuracy = [metric.compute().item() for metric in metrics]

	return ClassificationEvalResult(loss, precision, recall, f1, accuracy)

def save_model(state_dict: dict, name: str):
	if not os.path.exists('models'):
		os.makedirs('models')
	savepath = f"models/{name}"
	torch.save(state_dict, savepath)


def train(
		model: torch.nn.Module,
		optimiser: torch.optim.Optimizer,
		train_loader: torch.utils.data.DataLoader,
		validation_loader: torch.utils.data.DataLoader,
		loss_function,
		metrics: Metrics,
		num_epochs: int = 10,
) -> dict:
	best_model_state_dict, best_loss = None, float("inf")

	for epoch in trange(num_epochs, unit="Epoch", desc="Training", position=0):
		model.train()

		train_loss_epoch = torch.empty((len(train_loader)), dtype=torch.float64)

		for batch_number, (data, target) in enumerate(tqdm(train_loader, leave=False, unit="batches", position=1)):
			data, target = data.to(device).float(), target.to(device).float()

			optimiser.zero_grad()

			output = model(data)

			loss = loss_function(output, target)

			loss.backward()

			optimiser.step()

			train_loss_epoch[batch_number] = loss.item()

		validation_result = evaluate_classification(model, validation_loader, loss_function, metrics)
		if validation_result.loss < best_loss:
			best_loss = validation_result.loss
			state_dict_extractor = model.__class__()  # We do it this way because it's the easiest way to get the state dict to the cpu
			state_dict_extractor.load_state_dict(model.state_dict())
			best_model_state_dict = state_dict_extractor.state_dict()
			tqdm.write(f"Saving new best model")
			save_model(best_model_state_dict, name)

		if with_logging:
			wandb.log({
						"train loss": train_loss_epoch.mean().item(),
						"validation loss": validation_result.loss,
						"validation precision": validation_result.precision,
						"validation recall": validation_result.recall,
						"validation f1": validation_result.f1,
						"validation accuracy": validation_result.accuracy,
					  })

		tqdm.write(
			f" Train loss: {round(train_loss_epoch.mean().item(), 3)}, Validation loss {round(validation_result.loss, 3)}")

	return best_model_state_dict if best_model_state_dict is not None else model.state_dict()


def run(
		name: str,
		architecture: networks.Architecture,
		loss_function,
		num_epochs: int = 100,
		batch_size: int = 64,
		lr: float = 0.0001,
		datapath_override: Optional[str] = None,
):
	print('reading data...\n')

	train_loader, val_loader, test_loader = dataloader.make_dataloader(
		batch_size=batch_size,
		dataset_path_override=datapath_override,
	)

	metrics = get_metrics()
	metrics = [metric.to(device) for metric in metrics]

	print('initializing network...\n')
	model = architecture.create_network()
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	if with_logging:
		print("initializing wandb...")
		wandb.init(
			# set the wandb project where this run will be logged
			project="guacamole",
			name=name,
			# track hyperparameters and run metadata
			config={
				"architecture": architecture.value,
				"epochs": num_epochs,
				"batch_size": batch_size,
				"lr": lr,
			}
		)

		print(wandb.run.name)
		summary(model, (3, 64, 64))

	print('training... \n')

	best_model_state = train(model, optimizer, train_loader, val_loader, loss_function, num_epochs=num_epochs, metrics=metrics)
	model.load_state_dict(best_model_state)

	print(f"saving model to {savepath}")
	save_model(best_model_state, name)

	test_result = evaluate(model, test_loader, loss_function)
	wandb.log({
				  "test loss": test_result.loss,
			  })



if __name__ == "__main__":
	def convert_to_architecture_repr_or_identity(string: str) -> str:  # We want to use repr for the user input choices
		architecture = Architecture.from_string(string)
		return repr(architecture) if architecture is not None else string

	parser = argparse.ArgumentParser(description='Train a model on a dataset.')
	parser.add_argument('--name', type=str, help='Name of the run. If not specified, a timestamp will be used.')
	parser.add_argument('-m', '--model',
						type=convert_to_architecture_repr_or_identity,
						help='Model to use', required=True,
						choices=[repr(model) for model in Architecture])
	parser.add_argument('--data_path', type=str, help='Path to the dataset', default=None)

	parser.add_argument('--logging', action="store_true", help='Whether to log to wandb')

	parser.add_argument('--epochs', type=int, help='Number of epochs', default=50)
	parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
	parser.add_argument('--lr', type=float, help='Learning rate', default=0.0001)

	# parser.add_argument('--loss', type=str, help='Loss function to use', required=True,
	# 					choices=[loss.name for loss in metrics.Loss])
	# parser.add_argument('--optimizer', type=str, help='Optimizer to use', required=True, choices=["adam"])
	# parser.add_argument('-a', '--augmentation', type=str, help='Data augmentation to use to use', default=[], nargs="+",
	# 					choices=[aug for aug in data_augmentation.augmentations.keys()])

	loss = metrics.cross_entropy

	args = parser.parse_args()
	name = args.name
	if name is None:
		name = datetime.timestamp(datetime.now())
		print(f"No name provided, using timestamp '{name}'")

	architecture = Architecture.from_string(args.model)

	with_logging = args.logging
	if with_logging:
		import wandb
		from torchsummary import summary

	if args.epochs is None or args.epochs < 0:
		print("Invalid number of epochs. Number of epochs must be positive", file=sys.stderr)
		exit(1)
	num_epochs = args.epochs
	if args.batch_size is None or args.batch_size <= 0:
		print("Invalid batch size. Batch size must be strictly greater than 0", file=sys.stderr)
		exit(1)
	batch_size = args.batch_size
	if args.lr is None or args.lr <= 0:
		print("Invalid learning rate. Learning rate must be strictly greater than 0", file=sys.stderr)
		exit(1)
	lr = args.lr

	run(
		name=name,
		datapath_override=args.data_path,
		architecture=architecture,
		loss_function=loss,
		num_epochs=num_epochs,
		batch_size=batch_size,
		lr=lr,
	)

