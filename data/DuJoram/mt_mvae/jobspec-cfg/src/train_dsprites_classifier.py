import argparse
import os
from datetime import datetime
from typing import List

import matplotlib as mpl
import numpy as np
import torch.cuda
import tqdm
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.datasets import DSpritesDataset
from experiments.mdsprites.networks import DSpritesImageClassifier


class DSpritesClassifierTrainer:
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        evaluation_frequency: int,
        checkpoint_frequency: int,
        num_workers: int,
        dsprites_archive_path: str,
        output_dir: str,
        checkpoints_dir: str,
        log_dir: str,
    ):
        self._epochs = epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate

        self._evaluation_frequency = evaluation_frequency
        self._checkpoint_frequency = checkpoint_frequency
        self._output_dir = output_dir
        self._checkpoint_dir = checkpoints_dir
        self._log_dir = log_dir

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._train_dataset = DSpritesDataset(
            dsprites_archive_path=dsprites_archive_path, train=True
        )

        self._test_dataset = DSpritesDataset(
            dsprites_archive_path=dsprites_archive_path, train=False
        )

        self._train_data_loader = DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True,
        )

        self._test_data_loader = DataLoader(
            self._test_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True,
        )

        self._classifier = DSpritesImageClassifier().to(self._device)

        self._optimizer = torch.optim.Adam(
            params=self._classifier.parameters(),
            lr=self._learning_rate,
        )

        self._losses = [CrossEntropyLoss() for _ in range(5)]

        self._writer = SummaryWriter(log_dir=self._log_dir)

    def run(self):

        for epoch in (tq := tqdm.trange(self._epochs)):
            epoch_loss = self._train(epoch)
            if epoch % self._evaluation_frequency == 0:
                self._eval(epoch)

            if epoch % self._checkpoint_frequency == 0:
                epoch_checkpoint_dir = os.path.join(
                    self._checkpoint_dir, f"{epoch:05d}"
                )
                os.makedirs(epoch_checkpoint_dir)
                model_save = os.path.join(epoch_checkpoint_dir, "dsprites_classifier")
                torch.save(self._classifier.state_dict(), model_save)

    def _train(self, epoch: int):
        running_loss = 0
        epoch_loss = 0
        batch_index = 0
        epoch_length = len(self._train_data_loader)
        for batch_index, (batch_data, batch_labels) in (
            tq := tqdm.tqdm(
                enumerate(self._train_data_loader),
                total=len(self._train_data_loader),
                desc="Epoch",
            )
        ):
            self._optimizer.zero_grad()
            batch_data = batch_data.to(self._device)
            predictions = self._classifier(batch_data)

            acc_loss = torch.zeros(1, device=self._device)
            for loss, prediction, label in zip(self._losses, predictions, batch_labels):
                label = label.to(self._device)
                acc_loss += loss(prediction, label)

            acc_loss.backward()
            self._optimizer.step()
            step_loss = acc_loss.item()
            running_loss += step_loss
            epoch_loss += step_loss
            self._writer.add_scalar(
                "Train/loss",
                step_loss,
                global_step=(epoch * epoch_length + batch_index),
            )

            if (batch_index + 1) % 10 == 0:
                tq.set_postfix({"loss": running_loss / 10})
                running_loss = 0

        epoch_loss /= epoch_length

        self._writer.add_scalar(
            "Train/epoch_loss", epoch_loss, global_step=((epoch + 1) * epoch_length - 1)
        )
        return epoch_loss

    def _eval_loader(self, loader: DataLoader, epoch: int, prefix: str):
        predictions: List[torch.Tensor] = [
            torch.tensor([], device=self._device) for _ in range(5)
        ]
        ground_truths: List[torch.Tensor] = [
            torch.tensor([], device=self._device) for _ in range(5)
        ]

        epoch_length = len(self._train_data_loader)

        for batch_index, (batch_data, batch_labels) in enumerate(loader):
            batch_data = batch_data.to(self._device)
            prediction = self._classifier(batch_data)

            for idx, (prediction, ground_truth) in enumerate(
                zip(prediction, batch_labels)
            ):
                ground_truth = ground_truth.to(self._device)
                _, predicted_idx = torch.max(prediction, -1)
                predictions[idx] = torch.cat((predictions[idx], predicted_idx))
                ground_truths[idx] = torch.cat((ground_truths[idx], ground_truth))

        accuracies: List[float] = list()
        mean_accuracy = 0
        confusion_matrices: List[np.array] = list()
        possible_values = self._test_dataset.get_latents_possible_values()
        global_step = (epoch + 1) * epoch_length - 1

        for predicted, ground_truth, attribute_name, attribute_size in zip(
            predictions,
            ground_truths,
            self._test_dataset.get_latents_names(),
            self._test_dataset.get_latents_sizes(),
        ):
            attribute_values = possible_values[attribute_name]
            correct_classifications = torch.sum(predicted == ground_truth)
            accuracy = correct_classifications / predicted.shape[0]
            accuracies.append(accuracy)
            mean_accuracy += accuracy

            confusion_matrix_slices: List[torch.Tensor] = list()
            for predicted_idx in range(attribute_size):
                confusion_matrix_slice = torch.sum(
                    predicted[ground_truth == predicted_idx][:, None]
                    == torch.arange(0, attribute_size, device=self._device)[None, :],
                    dim=0,
                )
                confusion_matrix_slice = confusion_matrix_slice / torch.sum(
                    confusion_matrix_slice
                )
                confusion_matrix_slices.append(confusion_matrix_slice)

            confusion_matrix = torch.stack(confusion_matrix_slices)
            confusion_matrices.append(confusion_matrix)
            figsize = (7, 7)
            if attribute_name.startswith("pos") or attribute_name == "orientation":
                figsize = (20, 20)
            fig = plt.figure(figsize=figsize)
            ax = fig.subplots()
            ax.set_title(f"{prefix} Confusion Matrix {attribute_name}")
            ax.matshow(confusion_matrix.cpu(), cmap=mpl.cm.Blues)

            tick_labels = attribute_values.tolist()
            if attribute_name == "shape":
                tick_labels = list(map(lambda x: f"{int(x)}", tick_labels))
            else:
                tick_labels = list(map(lambda x: f"{x:0.2f}", tick_labels))

            ax.set_ylabel("True")
            ax.set_xlabel("Prediction")
            ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(range(len(tick_labels))))
            ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(range(len(tick_labels))))

            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(tick_labels)
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(tick_labels)

            for (idx, jdx), value in np.ndenumerate(confusion_matrix.cpu()):
                ax.text(idx, jdx, f"{value:0.2f}", ha="center", va="center")

            self._writer.add_figure(
                f"{prefix}/confusion_matrix_{attribute_name}",
                fig,
                global_step=global_step,
            )
            self._writer.add_scalar(
                f"{prefix}/attribute_accuracy_{attribute_name}",
                accuracy,
                global_step=global_step,
            )

        mean_accuracy = mean_accuracy / 5
        self._writer.add_scalar(
            f"{prefix}/mean_accuracy", mean_accuracy, global_step=global_step
        )

    def _eval(self, epoch: int):
        classifier_state = self._classifier.training
        self._classifier.train(True)

        self._eval_loader(loader=self._test_data_loader, epoch=epoch, prefix="Test")

        self._eval_loader(loader=self._train_data_loader, epoch=epoch, prefix="Train")

        self._classifier.train(classifier_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dsprites-archive-path",
        type=str,
        default="resources/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        help="Path to dsprites data archive",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning Rate"
    )

    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train for"
    )

    parser.add_argument(
        "--evaluation-frequency", type=int, default=10, help="How often to test"
    )

    parser.add_argument(
        "--checkpoint-frequency", type=int, default=10, help="Checkpoint save frequency"
    )

    parser.add_argument("--output-dir", type=str, default=None, help="Logs directory")

    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="Number of parallel workers for data loaders",
    )

    args = parser.parse_args()

    # Hyper parameters
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs

    # Training options
    evaluation_frequency = args.evaluation_frequency
    checkpoint_frequency = args.checkpoint_frequency
    dsprites_archive_path = args.dsprites_archive_path
    num_workers = args.num_workers

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        output_dir = f"resources/runs/{timestamp}"
    else:
        output_dir = args.output_dir

    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")

    os.makedirs(output_dir)
    os.makedirs(checkpoints_dir)
    os.makedirs(log_dir)

    trainer = DSpritesClassifierTrainer(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_frequency=evaluation_frequency,
        checkpoint_frequency=checkpoint_frequency,
        num_workers=num_workers,
        dsprites_archive_path=dsprites_archive_path,
        output_dir=output_dir,
        checkpoints_dir=checkpoints_dir,
        log_dir=log_dir,
    )
    trainer.run()


if __name__ == "__main__":
    main()
