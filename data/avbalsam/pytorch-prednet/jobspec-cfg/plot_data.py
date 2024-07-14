import argparse
import csv
import os.path
import shutil

import pandas
import pandas as pd
import scipy.special
import seaborn as sns

import csv

import torch
import torchvision.transforms
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import get_model_by_name, get_dataset_by_name
from utility import get_accuracy, get_reconstructed_images

from PIL.Image import Image


def plot_epochs(plot_type, dir_name):
    valid_plot_types = ['loss', 'accuracy']
    assert plot_type in valid_plot_types, f"Please choose a valid plot type from {valid_plot_types}"

    # dirname = f"/Users/avbalsam/Documents/openmind/prednet/model_5_gaussian_1.0/"

    if plot_type in ['loss', 'accuracy']:
        filename = f"{plot_type}_log.csv"

        data = list()

        with open(f"{dir_name}/{filename}", newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                data.append(row)

        fields = data.pop(0)

        data = [[float(i) for i in row] for row in data]

        sns.set_theme()

        data = pd.DataFrame(data[1:], columns=fields)

        dfm = data.melt('Epochs', var_name=f'{plot_type} type', value_name=plot_type)

        return sns.relplot(
            data=dfm, kind="line",
            x="Epochs", y=plot_type, hue=f'{plot_type} type'
        )


def plot_batch_across_timesteps(model, dataset):
    nt = model.nt
    ds = dataset(nt, train=False)
    val_loader = DataLoader(ds, batch_size=4, shuffle=True)

    batch_dict = dict()

    for i, (inputs, labels) in enumerate(val_loader):
        for l in range(len(labels)):
            label = labels[l].item()
            if label not in batch_dict.keys():
                batch_dict[label] = inputs[l]

    labels = ds.get_labels()

    data = [['timestep', 'predicted_label', 'confidence']]

    # if os.path.exists(f"./{model.get_name()}/input_img_over_timesteps/"):
    #     shutil.rmtree(f"./{model.get_name()}/input_img_over_timesteps/")

    for label in labels:
        if label not in batch_dict:
            continue
        input = batch_dict[label]
        batch_list = [input] * 16
        batch_tensor = torch.stack(batch_list)
        for timestep in range(nt):
            print(f"Plotting batch on timestep {timestep}...")
            classification = model(batch_tensor, timestep)
            _class = classification[0].detach().cpu().numpy()
            _class = [i - min(_class) for i in _class]
            for c in range(len(_class)):
                # ['timestep', 'predicted_label', 'confidence']
                if c == len(labels):
                    print(f"{c}: {labels} {_class}")
                else:
                    data.append([timestep, sorted(labels)[c], _class[c]])

            output_path = f"./{model.get_name()}/input_img_over_timesteps/label_{label}"
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            plot = sns.barplot(x=sorted(labels), y=_class)
            print(f"{output_path}/timestep_{timestep}.png")
            plt.imsave(f"{output_path}/input_image_{label}.png", input[0][0], cmap='gray')
            plot.set_title(f"{model.get_name()} step {timestep} label {label}")
            plt.savefig(f"{output_path}/timestep_{timestep}.png")
            plt.show()

        # The following code creates a FacetGrid with timestep data. I found that it didn't look as good.
        """
        df = pd.DataFrame(data[1:], columns=data[0])
        grid = sns.FacetGrid(df, col="timestep")
        grid.map(sns.barplot, "predicted_label", "confidence", errorbar=None)
        plt.savefig(f"{output_path}/grid_label_{label}.png")
        plt.clf()
        print(df)
        """


def plot_timesteps(model, val_loader, plot_type):
    nt = model.nt

    valid_plot_types = ['timestep accuracy']
    assert plot_type in valid_plot_types, f"Please pick a valid plot type from {valid_plot_types}"
    data = [["Timestep", "Accuracy"]]
    for timestep in range(nt):
        accuracy = get_accuracy(val_loader, model, timestep)
        print(f"Timestep: {timestep}, Accuracy: {accuracy}")
        data.append([timestep, accuracy])

    data = pd.DataFrame(data[1:], columns=data[0])

    return sns.relplot(
        data=data, kind="line",
        x="Timestep", y="Accuracy"
    )


def plot_noise_levels(model, dataset, noise_type='gaussian', noise_levels=None):
    nt = model.nt

    accuracy_over_noise = [["Noise level", "Accuracy", "Timestep"]]
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for timestep in range(nt):
        for level in noise_levels:
            noisy_data = dataset(nt, train=False, noise_type=noise_type, noise_intensities=[level])
            val_loader = DataLoader(noisy_data, batch_size=16, shuffle=True)
            accuracy_over_noise.append([level, get_accuracy(val_loader, model, timestep), timestep])

    data = pd.DataFrame(accuracy_over_noise[1:], columns=accuracy_over_noise[0])
    return sns.relplot(
        data=data, kind="line",
        x="Noise level", y="Accuracy", hue="Timestep"
    )


def plot(dir_name, model, dataset):

    print(f"Plotting loss and accuracy over epochs for model {model.get_name()} and dataset {dataset.get_name()}...")
    plot_epochs('loss', dir_name).savefig(f"{dir_name}/loss_plot.png")
    plot_epochs('accuracy', dir_name).savefig(f"{dir_name}/accuracy_plot.png")

    # Deprecated
    # print(f"Plotting accuracy over noise levels for model {dir_name}...")
    # plot_noise_levels(model, dataset).savefig(f"{dir_name}/noise_level_accuracy_plot.png")
    val_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for i, frame in enumerate(get_reconstructed_images(val_loader, model, None)):
        img = torchvision.transforms.ToPILImage()(frame)
        img.save(f'{dir_name}/reconstructed_image_{i}.png')

    print(f"Plotting accuracy over timestep for model {dir_name}...")
    plot_timesteps(model, val_loader, 'timestep accuracy').savefig(
        f"{dir_name}/timestep_accuracy_plot_{dataset.get_half()}.png")

    # Due to a memory leak, the following function is broken
    # plot_batch_across_timesteps(model, dataset)

    print(f"Finished plotting {dir_name}!\n\n")


def plot_model_dataset(dir_name, model_name, ds_name, nt, class_weight, rec_weight, half=None):
    """
    Plots important information about a model-dataset combination,
    given specifications of the model and dataset

    :param dir_name: Directory in which this model is contained
    :param model_name: Name of the model
    :param ds_name: Name of the dataset. If CKStatic, make sure to include the frame number as the last two characters
    of the name.
    :param nt: Number of timesteps (frames) to test with
    :param class_weight: Weight given to classification loss
    :param rec_weight: Weight given to reconstruction loss
    :param half: Which half of the image to use (either 'top', 'bottom' or None)
    :return: None, plots model and dataset
    """
    model = get_model_by_name(model_name, nt=nt, class_weight=class_weight, rec_weight=rec_weight)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.load_state_dict(torch.load(f"{dir_name}/model.pt", map_location=device))

    # plot_psychometric_function('./ko_data', model, nt, 16)

    # For now, we won't apply any transforms when testing the model except taking the specified half
    dataset = get_dataset_by_name(ds_name, nt=nt, train=False, transforms=None, half=half)

    plot(dir_name, model, dataset)


def plot_model(dir_name, model_name, nt, class_weight, rec_weight):
    model = get_model_by_name(model_name, nt=nt, class_weight=class_weight, rec_weight=rec_weight)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.load_state_dict(torch.load(f"{dir_name}/model.pt", map_location=device))

    plot_psychometric_function('./ko_data', model, nt, 4, dir_name)


def plot_psychometric_function(img_dir, model, nt, batch_size, output_dir):
    """
    Plots psychometric function for a given group of ko_data

    :param img_dir: Directory which contains the ko_data to plot
    :param model: Prednet model in classification mode
    :param nt: Number of timesteps to plot
    :return:
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("Gathering psychometric data...")
    data = []
    for filename in os.listdir(img_dir):
        if filename[0] == '.':
            continue
        words = filename.split(".")[0].split('-')
        fear = int(words[1].replace("fe", ""))
        happiness = int(words[2].replace("ha", ""))
        img = torchvision.io.read_image(f"{img_dir}/{filename}")
        if img.size(dim=0) == 1:
            rgb_like = torchvision.transforms.functional.rgb_to_grayscale(img.repeat(3, 1, 1))
        else:
            rgb_like = img
        resized = torchvision.transforms.Resize((256, 256))(rgb_like)
        img = torch.unsqueeze(resized, 0).repeat(1, 3, 1, 1)
        frames = list()
        for _ in range(nt):
            frames.append(img)

        frames = torch.cat(frames, 0)

        data.append((frames, (happiness, fear)))

    # if os.path.exists(f"./{output_dir}/data_log.csv"):
    #    file = open(f"./{output_dir}/data_log.csv", "r")
    #    data_to_plot = list(csv.reader(file, delimiter=","))
    #    file.close()
    # else:
    data_to_plot = list()

    print(f"Plotting psychometric function for model {model.get_name()}...")
    batch = list()
    labels = list()
    for i, (frames, (happiness, fear)) in enumerate(data):
        if i < len(data_to_plot):
            continue
        print(f"Working on image {i}...")
        frames = frames.unsqueeze(0)
        batch.append(frames)
        labels.append((happiness, fear))
        if len(batch) == batch_size:
            batch = torch.cat(batch, 0)

            classification = model(batch)
            for j in range(len(classification)):
                # happiness / fear
                happiness, fear = labels[j]
                data_to_plot.append([happiness, fear, [float(x) for x in classification[j]]])
            batch = list()
            labels = list()
            print("Finished a batch!")

    data_to_plot = [[happiness, fear, list(scipy.special.softmax(_class))] for happiness, fear, _class in data_to_plot]
    happiness_data = [[happiness, _class[1] / _class[0]] for happiness, _, _class in data_to_plot]
    fear_data = [[fear, _class[0] / _class[1]] for happiness, fear, _class in data_to_plot]

    with open(f'./{output_dir}/data_log.csv', 'w') as f:
        write = csv.writer(f)
        for datum in data_to_plot:
            write.writerow(datum)

    happiness_data = pd.DataFrame(happiness_data, columns=["H_Actual", "Confidence"])
    fear_data = pd.DataFrame(fear_data, columns=["F_Actual", "Confidence"])

    sns.regplot(
        data=happiness_data,
        x="H_Actual", y="Confidence"
    ).set(yscale="log").savefig(f"./{output_dir}/happiness_psych_plot_regression.png")

    sns.regplot(
        data=fear_data,
        x="F_Actual", y="Confidence"
    ).set(yscale="log").savefig(f"./{output_dir}/fear_psych_plot_regression.png")


def show_sample_input(dataset):
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for (label, frames) in enumerate(data_loader):
        for i, frame in enumerate(frames[0][0]):
            img = torchvision.transforms.ToPILImage()(frame[0])
            Image.show(img)
        break


if __name__ == "__main__":
    # show_sample_input(dataset=get_dataset_by_name(name='psych', nt=10, train=False, transforms=None, half=None))
    # plot_model_dataset('model:prednet_10_c0.9_r0.1:dataset:Psychometric_no_half', 'prednet', 'psych', nt=10, class_weight=0.9, rec_weight=0.1, half=None)
    plot_model('model:prednet_10_c0.9_r0.1:dataset:Psychometric_no_half', 'prednet', nt=10, class_weight=0.9, rec_weight=0.1)
    # show_sample_input(DATASETS['CK'], nt=10)
    # plot(get_model_by_name('prednet', class_weight=0.9, rec_weight=0.1, nt=10, noise_type='gaussian',
    #                       noise_intensities=[0.0]), DATASETS['CKStatic'], 'bottom')
    # plot(get_model_by_name('prednet', class_weight=0.9, rec_weight=0.1, nt=10, noise_type='gaussian',
    #                       noise_intensities=[0.0]), DATASETS['CKStatic'], 'top')
    # plot(get_model_by_name('prednet', class_weight=0.9, rec_weight=0.1, nt=10, noise_type='gaussian',
    #                       noise_intensities=[0.0]), DATASETS['CKStatic'], '')
    # print("Finished plotting prednet no noise\n\n\n", flush=True)
    # plot(get_model_by_name('prednet', class_weight=0.1, rec_weight=0.9, nt=10, noise_type='gaussian',
    #                       noise_intensities=[0.0, 0.25, 0.5]), DATASETS['mnist_frames'])
    # print("Finished plotting prednet with noise\n\n\n", flush=True)
    # plot(get_model_by_name('prednet_additive', class_weight=0.1, rec_weight=0.9, nt=10, noise_type='gaussian',
    #                       noise_intensities=[0.0, 0.25, 0.5]), DATASETS['mnist_frames'])
    # exit(0)
    # model_names = [name for name in MODELS.keys()]
    # dataset_name = 'mnist_frames'
    # for m in model_names:
    #    plot_dir(m, dataset_name)
