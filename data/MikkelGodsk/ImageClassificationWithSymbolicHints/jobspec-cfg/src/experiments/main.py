import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import warnings
from time import time
from typing import Tuple, Union

import click
import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np

import src.models.BERT as BERT
import src.models.LitModel as LitModel
import src.models.MultimodalModel as MultimodalModel
import src.models.ResNet50 as ResNet50
import src.models.VGG16 as VGG16
from src.features.dataset import *

warnings.filterwarnings("ignore")


# Enable/disable experiments. If experiment i is dependent on a previous experiment j, then at least a part of j will also be run.
RUN_EXPERIMENT_1 = True
RUN_EXPERIMENT_2 = True
RUN_EXPERIMENT_3 = True
RUN_EXPERIMENT_4 = True
RUN_EXPERIMENT_4C = False # Not used in paper due to poor performance
RUN_EXPERIMENT_5 = False  # Not used in paper due to poor performance
RUN_EXPERIMENT_6 = False  # Not used in paper due to poor performance

# Checkpoint path
checkpoint_path = "/work3/s184399/checkpoints"

# Typedef
img_clf_t = Union[ResNet50.LitResNet50Model, VGG16.LitVGG16Model]
text_clf_t = BERT.LitBERTModel

## Models ##
def get_img_classifier(ds_dir: str, dataset_name:str) -> img_clf_t:
    if dataset_name.lower() == "imagenet":
        resnet50_clf: ResNet50.LitResNet50Model = ResNet50.LitResNet50Model()
        return resnet50_clf
    elif dataset_name.lower() == "cmplaces":
        # Load VGG16
        vgg16_clf: VGG16.LitVGG16Model = VGG16.LitVGG16Model(os.path.join(ds_dir, 'CMPlaces'))
        return vgg16_clf
    else:
        raise ValueError


def get_text_classifier(
    load_bert: bool,
    train_text_ds=None,
    bert_hparams=None,
    dataset_name=None,  # Used for calls from outside of main.py
):
    bert_embedding_dim = 1024 if dataset_name.lower() == "imagenet" else 768
    bert_model_name = (
        "bert-large-cased" if dataset_name.lower() == "imagenet" else "bert-base-cased"
    )
    if bert_hparams is None:
        if dataset_name.lower() == "imagenet":
            bert_hparams = {
                "learning_rate": 0.0005008982647821122,
                "adam_epsilon": 1e-08,
                "weight_decay": 0.0,
                "dropout_rate": 0.005137990668310669,
                "top_dense_layer_units": [],
                "num_classes": 1000,
                "bert_embedding_dim": bert_embedding_dim,
                "bert_model_name": bert_model_name,
            }

        elif dataset_name.lower() == "cmplaces":
            # Train BERT
            bert_hparams = {
                "learning_rate": 0.0006870443398072322,
                "adam_epsilon": 1e-08,
                "weight_decay": 0.0,
                "dropout_rate": 0.31547838646677684,
                "top_dense_layer_units": [],
                "num_classes": 205,
                "bert_embedding_dim": bert_embedding_dim,
                "bert_model_name": bert_model_name,
            }  # Found using Optuna (see file tune_BERT_hparams.py)
        else:
            raise ValueError

    if not load_bert:
        text_clf: BERT.LitBERTModel = BERT.train_and_evaluate(
            bert_hparams,
            train_ds=text_dataloader(train_text_ds),
            n_epochs=5,
            model_version="Train_BERT",
        )[0]
        torch.save(text_clf.state_dict(), os.path.join('models', "BERT_model_" + dataset_name.lower()))
    else:
        text_clf: BERT.LitBERTModel = BERT.LitBERTModel(**bert_hparams)
        text_clf.load_state_dict(torch.load(os.path.join('models', "BERT_model_" + dataset_name.lower())))
    return text_clf


def evaluate_model(
    model: pl.LightningModule, dataloader: torch.utils.data.DataLoader, model_description: str
):
    trainer = pl.Trainer(
        gpus=min(1, torch.cuda.device_count()),
        max_epochs=5,
        # callbacks=[pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path)],
        # checkpoint_callback=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    print('\n\033[91m\033[4mModel "{:s}" results:\033[0m'.format(model_description))
    performance = trainer.test(model, dataloaders=dataloader)
    print(performance)
    return performance[0]["hp_test_top_1_acc"]


def print_time(t_prep, t_eval):
    # Format time correctly and print
    text_ = "'\nTime for preparation: {:s} ({:.2f}s) - Time for evaluation: {:s} ({:.2f}s)\n'"
    times = []
    for t in [t_prep, t_eval]:
        if t < 60:
            times.append("{:.2f}s".format(t))
        if t >= 60 and t < 60 * 60:
            minutes = int(t) // 60
            seconds = t - 60 * minutes
            times.append("{:d}m {:.2f}s".format(minutes, seconds))
        if t >= 60 * 60:
            hours = int(t) // (60 * 60)
            minutes = int(t - 60 * 60 * hours) // 60
            seconds = t - 60 * 60 * hours - 60 * minutes
            times.append("{:d}h {:d}m {:.2f}s".format(hours, minutes, seconds))
        times.append(t)

    print(text_.format(*times), end=" ", flush=True)


def experiment1(
    bimodal_val_ds: DataHandler.BimodalDS,
    train_text_ds: DataHandler.TextModalityDS,
    load_bert=True,
    dataset_name: str=None,
    ds_dir: str=None
) -> Tuple[text_clf_t, img_clf_t]:
    """
        1) We trained a BERT-based classifier on the text modality and evaluated both the text and image classifier on their respective modalities to get unimodal baseline performances;
    """
    img_clf = get_img_classifier(dataset_name=dataset_name, ds_dir=ds_dir)
    text_clf = get_text_classifier(
        train_text_ds=train_text_ds, load_bert=load_bert, dataset_name=dataset_name
    )

    # Break the calibration to prove a point
    with torch.no_grad():
        bert_multiplier = 2.5
        img_clf_multiplier = 0.9
        text_clf.top[2].weight *= bert_multiplier
        text_clf.top[2].bias *= bert_multiplier
        img_clf.top.weight *= img_clf_multiplier
        img_clf.top.bias *= img_clf_multiplier

    # Evaluate both
    if RUN_EXPERIMENT_1:
        eval_text_start = time()
        evaluate_model(
            text_clf,
            text_dataloader(bimodal_val_ds.text_ds, shuffle=False),
            "Uncalibrated {:s} based classifier".format(text_clf.name),
        )
        eval_text_end = time()
        print_time(0, eval_text_end - eval_text_start)
        eval_img_start = time()
        evaluate_model(
            img_clf,
            img_dataloader(bimodal_val_ds.image_ds, shuffle=False),
            "Uncalibrated {:s} classifier".format(img_clf.name),
        )
        eval_img_end = time()
        print_time(0, eval_img_end - eval_img_start)

    return text_clf, img_clf


def experiment2(
    text_clf: text_clf_t, img_clf: img_clf_t, bimodal_val_ds: DataHandler.BimodalDS, dataset_name: str=None, num_classes:int=205,
) -> MultimodalModel.NaiveBayesFusion:
    """
        2) We combined the two uncalibrated classifiers using equation (1) and evaluated the fusion model;
    """
    fusion_model: MultimodalModel.NaiveBayesFusion = MultimodalModel.NaiveBayesFusion(
        img_clf, text_clf
    )
    if RUN_EXPERIMENT_2:
        prep_combined_start = time()
        fusion_model.estimate_prior_class_probs(bimodal_val_ds, n_classes=num_classes)
        prep_combined_end = time()
        fusion_model.cuda()
        eval_combined_start = time()
        evaluate_model(
            fusion_model,
            bimodal_dataloader(bimodal_val_ds, shuffle=False),
            "Naive Bayes fusion of uncalibrated {:s} and uncalibrated {:s}".format(
                img_clf.name, text_clf.name
            ),
        )
        eval_combined_end = time()
        print_time(
            prep_combined_end - prep_combined_start,
            eval_combined_end - eval_combined_start,
        )
        fusion_model.cuda()
        LitModel.reliability_plot(
            fusion_model,
            bimodal_dataloader(bimodal_val_ds, shuffle=False),
            "Uncalibrated fusion model for {:s}".format(dataset_name),
        )

    return fusion_model


def experiment3(
    text_clf: text_clf_t, img_clf: img_clf_t, bimodal_val_ds: DataHandler.BimodalDS, dataset_name: str=None, num_classes:int=205,
) -> Tuple[
    LitModel.CalibratedLitModel,
    LitModel.CalibratedLitModel,
    MultimodalModel.NaiveBayesFusion,
]:
    """
    3) We calibrated both classifiers and evaluated them. This experiment was motivated by Section 2.2 and our hypothesis that the fusion model performs better if we use well-calibrated unimodal classifiers as the base models;
    4) We combined the calibrated classifiers using equation (1) and evaluated the fusion model;

    Note: Reliability plots are also created here.
    """
    prep_combined_start = time()
    prep_text_start = time()
    calibrated_text_clf: LitModel.CalibratedLitModel = LitModel.calibrate(
        text_clf.cuda(),
        text_dataloader(
            bimodal_val_ds.text_ds, shuffle=False
        ),  # Large batch size is preferred for calibration
        model_name="{:s} for {:s}".format(text_clf.name, dataset_name),
        learning_rate=0.3,
        max_iter=50,
    )
    prep_text_end = time()
    prep_img_start = time()
    calibrated_img_clf: LitModel.CalibratedLitModel = LitModel.calibrate(
        img_clf.cuda(),
        img_dataloader(bimodal_val_ds.image_ds, shuffle=False),
        model_name="{:s} for {:s}".format(img_clf.name, dataset_name),
        max_iter=50,
        learning_rate=0.1,
    )
    prep_img_end = time()
    calibrated_fusion_model: MultimodalModel.NaiveBayesFusion = (
        MultimodalModel.NaiveBayesFusion(calibrated_img_clf, calibrated_text_clf, num_classes=num_classes)
    )
    calibrated_fusion_model.estimate_prior_class_probs(
        bimodal_val_ds, n_classes=num_classes
    )
    prep_combined_end = time()

    calibrated_fusion_model.cuda()
    calibrated_text_clf.cuda()
    calibrated_img_clf.cuda()

    eval_text_start = time()
    evaluate_model(
        calibrated_text_clf,
        text_dataloader(bimodal_val_ds.text_ds, shuffle=False),
        "Calibrated {:s} model".format(text_clf.name),
    )
    eval_text_end = time()
    print_time(prep_text_end - prep_text_start, eval_text_end - eval_text_start)

    eval_img_start = time()
    evaluate_model(
        calibrated_img_clf,
        img_dataloader(bimodal_val_ds.image_ds, shuffle=False),
        "Calibrated {:s} model".format(img_clf.name),
    )
    eval_img_end = time()
    print_time(prep_img_end - prep_img_start, eval_img_end - eval_img_start)

    eval_combined_start = time()
    evaluate_model(
        calibrated_fusion_model,
        bimodal_dataloader(bimodal_val_ds, shuffle=False),
        "Calibrated naive Bayes fusion model",
    )
    eval_combined_end = time()
    print_time(
        prep_combined_end - prep_combined_start, eval_combined_end - eval_combined_start
    )

    calibrated_fusion_model.cuda()
    LitModel.reliability_plot(
        calibrated_fusion_model,
        bimodal_dataloader(bimodal_val_ds, shuffle=False),
        "Calibrated naive Bayes fusion model for {:s}".format(dataset_name),
    )  # Reliability plots for the constituent classifiers are made in the call to `LitModel.calibrate`

    return calibrated_text_clf, calibrated_img_clf, calibrated_fusion_model


def experiment4a(
    text_clf: text_clf_t,
    img_clf: img_clf_t,
    train_ds: DataHandler.BimodalDS,
    val_ds: DataHandler.BimodalDS,
    regularization_strength: int = 3e-7,  # Found using `tune_SVM.py`
    dataset_name: str=None,
    n_classes=None,  # For compatibility with older code,
    n_jobs:int=1,
    debug=False
):
    """
        5) We trained a linear SVM classifier and an XGBoost classifier on the concatenated embeddings coming from both classifiers 
        (i.e. for this experiment, we stripped the models of their tops) and evaluated it. 
        The regularization parameter was tuned using the hold-out method (on a validation split of the training set).
    """
    if n_classes is not None:
        n_classes = num_classes  # So this function can be used in tune_SVM.py
    elif dataset_name is not None:
        n_classes = 205 if dataset_name.lower() == "cmplaces" else 1000

    svm_fusion = MultimodalModel.SVMBasedFusion(
        img_clf, text_clf, regularization_strength=regularization_strength, n_jobs=n_jobs
    )
    svm_fusion.cuda()
    prep_svm_start = time()
    svm_fusion.fit_top(
        bimodal_dataloader(train_ds, shuffle=True, batch_size=32),
        epochs=1,
        verbose=debug,
        n_classes=n_classes,
    )
    svm_fusion.calibrate_top(bimodal_dataloader(val_ds, shuffle=True), verbose=debug)
    prep_svm_end = time()
    eval_svm_start = time()
    evaluate_model(
        svm_fusion, bimodal_dataloader(val_ds, shuffle=False), "SVM-based fusion"
    )
    eval_svm_end = time()
    print_time(prep_svm_end - prep_svm_start, eval_svm_end - eval_svm_start)
    return svm_fusion


def experiment4b(
    text_clf: text_clf_t,
    img_clf: img_clf_t,
    train_ds: DataHandler.BimodalDS,
    val_ds: DataHandler.BimodalDS,
    n_epochs=1,
    dataset_name: str=None,
    debug=False
):
    """
        NOT USED. Gave very poor performance.
        We train a logistic regression on the concatenated embeddings coming from both classifiers and evaluate on the validation set.
    """
    params = {"lr": 0.0003172298313831537, "top_l2_rate": 2.723639829156825e-05}
    fusion_model = MultimodalModel.LogisticRegressionBasedFusion(
        img_clf, text_clf, out_features=num_classes, **params
    )
    fusion_model.cuda()

    logger = pl.loggers.TensorBoardLogger(
        save_dir="lightning_logs",
        name="Experiments",
        version="log_reg_fusion_model_{:s}".format(dataset_name),
    )

    trainer = pl.Trainer(
        gpus=min(1, torch.cuda.device_count()),
        max_epochs=n_epochs,
        logger=logger,
        max_time="00:23:00:00",
        # callbacks=[pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path)],
        # checkpoint_callback=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=debug,
        # auto_lr_find = True
    )
    trainer.fit(fusion_model, bimodal_dataloader(train_ds, shuffle=False))
    evaluate_model(
        fusion_model,
        bimodal_dataloader(val_ds, shuffle=False),
        "Logistic regression fusion model",
    )
    fusion_model.cuda()
    LitModel.reliability_plot(
        fusion_model,
        bimodal_dataloader(val_ds, shuffle=False),
        "Logistic regression fusion model for {:s}".format(dataset_name),
    )
    return fusion_model


def experiment4c(
    text_clf: text_clf_t,
    img_clf: img_clf_t,
    train_ds: DataHandler.BimodalDS,
    val_ds: DataHandler.BimodalDS,
    n_classes=None,
    dataset_name: str=None,
    cache_dir: str=None,
    n_jobs:int=1,
    subsample:bool=False,
    debug=False
):
    """
        5) We trained a linear SVM classifier and an XGBoost classifier on the concatenated embeddings coming from both classifiers 
        (i.e. for this experiment, we stripped the models of their tops) and evaluated it. 
        The regularization parameter was tuned using the hold-out method (on a validation split of the training set).

    """
    import wandb
    wandb.init(project="XGBoost fusion")
    from sklearn.utils.validation import check_is_fitted
    if n_classes is not None:
        n_classes = num_classes
    elif dataset_name is not None:
        n_classes = 205 if dataset_name.lower() == "cmplaces" else 1000

    xgb_fusion = MultimodalModel.XGBFusion(
        img_clf, text_clf, n_jobs=n_jobs, cache_dir=cache_dir, enable_subsample=subsample
    )
    wandb.watch(xgb_fusion, log_freq=100)
    prev_clf = None
    for i in range(100):
        try:
            xgb_fusion.cuda()
            print("n_estimators={:d}".format(i+1))
            prep_svm_start = time()
            xgb_fusion.fit_top(
                bimodal_dataloader(train_ds, shuffle=True, batch_size=32),
                verbose=debug,
                n_classes=n_classes,
                n_estimators=1,
                xgb_model=prev_clf
            )
            check_is_fitted(xgb_fusion.xgc)
            
            #xgb_fusion.calibrate_top(bimodal_dataloader(val_ds, shuffle=True), verbose=debug)   # Does not work for whatever reason
            prep_svm_end = time()
            eval_svm_start = time()
            evaluate_model(
                xgb_fusion, bimodal_dataloader(val_ds, shuffle=False), "XGB-based fusion"
            )
            eval_svm_end = time()
            print_time(prep_svm_end - prep_svm_start, eval_svm_end - eval_svm_start)
            prev_clf = xgb_fusion.xgc.get_booster()
        except np.core._exceptions._ArrayMemoryError:
            break
    return xgb_fusion

def experiment5(
    fusion_model: MultimodalModel.NaiveBayesFusion,
    train_ds: DataHandler.BimodalDS,
    val_ds: DataHandler.BimodalDS,
    n_epochs=1,
    make_plots=True,
    train_log_prior_probs=None,
    val_log_prior_probs=None,
    params=None,
    reset_tops=True,
    dataset_name: str=None,
    debug=False
) -> MultimodalModel.NaiveBayesFusion:
    """
    NOT USED: Gave very poor performance
    We now fine-tune the naive Bayes fusion model of the uncalibrated BERT-based classifier and image classifier on the training-set and evaluate the model on the validation set.
    """
    if params is None:
        if dataset_name.lower() == "cmplaces":
            # params = {
            #    'img_l2_rate': 0.0011873241371175235,
            #    'lr': 3.2794668207275364e-05/10,
            #    'text_l2_rate': 3.7819905197507134e-06
            # }
            # params = {
            #    'img_l2_rate': 5.963608800848493e-10,
            #    'lr': 6.129034610142461e-05,
            #    'text_l2_rate': 1.342031423378103e-06
            # }
            # params = {
            #    'img_l2_rate': 1.0867701841181138e-12,
            #    'text_l2_rate': 4.123133770554182e-11,
            #    'lr': 8.309504386925207e-05
            # }
            params = {
                "lr": 2.84e-5,
                "img_l2_rate": 0.004811374989058345,
                "text_l2_rate": 2.6700507325691306e-07,
            }

        elif dataset_name.lower() == "imagenet":
            # params = {
            #    'img_l2_rate': 0.0035909569066556383,
            #    'lr': 1.7673281199723105e-05,
            #    'text_l2_rate': 1.8042518669844849e-09
            # }
            # params = {
            #    'img_l2_rate': 2.7657018755348782e-05,
            #    'lr': 2.7126771897842036e-05,
            #    'text_l2_rate': 0.00037447509546665946,
            # }
            params = {
                "lr": 0.00015130540255110472,
                "img_l2_rate": 2.16066281693861e-05,
                "text_l2_rate": 0.6142140199001819,
            }

        fusion_model.learning_rate = params["lr"]
        fusion_model.weight_decay = [params["img_l2_rate"], params["text_l2_rate"]]

    # Reset tops if needed
    if reset_tops:
        for clf in fusion_model.classifiers:
            if isinstance(clf.top, torch.nn.Sequential):
                # If top is sequential, we find all linear layers
                for layer in clf.top:
                    if isinstance(layer, torch.nn.Linear):
                        layer.reset_parameters()
            elif isinstance(clf.top, torch.nn.Linear):
                clf.top.reset_parameters()
            else:
                raise ValueError("Top in {:s} is not sequential nor linear".format(clf))

    logger = pl.loggers.TensorBoardLogger(
        save_dir="~/bscproj/CMPlaces/lightning_logs",
        name="Experiments",
        version="Fine-tuned_fusion_model_{:s}".format(dataset_name),
    )

    prep_combined_start = time()
    fusion_model.estimate_prior_class_probs(
        train_ds, n_classes=num_classes, precomputed_log_probs=train_log_prior_probs
    )
    fusion_model.cuda()

    fusion_model.freeze_net = True
    trainer = pl.Trainer(
        gpus=min(1, torch.cuda.device_count()),
        max_epochs=n_epochs,
        logger=logger,
        max_time="00:23:00:00",
        # callbacks=[pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path)],
        # checkpoint_callback=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=debug,
        # auto_lr_find = True
    )
    trainer.fit(fusion_model, bimodal_dataloader(train_ds, shuffle=False))

    fusion_model.estimate_prior_class_probs(
        val_ds, n_classes=num_classes, precomputed_log_probs=val_log_prior_probs
    )
    prep_combined_end = time()

    eval_combined_start = time()
    fusion_model.cuda()
    top_1_acc = evaluate_model(
        fusion_model,
        bimodal_dataloader(val_ds, shuffle=False),
        "Fine-tuned fusion model",
    )
    eval_combined_end = time()
    print_time(
        prep_combined_end - prep_combined_start, eval_combined_end - eval_combined_start
    )
    fusion_model.cuda()
    if make_plots:
        LitModel.reliability_plot(
            fusion_model,
            bimodal_dataloader(val_ds, shuffle=False),
            "Fine-tuned fusion model for {:s}".format(dataset_name),
        )
    return top_1_acc, fusion_model


def experiment6(fusion_model: MultimodalModel.NaiveBayesFusion, val_ds: DataHandler.BimodalDS, dataset_name: str=None):
    """
    NOT USED: Gave very poor performance
    We recover the two classifiers after fine-tuning and evaluate them on the validation set invidiually.
    We also examine their reliability plots to see if the classifiers become better calibrated in the process.
    """
    print("Number of classifiers:")
    print(len(fusion_model.classifiers))
    img_clf, text_clf = fusion_model.classifiers[0], fusion_model.classifiers[1]
    img_ds = val_ds.image_ds
    text_ds = val_ds.text_ds

    img_clf.cuda()
    text_clf.cuda()

    eval_img_start = time()
    evaluate_model(
        img_clf,
        img_dataloader(img_ds, shuffle=False),
        "Recovered (fine-tuned) {:s} model".format(img_clf.name),
    )
    eval_img_end = time()
    print_time(0, eval_img_end - eval_img_start)
    eval_text_start = time()
    evaluate_model(
        text_clf,
        text_dataloader(text_ds, shuffle=False),
        "Recovered (fine-tuned) {:s} model".format(text_clf.name),
    )
    eval_text_end = time()
    print_time(0, eval_text_end - eval_text_start)

    img_clf.cuda()  # PyTorch lightning moves model to CPU after training/evaluation.
    text_clf.cuda()

    LitModel.reliability_plot(
        img_clf,
        img_dataloader(img_ds, shuffle=False),
        "Recovered (fine-tuned) {:s} model for {:s}".format(img_clf.name, dataset_name),
    )
    LitModel.reliability_plot(
        text_clf,
        text_dataloader(text_ds, shuffle=False),
        "Recovered (fine-tuned) {:s} model for {:s}".format(text_clf.name, dataset_name),
    )


@click.command()
@click.option(
    "--dataset",
    default="CMPlaces",
    help="Dataset to use. Either 'CMPlaces' or 'ImageNet'"
)
@click.option(
    "--ds_dir",
    default="/work3/s184399",
    help="The directory in which the datasets are located"
)
@click.option(
    "--n_workers",
    default=1,
    help="The number of workers used for XGBoost and SVM"
)
@click.option(
    "--xgb_subsample",
    default=0,
    help="Whether to subsample the dataset in XGBoost"
)
@click.option(
    "--debug",
    default=0,
    help="Runs the code on the validation set instead to see if we can overfit a model to it"
)
def main(*args, **kwargs):
    dataset_name = kwargs["dataset"]
    ds_dir = kwargs['ds_dir']
    n_jobs = kwargs['n_workers']
    debug = kwargs['debug'] > 0
    #set_dataset(dataset_name)   # Only relic of a bad practice (hopefully)
    dhandler = DataHandler(ds_name=dataset_name, ds_dir=ds_dir)
    train_bert = not os.path.isfile(
        os.path.join('models', "BERT_model_" + dataset_name.lower())
    )  # Must be true the first time, then set to false the second time (for each dataset)


    print(
        "\n\033[1m\033[34mUsing dataset: {:s}\033[0m".format(dataset_name)
    )  # Formatting codes: \033[Xm where X is an integer

    # Datasets
    if debug:
        train_text_ds: DataHandler.TextModalityDS = dhandler.load_text_ds("val_text.json")
    else:
        train_text_ds: DataHandler.TextModalityDS = dhandler.load_text_ds("train_text.json")
    val_text_ds: DataHandler.TextModalityDS = dhandler.load_text_ds("val_text.json")
    val_img_ds: DataHandler.ImageModalityDS = dhandler.load_img_ds("val")
    bimodal_val_ds: DataHandler.BimodalDS = dhandler.BimodalDS(image_ds=val_img_ds, text_ds=val_text_ds)

    # Experiment 1
    if (
        RUN_EXPERIMENT_1
        or RUN_EXPERIMENT_2
        or RUN_EXPERIMENT_3
        or RUN_EXPERIMENT_4
        or RUN_EXPERIMENT_4C
        or RUN_EXPERIMENT_5
        or RUN_EXPERIMENT_6
    ):
        print("\n\033[1m\033[34mExperiment 1\033[0m")
        models: Tuple[text_clf_t, img_clf_t] = experiment1(
            bimodal_val_ds=bimodal_val_ds,
            train_text_ds=train_text_ds,
            load_bert=not train_bert,
            dataset_name=dataset_name,
            ds_dir=ds_dir
        )
        text_clf: text_clf_t = models[0]
        img_clf: img_clf_t = models[1]

    # Experiment 2
    if RUN_EXPERIMENT_2 or RUN_EXPERIMENT_5 or RUN_EXPERIMENT_6:
        print("\n\033[1m\033[34mExperiment 2\033[0m")
        fusion_model: MultimodalModel.NaiveBayesFusion = experiment2(
            text_clf=text_clf, img_clf=img_clf, bimodal_val_ds=bimodal_val_ds, dataset_name=dataset_name, num_classes=dhandler.num_classes,
        )

    # Experiment 3
    if RUN_EXPERIMENT_3:
        print("\n\033[1m\033[34mExperiment 3\033[0m")
        experiment3(
            text_clf=text_clf, img_clf=img_clf, bimodal_val_ds=bimodal_val_ds, dataset_name=dataset_name, num_classes=dhandler.num_classes,
        )  # For now, we do not reuse the models here

    # Load image training set (is huge)
    if (RUN_EXPERIMENT_4 or RUN_EXPERIMENT_4C or RUN_EXPERIMENT_5 or RUN_EXPERIMENT_6) and not debug:
        train_img_ds: DataHandler.ImageModalityDS = dhandler.load_img_ds(
            "train"
        )  # This call will take a long time... (Approx. 40m)
        bimodal_train_ds: DataHandler.BimodalDS = dhandler.BimodalDS(
            image_ds=train_img_ds, text_ds=train_text_ds
        )
    elif debug:
        train_img_ds: DataHandler.ImageModalityDS = dhandler.load_img_ds(
            "val"
        )  # This call will take a long time... (Approx. 40m)
        bimodal_train_ds: DataHandler.BimodalDS = dhandler.BimodalDS(
            image_ds=train_img_ds, text_ds=train_text_ds
        )

    # Experiment 4
    if RUN_EXPERIMENT_4:
        print("\n\033[1m\033[34mExperiment 4\033[0m")
        experiment4a(
            text_clf,
            img_clf,
            bimodal_train_ds,
            bimodal_val_ds,
            regularization_strength=3e-7 if dataset_name.lower() == "cmplaces" else 3e-3,
            dataset_name=dataset_name,
            n_jobs=n_jobs,
            debug=debug
        )
        # experiment4b(
        #    text_clf,
        #    img_clf,
        #    bimodal_train_ds if not debug else bimodal_val_ds,
        #    bimodal_val_ds,
        #    debug=debug
        # )

    if RUN_EXPERIMENT_4C:
        print("\n\033[1m\033[34mExperiment 4c\033[0m")
        experiment4c(
            text_clf,
            img_clf,
            bimodal_train_ds,
            bimodal_val_ds,
            dataset_name=dataset_name,
            cache_dir=ds_dir,
            n_jobs=n_jobs,
            subsample=kwargs['xgb_subsample'] > 0,
            debug=debug
        )


    # Experiment 5
    if RUN_EXPERIMENT_5 or RUN_EXPERIMENT_6:
        print("\n\033[1m\033[34mExperiment 5\033[0m")
        train_val_split = 0.8
        train_text_split, val_text_split = dhandler.split_text_ds(train_text_ds, train_val_split)
        train_img_split, val_img_split = dhandler.split_img_ds(train_img_ds, train_val_split)
        train_split = dhandler.BimodalDS(image_ds=train_img_split, text_ds=train_text_split)
        res = experiment5(
            fusion_model=fusion_model,
            train_ds=train_split,  # bimodal_train_ds if not debug else bimodal_val_ds,
            val_ds=bimodal_val_ds,
            n_epochs=1,
            dataset_name=dataset_name,
            debug=debug
        )  # For now, fine_tuned_fusion_model refers to fusion_model, and fusion_model is trained
        fine_tuned_fusion_model: MultimodalModel.NaiveBayesFusion = res[1]

    # Experiment 6
    if RUN_EXPERIMENT_6:
        print("\n\033[1m\033[34mExperiment 6\033[0m")
        experiment6(fusion_model=fine_tuned_fusion_model, val_ds=bimodal_val_ds, dataset_name=dataset_name)


if __name__ == "__main__":
    main()
