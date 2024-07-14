import tensorflow as tf

from tf_utils import TFRecordsManager, misc, UNet
from model import solver


def main(params):
    base_path, ckp_path = misc.get_base_path(training=True, prefix="SEG-")
    records_manager = TFRecordsManager()
    datasets = records_manager.load_datasets(base_path + params["data_path"], params["batch_size"])

    # Model
    network = UNet(params["out_ch"], params["n_layers"], params["starting_filters"], params["k_size"], params["kernel_initializer"], params["batch_norm"],
                   params["dropout"], tf.keras.layers.LeakyReLU, params["conv_per_layer"], params["max_pool"], params["upsampling"],
                   params["kernel_regularizer"])
    # network.summary((128, 128, 128, 1))

    slv = solver.Solver(ckp_path, params, list(datasets.keys()))
    for n_epoch in range(1000000):
        for mode in datasets:
            slv.iterate_dataset(network, datasets[mode], mode, n_epoch)

        if slv.consecutive_check_validation >= params["early_stopping"]:
            break

    # Save stats after training
    slv.best_metrics["epoch"] = n_epoch
    slv.best_metrics = {key: str(slv.best_metrics[key]) for key in slv.best_metrics}
    misc.save_json(ckp_path + "training.json", slv.best_metrics)


if __name__ == "__main__":

    n_layers_vector = [5]
    lr_vector = [3e-4]
    batch_norm_vector = [True]
    dropout_vector = [0.]
    batch_size_vector = [4]
    k_size_vector = [3]
    starting_filters_vector = [32]
    max_pool_vector = [False]
    upsampling_vector = [False]
    conv_per_layer_vector = [1]
    loss_vector = ["DICEL"]
    kernel_initializer_vector = ["he_normal"]
    kernel_regularizer_vector = [None]
    early_stopping = 25
    data_path = "dataset/original/TF_records_20210521_003030/"
    out_ch = 2

    for n_layers in n_layers_vector:
        for lr in lr_vector:
            for batch_norm in batch_norm_vector:
                for dropout in dropout_vector:
                    for batch_size in batch_size_vector:
                        for k_size in k_size_vector:
                            for starting_filters in starting_filters_vector:
                                for max_pool in max_pool_vector:
                                    for upsampling in upsampling_vector:
                                        for conv_per_layer in conv_per_layer_vector:
                                            for loss in loss_vector:
                                                for kernel_initializer in kernel_initializer_vector:
                                                    for kernel_regularizer in kernel_regularizer_vector:
                                                        main({"n_layers": n_layers,
                                                              "lr": lr,
                                                              "batch_norm": batch_norm,
                                                              "dropout": dropout,
                                                              "batch_size": batch_size,
                                                              "k_size": k_size,
                                                              "starting_filters": starting_filters,
                                                              "max_pool": max_pool,
                                                              "upsampling": upsampling,
                                                              "conv_per_layer": conv_per_layer,
                                                              "loss": loss,
                                                              "kernel_initializer": kernel_initializer,
                                                              "kernel_regularizer": kernel_regularizer,
                                                              "early_stopping": early_stopping,
                                                              "data_path": data_path,
                                                              "out_ch": out_ch})
