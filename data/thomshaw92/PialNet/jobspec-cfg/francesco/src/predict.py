import tensorflow as tf
import nibabel as nib
import argparse
import numpy as np

from tf_utils import misc, UNet
from model import solver
from preprocessing import data_loader


def main(ckp_path, ckp_name, input_path, label_path):
    assert (ckp_path[-1] == "/")
    base_path = misc.get_base_path(training=False)
    params = misc.load_json(base_path + ckp_path + "params.json")

    network = UNet(params["out_ch"], params["n_layers"], params["starting_filters"], params["k_size"], params["kernel_initializer"], params["batch_norm"],
                   params["dropout"], tf.keras.layers.LeakyReLU, params["conv_per_layer"], params["max_pool"], params["upsampling"],
                   params["kernel_regularizer"])
    network.load_weights(base_path + ckp_path + ckp_name)

    slv = solver.Solver(None, params, ["predicting"])

    # Load data
    data, meta = data_loader.load_testing_volume({"x": base_path + input_path, "y": base_path + label_path if label_path is not None else label_path},
                                                 base_path + params["data_path"] + "norm.pkl")

    logits = np.zeros([data["x"].shape[0], data["x"].shape[1], data["x"].shape[2], data["x"].shape[3], params["out_ch"]])
    for j in range(0, logits.shape[1], 256):
        for k in range(0, logits.shape[2], 256):
            logits[:, j:j + 256, k:k + 256, :, :] = slv.test_step(network, data["x"][:, j:j + 256, k:k + 256, :, :], None, "test")

    logits = tf.cast(logits, tf.float64)
    predictions = misc.get_argmax_prediction(logits)

    if label_path is not None:
        with open(base_path + ckp_path + "prediction_stats.txt", "a") as file:
            file.write(ckp_name + " " + input_path + " : " + str(slv.loss_manager.dice_score_from_logits(tf.cast(data["y"], tf.float64), logits).numpy()) + "\n")

    assert (len(predictions.shape) == 5 and predictions.shape[-1] == 1)
    if "pad_added" in meta:
        predictions = misc.remove_padding(predictions, meta["orig_shape"], meta["pad_added"])

    filename = input_path.split("/")[-1].split(".")[0]
    nib.save(nib.Nifti1Image(predictions[0, :, :, :, 0], meta["affine"], meta["header"]), base_path + ckp_path + str(ckp_name) + "-" + filename + ".nii.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckp_path", type=str, required=True)
    parser.add_argument("--ckp_name", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--label_path", type=str, required=False)
    args = parser.parse_args()

    main(args.ckp_path, args.ckp_name, args.input_path, args.label_path)
