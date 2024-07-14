from load_mnist import *
from example_cnn import *

## here we can load our trained model and compare some predictions to the true labels and corresponding images as well as the confusion matrix

## load model

file_path = get_most_recent_model(file_path="saved_models/")
print(file_path)
conv_net_instance = conv_net( output_size=10, hidden_size=16)
conv_net_instance.load_state_dict(torch.load(file_path))
## load data
mnist_dataloader = MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath,
    batch_size=3,
)
train_loader, test_loader = mnist_dataloader.load_data()
## plot images and predictions
show_images(test_loader, num_batches=2, batch_size=3, model=conv_net_instance)
show_images(
    train_loader, num_batches=2, batch_size=3, model=None, show_fig=False, save_fig=True
)

## plot confusion matrix.
plot_confusion_matrix(conv_net_instance, test_loader)
