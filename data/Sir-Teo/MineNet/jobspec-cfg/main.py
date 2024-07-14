# main.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from load_dataset import SatelliteDataset
from nets import (
    VisionTransformer, SwinTransformer, ConvNeXt, ResNet50, DenseNet121, DenseNet169,
    DenseNet201, EfficientNetB0, EfficientNetB7, VSSM_Model
)
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if args.loss == "bce":
            loss = criterion(outputs.squeeze(), labels.float())
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, val_loader, test_loader, criterion, device):
    model.eval()
    val_loss, val_acc, val_f1, val_precision, val_recall = _evaluate_set(model, val_loader, criterion, device)
    test_loss, test_acc, test_f1, test_precision, test_recall = _evaluate_set(model, test_loader, criterion, device)
    return val_loss, val_acc, val_f1, val_precision, val_recall, test_loss, test_acc, test_f1, test_precision, test_recall

def _evaluate_set(model, dataloader, criterion, device):
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if args.loss == "bce":
                loss = criterion(outputs.squeeze(), labels.float())
                threshold = 0.5
                preds = (outputs.squeeze() >= threshold).long()
            else:
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())


    accuracy = correct / total if total != 0 else 0.0
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)

    return running_loss / len(dataloader.dataset), accuracy, f1, precision, recall

def plot_metrics(train_losses, val_losses, test_losses, val_accuracies, test_accuracies, val_f1_scores, test_f1_scores,
                 val_precisions, test_precisions, val_recalls, test_recalls, output_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(16, 12))
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.plot(epochs, test_losses, 'g', label='Testing Loss')
    plt.title('Training, Validation and Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.plot(epochs, test_accuracies, 'g', label='Testing Accuracy')
    plt.title('Validation and Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(epochs, val_f1_scores, 'r', label='Validation F1 Score')
    plt.plot(epochs, test_f1_scores, 'g', label='Testing F1 Score')
    plt.title('Validation and Testing F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs, val_precisions, 'r', label='Validation Precision')
    plt.plot(epochs, test_precisions, 'g', label='Testing Precision')
    plt.title('Validation and Testing Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(epochs, val_recalls, 'r', label='Validation Recall')
    plt.plot(epochs, test_recalls, 'g', label='Testing Recall')
    plt.title('Validation and Testing Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'metrics_plot_{args.model}.png'))
    plt.close()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a unique subdirectory for each run
    run_dir = os.path.join(args.output_dir, f"{args.model}_bands{'_'.join(args.bands.split(','))}_lr{args.learning_rate}_batchsize{args.batch_size}_optimizer{args.optimizer}")
    os.makedirs(run_dir, exist_ok=True)

    # Create subdirectories for TensorBoard logs and model checkpoints
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    model_dir = os.path.join(run_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Load the dataset
    train_dataset = SatelliteDataset(args.data_dir, "train", args.bands, val_size=0.2, test_size=0.1, random_state=42)
    val_dataset = SatelliteDataset(args.data_dir, "val", args.bands, val_size=0.2, test_size=0.1, random_state=42)
    test_dataset = SatelliteDataset(args.data_dir, "test", args.bands, val_size=0.2, test_size=0.1, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create the model
    if args.model == "vit":
        model = VisionTransformer(num_classes=args.num_classes, num_channels=len(args.bands.split(",")), loss=args.loss, use_weights=args.use_weights)
    elif args.model == "swin":
        model = SwinTransformer(num_classes=args.num_classes, num_channels=len(args.bands.split(",")), loss=args.loss, use_weights=args.use_weights)
    elif args.model == "convnext":
        model = ConvNeXt(num_classes=args.num_classes, num_channels=len(args.bands.split(",")), loss=args.loss, use_weights=args.use_weights)
    elif args.model == "resnet50":
        model = ResNet50(num_classes=args.num_classes, num_channels=len(args.bands.split(",")), loss=args.loss, use_weights=args.use_weights)
    elif args.model == "densenet121":
        model = DenseNet121(num_classes=args.num_classes, num_channels=len(args.bands.split(",")), loss=args.loss, use_weights=args.use_weights)
    elif args.model == "densenet169":
        model = DenseNet169(num_classes=args.num_classes, num_channels=len(args.bands.split(",")), loss=args.loss, use_weights=args.use_weights)
    elif args.model == "densenet201":
        model = DenseNet201(num_classes=args.num_classes, num_channels=len(args.bands.split(",")), loss=args.loss, use_weights=args.use_weights)
    elif args.model == "efficientnet_b0":
        model = EfficientNetB0(num_classes=args.num_classes, num_channels=len(args.bands.split(",")), loss=args.loss, use_weights=args.use_weights)
    elif args.model == "efficientnet_b7":
        model = EfficientNetB7(num_classes=args.num_classes, num_channels=len(args.bands.split(",")), loss=args.loss, use_weights=args.use_weights)
    elif args.model == "vmamba":
        model = VSSM_Model(num_classes=args.num_classes, num_channels=len(args.bands.split(",")), use_weights=args.use_weights)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    model.to(device)

    class_weights = torch.tensor([2.0, 1.0]).to(device)
    if args.loss == "crossentropy":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss == "bce":
        criterion = nn.BCELoss()

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0001)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0001)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=tensorboard_dir)

    train_losses = []
    val_losses = []
    test_losses = []
    val_accuracies = []
    test_accuracies = []
    val_f1_scores = []
    test_f1_scores = []
    val_precisions = []
    test_precisions = []
    val_recalls = []
    test_recalls = []

    # Train the model
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_precision, val_recall, test_loss, test_acc, test_f1, test_precision, test_recall = evaluate(model, val_loader, test_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)
        val_f1_scores.append(val_f1)
        test_f1_scores.append(test_f1)
        val_precisions.append(val_precision)
        test_precisions.append(test_precision)
        val_recalls.append(val_recall)
        test_recalls.append(test_recall)

        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}, Val Precision: {val_precision:.4f}, Test Precision: {test_precision:.4f}, Val Recall: {val_recall:.4f}, Test Recall: {test_recall:.4f}")

        # Record metrics in TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalar("Accuracy/Test", test_acc, epoch)
        writer.add_scalar("F1 Score/Validation", val_f1, epoch)
        writer.add_scalar("F1 Score/Test", test_f1, epoch)
        writer.add_scalar("Precision/Validation", val_precision, epoch)
        writer.add_scalar("Precision/Test", test_precision, epoch)
        writer.add_scalar("Recall/Validation", val_recall, epoch)
        writer.add_scalar("Recall/Test", test_recall, epoch)

        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_name = f"{args.model}_epoch{epoch+1}_bands{'_'.join(map(str, args.bands))}_lr{args.learning_rate}_batchsize{args.batch_size}.pth"
            torch.save(model.state_dict(), os.path.join(model_dir, model_name))

            # Plot metrics every 10 epochs
            plot_metrics(train_losses, val_losses, test_losses, val_accuracies, test_accuracies,
                         val_f1_scores, test_f1_scores, val_precisions, test_precisions,
                         val_recalls, test_recalls, run_dir)

    # Save the final model
    model_name = f"{args.model}_final_bands{'_'.join(args.bands.split(','))}_lr{args.learning_rate}_batchsize{args.batch_size}.pth"
    torch.save(model.state_dict(), os.path.join(model_dir, model_name))

    # Plot the final metrics
    plot_metrics(train_losses, val_losses, test_losses, val_accuracies, test_accuracies,
                 val_f1_scores, test_f1_scores, val_precisions, test_precisions,
                 val_recalls, test_recalls, run_dir)

    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite Image Classification")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--bands", type=str, default="0,1,2", help="Comma-separated list of bands to use for training")
    parser.add_argument("--model", type=str, default="resnet50", help="Model architecture")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for Vision Transformer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use (sgd, adam, rmsprop)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the trained models and plots")
    parser.add_argument("--loss", type=str, default="crossentropy", help="Loss function to use (crossentropy, bce)")
    parser.add_argument("--use_weights", action="store_true", help="Use pre-trained weights for the model")
    args = parser.parse_args()
    main(args)