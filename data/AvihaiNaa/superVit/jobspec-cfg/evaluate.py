import torch
from tqdm import tqdm, trange
from config import CONFIG
from utils import load_dataset
from torch.nn import CrossEntropyLoss
from models.train_model import _load_checkpoint
from VIT.VIT import MyViT
import argparse



parser = argparse.ArgumentParser(description='Process args')
parser.add_argument('--ds_name', type=str, default=CONFIG.VIT.DS_NAME,
                    help="Dataset name")

args = parser.parse_args()
CONFIG.VIT.DS_NAME = args.ds_name


def evaluate():
    _, _, test_loader = load_dataset(dataset_name=CONFIG.VIT.DS_NAME, batch_size=CONFIG.VIT.BATCH_SIZE)
    model = load_model()
    evaluate_model(model, test_loader)
    print("ansqnl")


def load_model():
    model_weights = load_model_weights(exp_name=CONFIG.VIT.NAME)
    model = MyViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(CONFIG.DEVICE)
    model.load_state_dict(model_weights)
    return model

def load_model_weights(exp_name:str):
    with torch.no_grad():
        model, optimizer, loss = _load_checkpoint(exp_name=CONFIG.VIT.NAME +"_"+CONFIG.VIT.DS_NAME+"_"+CONFIG.VIT.TYPE)
        return model

def evaluate_model(model, test_loader):
    # Test loop
    criterion = CrossEntropyLoss()
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(CONFIG.DEVICE), y.to(CONFIG.DEVICE)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == "__main__":
    evaluate()