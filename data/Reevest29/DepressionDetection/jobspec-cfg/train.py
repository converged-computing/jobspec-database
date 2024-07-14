import argparse
from src.models import LSTM_CNN
from src.dataset import DAICSpectrogramDataset, collate_fn_padd, get_sampler, DAICSpecSpliceDataset
from src.utils import to_device
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score


def train_one_epoch(model, dataloader, optimizer, loss_fn,device):
    running_loss = 0.
    model.train()

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for data in tqdm(dataloader):
        # Every data instance is an input + label pair
        #inputs, lengths, masks, labels, severities = data
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs.to(device),None)

        c_out, r_out = outputs[:,:2], outputs[:,2:].reshape(-1)
        cls_loss, reg_loss = loss_fn

        # import pdb; pdb.set_trace()
        # Compute the loss and its gradients
        label_logits = torch.nn.functional.one_hot(labels,num_classes=2).type(torch.FloatTensor).to(device)
        # import pdb;pdb.set_trace()
        loss = cls_loss(outputs, label_logits.to(device)) #+ 0.001 * reg_loss(severities.type(torch.FloatTensor).to(device),r_out)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss / len(dataloader)

def get_accuracy(model, dataloader,device):
    model.eval()
    with torch.no_grad():
        all_labels = []
        all_outputs = []
        for data in tqdm(dataloader):
            #inputs, lengths, masks, labels, severities = data
            inputs,  labels  = data
            outputs = model(inputs.to(device),None)

            # import pdb; pdb.set_trace()
            # outputs=  torch.round(outputs)
            outputs = outputs.argmax(dim=1)
            
            
            all_labels = all_labels + list(labels.detach().cpu().numpy())
            all_outputs = all_outputs + list(outputs.detach().cpu().numpy())
    
    # print(all_labels)
    # print(all_outputs)
            
    return f1_score(all_labels,all_outputs)
class F1_loss(torch.nn.Module):
    def __init__(self,BCE):
        super().__init__()

        self.bce = BCE()


    def forward(self,y_true, y_pred, epsilon=1e-7):
        # Convert to float tensors
        y_true = y_true.float()
        y_pred = y_pred.float()

        tp = (y_true * y_pred).sum(axis=0)
        tn = ((1 - y_true) * (1 - y_pred)).sum(axis=0)
        fp = ((1 - y_true) * y_pred).sum(axis=0)
        fn = (y_true * (1 - y_pred)).sum(axis=0)

        p = tp / (tp + fp + epsilon)
        r = tp / (tp + fn + epsilon)

        f1 = 2 * p * r / (p + r + epsilon)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        return (1 - f1.mean())  + (0.01 * self.bce(y_true,y_pred))


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTM_CNN(input_size=(1,80,args.spec_seq_len),
                     lstm_input_size=2,
                     num_lstm_layers=1,
                     lstm_hidden_dim=100,
                     out_size=2).to(device)
    
    optimizer = torch.optim.Adadelta(model.parameters(),args.learning_rate)

    train_data = DAICSpecSpliceDataset(args.DAIC_location,train=True)
    val_data = DAICSpecSpliceDataset(args.DAIC_location,train=False)

    train_dataloader = DataLoader(train_data, batch_size=3,shuffle=True)#, sampler=get_sampler(),
                                  #collate_fn=collate_fn_padd)
    val_dataloader = DataLoader(val_data, batch_size=3, shuffle=True)#,
                                  #collate_fn=collate_fn_padd)
    
    loss_fn = (F1_loss(torch.nn.BCEWithLogitsLoss),torch.nn.MSELoss())
    
    for e in range(args.num_epochs):
        loss = train_one_epoch(model,train_dataloader,optimizer,loss_fn,device)
        #t_acc = get_accuracy(model,train_dataloader,device)
        v_acc = get_accuracy(model,val_dataloader,device)
        print(f"Epoch {e+1}: train loss={loss}, val acc={v_acc}")


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Train Deperession Detector',
                    description='Train a classifier that learn / whether a patient is depressed from a multimodal video interviews')
    parser.add_argument("--spec_seq_len", type=int, help="Length of Spectogram segments",default=500)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs",default=100)
    parser.add_argument("--learning_rate", "-lr", type=float,default=1)
    parser.add_argument("--DAIC_location", "-daic", type=str,default="/scratch1/tereeves/DAIC")
    args = parser.parse_args()

    train(args)
    