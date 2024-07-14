from pathlib import Path
from datetime import datetime 
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from models import StevenNet
from hdf5dataset import Hdf5Dataset






def plot(data, target):

    fig, axs = plt.subplots(nrows=12)

    x = np.arange(data.shape[1])

    nchannels = data.shape[0]
    for i in range(nchannels):
        axs[i].plot(x, data[i, :])

    plt.show()


def format_timedelta(dt, iso=False):

    # Remove microseconds
    s = dt.__str__()
    s = s[:s.find('.')]

    if not iso:
        # Remove hrs/mins if zero
        if dt.seconds < 3600:
            s = s[s.find(':')+1:]
        
        if dt.seconds < 60:
            s = s[s.find(':')+1:]

    return s


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-t', '--target', type=str, help='Target label name', required=True)
    parser.add_argument('-o', '--output_name_tag', required=True)
    parser.add_argument('-cp', '--save_checkpoints', action='store_true')
    args = parser.parse_args()


    model_save_file = (
        'models/'
        'model_' +
        args.target + '_' +
        args.output_name_tag + '_' +
        datetime.today().date().isoformat() +
        '.pt'
    )
    while Path(model_save_file).exists():
        model_save_file.replace('.pt', '_new.pt')

    
    use_gpu = True
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    run_data_precheck = False
    if args.target in ['pr', 'R_PeakAmpl_v5', 'T_PeakAmpl_v5', 'STJ_v5']:
        run_data_precheck = True
        print('Running data pre-check...')

    train_dataset = Hdf5Dataset(
        Path('../data/ECG_8lead_median_Run4_hdf5'),
        target_name=args.target,
        subset='train',
        subset_splits=(0.6, 0.2, 0.2),
        random_seed=123,
        #debug_file_limit=10000,
        precheck_data=run_data_precheck
    )
    val_dataset = Hdf5Dataset(
        Path('../data/ECG_8lead_median_Run4_hdf5'),
        target_name=args.target,
        subset='validation',
        subset_splits=(0.6, 0.2, 0.2),
        random_seed=123,
        #debug_file_limit=10000,
        precheck_data=run_data_precheck

    )

    plot = False
    if plot:
        for i in range(5):
            d, t = train_dataset[i]
            plot(d, t)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    model = StevenNet()
    #model = StandardConvNet()
    #model = TwoDimConvNet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, verbose=True)

    time0 = datetime.now()
    time1 = datetime.now()

    epochs = 100
    for i in range(epochs):
        
        avg_train_loss = 0
        avg_val_loss = 0
        
        model.train()
        for data, target in train_dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, target)
            avg_train_loss += F.mse_loss(output, target, reduction='sum').item()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.mse_loss(output, target)
                avg_val_loss += F.mse_loss(output, target, reduction='sum').item()

        avg_train_loss /= len(train_dataloader.dataset)
        avg_val_loss /= len(val_dataloader.dataset)
        scheduler.step(avg_val_loss)

        time2 = datetime.now()
        difftime_str = format_timedelta(time2 - time1)
        elapsed_str = format_timedelta(time2 - time0, iso=True)
        print('Epoch {:2d}: train loss = {:.4f}, val loss = {:.4f}, {} s/epoch, {} elapsed'.format(
            i, avg_train_loss, avg_val_loss, difftime_str, elapsed_str)
        )
        time1 = datetime.now()

        # Save checkpoint
        if args.save_checkpoint:
            if (i > 40) and (i % 2 == 0):
                chkpt_path = model_save_file.replace('.pt', f'_checkpoint_ep_{i}.pt')
                chkpt_path = chkpt_path.replace('models/', 'models/checkpoints/')
                torch.save(
                    {
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss
                    },
                    chkpt_path
                )

    torch.save(model.state_dict(), model_save_file)
    print('Saved model as', model_save_file)



