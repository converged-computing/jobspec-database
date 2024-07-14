import math
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import TensorboardLogger, save_checkpoint


# def lr_scheduler(args, optimizer):
#     l = lambda current_step: min(pow(current_steps, -0.5), current_steps * pow(args['WARMUP_STEP'], 1.5))
#     sceduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=l)
#     return sceduler

def cycle_loss(model, ciphers, plains, keys):
    pass

class compute_loss(nn.Module):
    def __init__(self, args):
        super(compute_loss, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')

class compute_loss_pos(compute_loss):
    def __init__(self, args):
        super().__init__(args)
        # self.mask = torch.zeros(size=[args['SEQ_LENGTH'] + 7])
        # self.mask[0:args['SEQ_LENGTH'] + 2] = 1
        # self.mask = self.mask.bool().to(args['DEVICE'])

    def forward(self, model, inputs, targets):
        pred_targets = model(inputs).permute(1, 2, 0) # []
        # pred_targets = pred_targets[..., self.mask]
        # targets = targets[..., self.mask]

        loss = self.ce(pred_targets, targets)
        return loss, pred_targets

class compute_loss_cp2k(compute_loss):
    def __init__(self, args):
        super().__init__(args)

    def forward(self, model, inputs, targets, masks):
        loss = []
        pred_targets = model(inputs, masks)
        for i, pred in enumerate(pred_targets): # pred: [rotor, seq, batch, out_channels] targets: [rotor, seq, batch] mask: [batch, seq]
            # Applying mask
            loss.append(self.ce(pred[~masks.T], targets[i][~masks.T])) # pred -> [batch, out_channels, seq] targets[i].T: [batch, seq]
        # pred_targets = model(inputs)
        # loss = self.ce(pred_targets, targets)
        return sum(loss), pred_targets


class compute_loss_cp2_further(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, model, inputs, targets, masks, further_experiment_targets):
        loss = []
        # fur_loss = []
        pred_targets, pred_further_targets = model(inputs, masks)
        for i, pred in enumerate(pred_targets): # pred: [rotor, seq, batch, out_channels] targets: [rotor, seq, batch] mask: [batch, seq]
            # Applying mask
            loss.append(self.ce(pred[~masks.T], targets[i][~masks.T])) # pred -> [batch, out_channels, seq] targets[i].T: [batch, seq]

        for idx, pred in enumerate(pred_further_targets):
            insss, inssss = pred, further_experiment_targets[idx]
            mins, maxs = torch.min(further_experiment_targets[idx]), torch.max(further_experiment_targets[idx])
            loss.append(self.ce(pred, further_experiment_targets[idx]))
        return sum(loss), pred_targets, pred_further_targets




def train(model, optimizer, dataset, dataloader, initial_step, initial_epoch, mix_scaler, args):
    '''
    config the training pipeline.


    :param model:
    :param dataloader:
    :param args:
    :return:
    '''
    # Set up tensorboard logger
    logger = TensorboardLogger(args)

    # Set up the loss
    critic = compute_loss_cp2_further(args=args) # Loss for further experiment

    # Tracking the training stats and learning rate scheduling
    loss_best = 9999999
    acc_best = 0
    current_steps = initial_step

    # Tracking process
    true_positive = 0
    samples = 0
    loss_sum = 0

    # Further experiment acc
    metric_tags_fur = []
    if args['Random_rotors_num'] is not None:
        metric_tags_fur.append('rotor')

    if args['Random_reflector'] is not None:
        metric_tags_fur.append('reflector')

    if args['Random_ringsetting_num'] is not None:
        metric_tags_fur.append('ringsetting')

    if args['Random_plugboard_num'] is not None:
        metric_tags_fur.append('plugboard')

    Fur_acc = [[0, 0] for _ in range(len(metric_tags_fur))]

    # Training loop
    for epoch in range(args['EPOCHS']):
        if args['PROGRESS_BAR'] == 1:
            bar = tqdm(dataloader, leave=True)
        else:
            bar = dataloader

        for idx, (inputs, targets, masks, further_experiment_targets) in enumerate(bar):
            # send to device
            inputs, targets, masks = inputs.to(args['DEVICE']), targets.to(args['DEVICE']), masks.to(args['DEVICE'])
            for fur_idx in range(len(further_experiment_targets)):
                further_experiment_targets[fur_idx] = further_experiment_targets[fur_idx].to(args['DEVICE'])

            # Make prediction
            with torch.cuda.amp.autocast():
                # compute the mean for optimizing and sum for logging
                loss, pred, fur_pred = critic(model, inputs, targets, masks, further_experiment_targets) # pred: [rotor, seq, batch, out_channels]

            # Update weights in mix precision
            optimizer.zero_grad()
            mix_scaler.scale(loss).backward()
            mix_scaler.step(optimizer)
            mix_scaler.update()

            # Warm up learning rate
            for g in optimizer.param_groups:
                g['lr'] = min(args['LEARNING_RATE'], current_steps * (args['LEARNING_RATE'] / args[
                    'WARMUP_STEP']))  # args['LEARNING_RATE'] * min(pow(current_steps, -0.5), current_steps * pow(args['WARMUP_STEP'], -1.5))
            current_steps += 1


            # Rotor acc
            outputs_indices = torch.argmax(pred, dim=-1) # -> [rotor, seq, batch]
            for rotor in range(outputs_indices.shape[0]):
                mask = outputs_indices[rotor][~masks.T] == targets[rotor][~masks.T]
                true_positive += mask.sum()
                samples += math.prod(mask.shape)

            # Fur experiment acc
            for idx, pred in enumerate(fur_pred):
                mask = torch.argmax(pred, dim=1) == further_experiment_targets[idx]
                Fur_acc[idx][0] += mask.sum()
                Fur_acc[idx][1] += math.prod(mask.shape)

            loss_sum += loss.item()


            # Bar
            bar.set_description(f"Epoch[{epoch + initial_epoch + 1}/{args['EPOCHS']}]") if args['PROGRESS_BAR'] == 1 else None
            bar.set_postfix(lr=optimizer.state_dict()['param_groups'][0]['lr'], steps=current_steps) if args['PROGRESS_BAR'] == 1 else None

            # Logging per iter
            logger.update('loss_batch', loss.item())
            logger.update('lr', optimizer.state_dict()['param_groups'][0]['lr'])


        # Testing
        # pass


        # Logging further experiment result
        for idx, result in enumerate(Fur_acc):
            logger.update(metric_tags_fur[idx]+' Accuracy', result[0] / result[1])

        # compute the accuracy and print out the result
        acc_train = true_positive / samples
        logger.update('acc_train', acc_train)
        logger.update('loss_avg_train', loss_sum)

        print('\n===============')
        print(f"Epoch: {epoch + initial_epoch + 1} \n"
              f"current step: {current_steps}\n"
              f"Acc_train: {acc_train}")
        for idx, result in enumerate(Fur_acc):
            print(f'Acc_{metric_tags_fur[idx]}: {result[0] / result[1]} ')
        print(f'Loss_avg: {loss_sum}')
        print('===============')

        # Clear the record for the next epoch
        true_positive = 0
        samples = 0
        loss_sum = 0
        Fur_acc = [[0, 0] for _ in range(len(metric_tags_fur))]

        # Saving checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            mix_scaler=mix_scaler,
            current_steps=current_steps,
            current_epochs = epoch + initial_epoch,
            args=args,
            filename=f"{args['TYPE']}_ckpt.pt",
            info=f"Epoch: {epoch + initial_epoch + 1}"
        )

    return model, optimizer

def test(model, dataset, critic, epoch, args):
    '''
    Calculate the accuracy and print some example

    :param model:
    :param dataloader:
    :param critic:
    :param args:
    :return:
    '''

    # Change the dataset and model into test mode
    # Dataset would gaves less amount of sample when testing
    model.eval()
    dataset.setMode(mode='test')

    # Tracking process
    true_positive = 0
    samples = 0
    loss_sum = 0

    # Training loop
    dataloader = DataLoader(
        dataset, batch_size=args['BATCH_SIZE'], shuffle=False, collate_fn=dataset.collate_fn_padding, drop_last=False
    )
    for inputs, targets in dataloader:
        # inputs = inputs.to(args['DEVICE'])
        # targets = targets.to(args['DEVICE'])
        inputs, targets = inputs.to(args['DEVICE']), targets.to(args['DEVICE'])

        # Compute loss
        with torch.cuda.amp.autocast():
            loss, pred_targets = critic(model, inputs, targets)

        # Compute metrics
        outputs_indices = torch.argmax(pred_targets, dim=1)
        mask = outputs_indices == targets

        # Track performance
        true_positive += mask.sum()
        samples += math.prod(mask.shape)
        loss_sum += loss.item()

    # print example
    inputs, targets = dataset.getsample()
    inputs, targets = inputs.to(args['DEVICE']), targets.to(args['DEVICE'])

    _, sample_pred = critic(model, inputs, targets)
    # sample_pred = torch.argmax(sample_pred, dim=-1)
    # Transfer tensor to string
    sample_input_str = ''.join([dataset.indice_to_char[int(index)] for index in inputs.squeeze(0)])
    # sample_decinput_str = ''.join([dataset.indice_to_char[int(index)] for index in targets.squeeze(0)])

    if args['TYPE'] == 'Encoder':
        sample_pred = torch.argmax(sample_pred, dim=1)
        sample_target_str = ''.join([dataset.indice_to_char[int(index)] for index in targets.squeeze(0)])
        sample_pred_str = ''.join([dataset.indice_to_char[int(index)] for index in sample_pred.squeeze(0)])
    elif args['TYPE'] == 'CP2K' or 'CP2K_RNN':
        sample_pred = torch.argmax(sample_pred, dim=1)

        sample_target_str = ''.join([dataset.indice_to_char[int(index)] for index in targets.squeeze(0)])
        sample_pred_str = ''.join([dataset.indice_to_char[int(index)] for index in sample_pred.squeeze(0)])
        # sample_target_str = int(targets.squeeze(0))
        # sample_pred_str = int(sample_pred.squeeze(0))
        #
        # sample_target_str = ''.join([dataset.indice_to_char[int(char)] for char in dataset.initial_state_dict[sample_target_str]])
        # sample_pred_str = ''.join([dataset.indice_to_char[int(char)] for char in dataset.initial_state_dict[sample_pred_str]])



    acc = true_positive / samples
    loss_sum /= mask.shape[0]

    #
    model.train()
    dataset.setMode(mode='train')


    print('\n===============')
    print(f"Testing...\n"
          f"Input: {sample_input_str} \n"
          f"Ground truth: {sample_target_str} \n"
          f"Prediction:   {sample_pred_str} \n"
          f"Acc: {acc} \nLoss_avg: {loss_sum}")
    print('===============')
    return acc, loss_sum




if __name__ == "__main__":
    pred_targets = torch.randn(size=(256,512,47))
    targets = torch.randint(low=0, high=26, size=(256, 47))

    mask = torch.zeros(size=[47])
    mask[7:30] = 1# Lower triangular matrix
    mask = mask.bool()
    # mask = mask.masked_fill(mask == 0, -9e3)  # Convert zeros to -inf
    # mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
    print(mask)
    print(pred_targets[..., mask].shape)
    print(targets[..., mask].shape)

    print(torch.mean(pred_targets, dim=0).shape)

