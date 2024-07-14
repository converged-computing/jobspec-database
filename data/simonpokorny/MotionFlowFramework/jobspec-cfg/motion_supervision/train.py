import glob
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from tqdm import tqdm
from FastFlow3D.models import FastFlow3DModelScatter
from FastFlow3D.data.util import ApplyPillarization, custom_collate_batch
from FastFlow3D.utils.pillars import remove_out_of_bounds_points

from datasets.argoverse.argoverse2 import Argoverse2_Sequence
from pytorch3d.ops.knn import knn_points

from motion_supervision.dataset import SceneFlowLoader
from motion_supervision.calculate_metric import Motion_Metric

def get_device_idx_for_port():  # TODO not for multiple gpus
    gpu_txt = open('/home/vacekpa2/gpu.txt', 'r').readlines()
    os.system('nvidia-smi -L > /home/vacekpa2/gpu_all.txt')

    time.sleep(0.1)
    gpu_all_txt = open('/home/vacekpa2/gpu_all.txt', 'r').readlines()

    gpu_all_txt = [text[7:] for text in gpu_all_txt]
    device_idx = 0
    for idx, gpu_id in enumerate(gpu_all_txt):
        if gpu_txt[0][7:] == gpu_id:
            device_idx = idx

    return device_idx

def get_device():
    if torch.cuda.is_available():
        device_idx = get_device_idx_for_port()
        device = torch.device(device_idx)
    else:
        device = torch.device('cpu')
    return device

def print_gpu_memory():
    if torch.cuda.is_available():
        free_memory = torch.cuda.mem_get_info()[0] / 1024 / 1024
        max_memory = torch.cuda.mem_get_info()[1] / 1024 / 1024
        memory_consumed = max_memory - free_memory
        print(f"Memory consumption: {memory_consumed:.0f} MB")

def store_batch_data(prev_pts, curr_pts, flow, mos, ego_label, loss, data_dir=f'{os.path.expanduser("~")}/data/toy/'):
    for i in range(len(prev_pts)):
        np.save(f'{data_dir}/prev_pts_{i:06d}.npy', prev_pts[i].cpu().detach().numpy())
        np.save(f'{data_dir}/curr_pts_{i:06d}.npy', curr_pts[i].cpu().detach().numpy())
        np.save(f'{data_dir}/flow_{i:06d}.npy', flow[i].cpu().detach().numpy())
        np.save(f'{data_dir}/mos_{i:06d}.npy', mos[i].cpu().detach().numpy())
        np.save(f'{data_dir}/ego_{i:06d}.npy', ego_label[i].cpu().detach().numpy())
        np.save(f'{data_dir}/loss_{i:06d}.npy', loss[i].cpu().detach().numpy())
        # print('storing only first sample from batch')
        break
    # store it inteligently

def pad_prev_and_curr(prev, flow, curr):

    max_pts_p = np.argmax((prev.shape[1], curr.shape[1]))
    N_pts_x = prev.shape[1]

    if max_pts_p == 0:
        N_pad = prev.shape[1] - curr.shape[1]
        curr = torch.nn.functional.pad(curr, (0,0,0,N_pad,0,0)) # padding with 0

    else:
        N_pad = curr.shape[1] - prev.shape[1]
        prev = torch.nn.functional.pad(prev, (0,0,0,N_pad,0,0))
        flow = torch.nn.functional.pad(flow, (0,0,0,N_pad,0,0))

    return prev, flow, curr

def NN_loss(x, y, x_lengths=None, y_lengths=None, reduction='mean'):

    # wrapper for different lengths --- NEDELAT!
    # max_pts_p = np.argmax((x.shape[1], y.shape[1]))
    # N_pts_x = x.shape[1]

    # if max_pts_p == 0:
    #     N_pad = x.shape[1] - y.shape[1]
    #     y = torch.nn.functional.pad(y, (0,0,0,N_pad,0,0))

    # else:
    #     N_pad = y.shape[1] - x.shape[1]
    #     x = torch.nn.functional.pad(y, (0,0,0,N_pad,0,0))


    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1, norm=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1, norm=1)

    # hack, maybe can be done better
    # if max_pts_p == 0:

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    nearest_to_y = x_nn[1]

    # else:
    #     cham_x = x_nn.dists[:N_pts_x, 0]  # (N, P1)
        # cham_y = y_nn.dists[:N_pts_x, 0]  # (N, P2)
        #
        # nearest_to_y = x_nn[1][:,N_pts_x]

    nn_loss = (cham_x + cham_y) / 2

    if reduction == 'mean':
        nn_loss = nn_loss.mean()
    elif reduction == 'sum':
        nn_loss = nn_loss.sum()
    elif reduction == 'none':
        nn_loss = nn_loss
    else:
        raise NotImplementedError

    # breakpoint()
    return nn_loss, nearest_to_y


def ego_loss(mos, ego_label):
    # Ego label
    mos = (mos - mos.min()) / (mos.max() - mos.min())  # normalize , probably keep the sigmoid function
    # mos = mos * 10 - 5  # for sigmoid range, is it correct?
    # I can normalize to -6 and 6 and then use sigmoid for 0-1 range
    mos = torch.sigmoid(mos)  # sigmoid  # this will make the 0.5 values on "zeros in visualization"
    # todo important - add weights for that
    nbr_dyn = (ego_label == 1).sum()
    nbr_stat = (ego_label == 0).sum()
    nbr_ego = nbr_dyn + nbr_stat

    ego_dynamic_loss = - torch.log(mos[ego_label == 1]).mean() if (ego_label == 1).sum() > 0 else 0
    ego_static_loss = - torch.log(1 - mos[ego_label == 0]).mean() if (ego_label == 0).sum() > 0 else 0

    MOS_loss = (nbr_stat / nbr_ego) * ego_dynamic_loss + (nbr_dyn / nbr_ego) * ego_static_loss  # dynamic from ego and static from ego road 1/0, mean reduction

    return MOS_loss

class FocalLoss_Image(nn.Module):
    def __init__(self, gamma=2, ce_weights=(1,1), reduction='mean'):
        super(FocalLoss_Image, self).__init__()
        self.gamma = gamma
        self.ce_weights = ce_weights
        self.reduction = reduction

        self.CE = nn.CrossEntropyLoss(weight=torch.tensor(ce_weights), ignore_index=-1, reduction='none')

    def forward(self, logits, target):
        # Logits: B, N, C, but to CrossEntropy it needs to be B, C, N
        # Target: B, N
        CE_loss = self.CE(logits, target)
        logits_soft = F.log_softmax(logits, dim=1)

        max_logits = torch.max(logits_soft, dim=1)[0]    # values, CE loss should be 0 on -1 index and therefore cancels this

        loss = (1 - max_logits) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == "none":
            return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, ce_weights=(1, 1), reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce_weights = ce_weights
        self.reduction = reduction

        self.CE = nn.CrossEntropyLoss(weight=torch.tensor(ce_weights), ignore_index=-1, reduction='none')

    def forward(self, logits, target):
        # Logits: B, N, C, but to CrossEntropy it needs to be B, C, N
        # Target: B, N
        # for speed, there can be only one softmax I guess
        logits = logits.permute(0, 2, 1)

        CE_loss = self.CE(logits, target)

        logits = F.log_softmax(logits, dim=1)

        pt = logits.permute(0, 2, 1)

        pt = pt.flatten(start_dim=0, end_dim=1)
        target_gather = target.flatten()

        ignore_index = -1
        valid_mask = target_gather != ignore_index
        valid_target = target_gather[valid_mask]
        valid_pt = pt[valid_mask]
        CE_loss = CE_loss.flatten()[valid_mask]

        valid_target = valid_target.tile(2,1).permute(1,0)    # get the same shape as pt
        only_probs_as_target = torch.gather(valid_pt, 1, valid_target)[:,0]

        loss = (1 - only_probs_as_target) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == "none":
            return loss

def get_real_lengths_in_batch(prev_pts, curr_pts):
    # This can be done by storing the lengths after padding to speed up
    # flow_non_zero = [flow[i][curr_pts[i].abs().sum(dim=1).bool()] for i in range(len(curr_pts))]
    prev_list = [pts[pts.abs().sum(dim=1).bool()] for pts in prev_pts]
    curr_list = [pts[pts.abs().sum(dim=1).bool()] for pts in curr_pts]
    # nbr_pts = np.min([pts.shape[0] for pts in prev_list] + [pts.shape[0] for pts in curr_list])# can be done beforehand

    x_lengths = torch.tensor([pts.shape[0] for pts in prev_list], dtype=torch.long)
    y_lengths = torch.tensor([pts.shape[0] for pts in curr_list], dtype=torch.long)

    if torch.cuda.is_available():
        x_lengths = x_lengths.cuda()
        y_lengths = y_lengths.cuda()

    return x_lengths, y_lengths


def cycle_preproces_data(batch, flow, dataset):
    # first eliminate out of boundaries
    prev_pts = batch[0][0]
    prev_pts = torch.cat((prev_pts[..., :3] + prev_pts[..., 3:6] + flow.cpu(), prev_pts[..., 6:8]), dim=2)

    x_1 = prev_pts.clone()
    x_1[..., :3] = prev_pts[..., :3] + flow.cpu()


    list_to_batch = []

    label_grid = - np.ones((cfg['grid_size'], cfg['grid_size']), dtype=int)

    for bs in range(len(prev_pts)):
        # print(bs)
        pts, mask1 = remove_out_of_bounds_points(x_1[bs], dataset.cfg['x_min'] + 0.2, dataset.cfg['x_max'] - 0.2,
                                              dataset.cfg['y_min'] + 0.2,
                                              dataset.cfg['y_max'] - 0.2,
                                              dataset.cfg['z_min'], dataset.cfg['z_max'])

        reverse_pts1, grid1 = dataset.pilarization(x_1[bs, mask1].detach().cpu().numpy())
        original_pts1, grid2 = dataset.pilarization(prev_pts[bs, mask1].detach().cpu().numpy()) # this might not be exactly the point cloud

        ego_label = - np.ones(reverse_pts1.shape[0])


        prev_batch = (reverse_pts1, grid1, mask1, ego_label, label_grid)
        current_batch = (original_pts1, grid2, mask1, ego_label, label_grid)

        list_to_batch.append((prev_batch, current_batch))

    out_batch = custom_collate_batch(list_to_batch)

    return out_batch

def fastflow_inference(model, val_dataloader, save_every=10):

    metric = Motion_Metric()
    model = model.eval()
    with torch.no_grad():
        for val_idx, batch in enumerate(val_dataloader):
            prev, curr = batch

            flow, mos = model(batch)

            # print(f'Epoch: {epoch:03d} Iter: {val_idx}/{max_val_iter}')

            predicted_mos_labels = torch.argmax(mos, dim=1)
            ground_truth_mos = curr[4]

            metric.update(ground_truth_mos.flatten(), predicted_mos_labels.flatten())

            mov_prec, mov_recall, mov_iou = metric.get_moving_stats()

            print(f"Moving Stats [%] ---> Precision: {mov_prec * 100:.2f} \t Recall: {mov_recall * 100:.2f} \t IoU: {mov_iou * 100:.2f}")

            mos = torch.softmax(mos, dim=1) # just for visul
            confidence, mos_prediction = torch.max(mos, dim=1)
            curr_pts = (curr[0][...,:3] + curr[0][...,3:6]) # in future solve inconsistent sending to cuda

            if save_every is not None:

                if val_idx % save_every == 0:

                    for i in range(len(mos)):
                        cell_size = np.abs(2 * cfg['x_min'] / cfg['grid_size'])
                        label_ind = ((curr_pts[i, :, :2].detach().cpu().numpy() - (cfg['x_min'], cfg['y_min'])) / cell_size).astype(int)

                        mos_pred = mos_prediction[i, label_ind[:,0], label_ind[:,1]]
                        conf_pred = confidence[i, label_ind[:,0], label_ind[:,1]]
                        static_pred = mos[i, 0, label_ind[:,0], label_ind[:,1]]
                        dynamic_pred = mos[i, 1, label_ind[:,0], label_ind[:,1]]

                        label_pred = ground_truth_mos[i, label_ind[:,0], label_ind[:,1]]

                        frame_idx = curr[5][i]
                        seq_id, frame_id = val_dataloader.dataset.all_indices[frame_idx]

                        vis_pts = torch.cat((curr_pts[i], static_pred[:, None].cpu(), dynamic_pred[:, None].cpu(), conf_pred[:, None].cpu(), mos_pred[:, None].cpu(), label_pred[:, None].cpu()), dim=1)

                        np.save(f"{os.path.expanduser('~')}/data/tmp_vis/{seq_id}_{frame_id:06d}_visul.npy", vis_pts.detach().cpu().numpy())

                        break   # save just one

def save_one_frame(prev_pts, prev_prior, curr_pts, curr_prior, flow, mos, path):
    for i in range(len(prev_pts)):

        data_dict = {'prev_pts' : prev_pts[i].detach().cpu().numpy(),
                     'prev_prior' : prev_prior[i].detach().cpu().numpy(),
                     'curr_pts' : curr_pts[i].detach().cpu().numpy(),
                     'curr_prior' : curr_prior[i].detach().cpu().numpy(),
                     'flow' : flow[i].detach().cpu().numpy(),
                     'mos' : mos[i].detach().cpu().numpy(),
                     }


        np.savez(path, **data_dict)

        break



def validate_fastflow(model, val_dataloader, verbose=True):

    mos_metric = Motion_Metric()
    running_epe = []

    with torch.no_grad():
        # model.eval()  # tmp removed
        model = model.train()

        for val_idx, batch in enumerate(val_dataloader):
            # Unpack
            prev, curr = batch

            # Predictions
            prev_pts, prev_prior, prev_mask = prev[0].cuda(), prev[1].cuda(), prev[2].cuda()
            curr_pts, curr_prior, curr_mask = curr[0].cuda(), curr[1].cuda(), curr[2].cuda()

            prev_batch = flow_dataset.create_pillar_batch_gpu(prev_pts, prev_mask)
            curr_batch = flow_dataset.create_pillar_batch_gpu(curr_pts, curr_mask)

            flow, mos = model(prev_batch, curr_batch)
            predicted_mos_labels = torch.argmax(mos, dim=1).detach().cpu()
            # Labels
            ground_truth_mos = curr[3]
            flow_label = prev[5]

            #metric
            mos_metric.update(ground_truth_mos.flatten(), predicted_mos_labels.flatten())
            epe = eval_flow(flow_label, flow)
            running_epe.append(epe)

            if verbose:
                mov_prec, mov_recall, mov_iou = metric.get_moving_stats()
                print(f"Iter: {val_idx:04d} EPE: {np.mean(running_epe):.3f} Precision: {mov_prec * 100:.2f} \t Recall: {mov_recall * 100:.2f} \t IoU: {mov_iou * 100:.2f}")

        mov_prec, mov_recall, mov_iou = metric.get_moving_stats()
        print(f"Iter: {val_idx:04d} EPE: {np.mean(running_epe):.3f} Precision: {mov_prec * 100:.2f} \t Recall: {mov_recall * 100:.2f} \t IoU: {mov_iou * 100:.2f}")

def construct_batched_cuda_grid(pts, feature, cfg, device):
    '''
    Assumes BS x N x CH (all frames same number of fake pts with zeros in the center)
    :param pts:
    :param feature:
    :param cfg:
    :return:
    '''
    BS = len(pts)
    bs_ind = torch.cat([bs_idx * torch.ones(pts.shape[1], dtype=torch.long, device=device) for bs_idx in range(BS)])


    feature_grid = - torch.ones(BS, cfg['grid_size'], cfg['grid_size'], device=device).long()

    cell_size = np.abs(2 * cfg['x_min'] / cfg['grid_size'])

    coor_shift = torch.tile(torch.tensor((cfg['x_min'], cfg['y_min']), dtype=torch.float, device=device), dims=(BS, 1, 1))

    feature_ind = ((pts[:, :, :2] - coor_shift) / cell_size).long()

    # breakpoint()
    feature_grid[bs_ind, feature_ind.flatten(0,1)[:, 0], feature_ind.flatten(0,1)[:, 1]] = feature.flatten().long()

    return feature_grid

def transfer_from_batched_cuda_grid(pts, feature_grid, cfg, device):

    BS = len(pts)
    bs_ind = torch.cat([bs_idx * torch.ones(pts.shape[1], dtype=torch.long, device=device) for bs_idx in range(BS)])

    cell_size = np.abs(2 * cfg['x_min'] / cfg['grid_size'])

    coor_shift = torch.tile(torch.tensor((cfg['x_min'], cfg['y_min']), dtype=torch.float, device=device),
                            dims=(BS, 1, 1))

    feature_ind = ((pts[:, :, :2] - coor_shift) / cell_size).long()

    feature = feature_grid[bs_ind, feature_ind.flatten(0, 1)[:, 0], feature_ind.flatten(0, 1)[:, 1]]

    feature = feature.reshape(BS, pts.shape[1])

    return feature

class Artificial_label_loss(nn.Module):

    def __init__(self, weights=None):
        super().__init__()

        self.CE = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, p_i, mos, p_j, error_p_i_flow, nearest_flow):

        error_p_i_rigid, nearest_rigid = NN_loss(p_i, p_j, reduction='none')  # flow is 0 after synchronization

        # dynamic 1, static 0
        # if error flow is smaller, the non-zero flow better explains state ---> dynamic
        # todo verify
        dynamic_states = error_p_i_flow > error_p_i_rigid

        # This equation splits the errors to dynamic and static and takes corresponding NN based on which is closer
        # Technical: we zero lower error indices so only one should always be zero and sum it to keep the indices: idx + 0 = idx
        art_labels_indices = nearest_flow[..., 0] * dynamic_states + nearest_rigid[..., 0] * (dynamic_states == False)
        art_labels = dynamic_states.long()
        # project NN and then assign motion states

        # this might slow training down
        p_j_by_nn = torch.stack([p_j[batch_idx, art_labels_indices[batch_idx]] for batch_idx in range(len(p_j))])

        # Project current time frame to grid and annotate it with art_labels from previous coor + flow
        art_label_grid = construct_batched_cuda_grid(p_j_by_nn, art_labels, cfg, device)

        print("Weights not implemented!")
        artificial_label_loss = self.CE(mos, art_label_grid)

        return artificial_label_loss

# next time loop
def eval_flow_ogc(gt_flow, flow_pred, epe_norm_thresh=0.05, eps=1e-10):
    """
    Compute scene flow estimation metrics: EPE3D, Acc3DS, Acc3DR, Outliers3D.
    :param gt_flow: (B, N, 3) torch.Tensor.
    :param flow_pred: (B, N, 3) torch.Tensor.
    :param epe_norm_thresh: Threshold for abstract EPE3D values, used in computing Acc3DS / Acc3DR / Outliers3D and adapted to sizes of different datasets.
    :return:
        epe & acc_strict & acc_relax & outlier: Floats.
    """
    gt_flow = gt_flow.detach().cpu()
    flow_pred = flow_pred.detach().cpu()

    epe_norm = torch.norm(flow_pred - gt_flow, dim=2)
    sf_norm = torch.norm(gt_flow, dim=2)
    relative_err = epe_norm / (sf_norm + eps)
    epe = epe_norm.mean().item()

    # Adjust the threshold to the scale of dataset
    acc_strict = (torch.logical_or(epe_norm < epe_norm_thresh, relative_err < 0.05)).float().mean().item()
    acc_relax = (torch.logical_or(epe_norm < (2 * epe_norm_thresh), relative_err < 0.1)).float().mean().item()
    outlier = (torch.logical_or(epe_norm > (6 * epe_norm_thresh), relative_err > 0.1)).float().mean().item()
    return epe, acc_strict, acc_relax, outlier

def eval_flow(flow_label, flow_pred):

    valid_gt_flow = flow_label[..., 3] != -1

    epe = torch.linalg.norm(flow_label[valid_gt_flow][:, :3].detach().cpu() - flow_pred[valid_gt_flow][:, :3].detach().cpu(), dim=1)
    epe_reduction = epe.mean()

    return epe_reduction

comb_exp = [  # NN, C, ART, P_F, P-Fek, Mag
                [0, 0, 0., 0, 1., 0, 'weight_test', 'waymo_toy', 'waymo_tst'],
                [0, 0, 0., 0, 1., 0, 'weight_test1', 'waymo_toy', 'waymo_tst'],
                [2, 1, 0, 0, 0, 0, 'MITTAL', 'waymo_trn', 'waymo_tst'],
                # [2, 1, 0.1, 0, 1., 0, 'SLIM_2', 'waymo_toy', 'waymo_toy'],
                [2, 1, 0.1, 1, 0.5, 0, 'SLIM_with_prior_1', 'waymo_toy', 'waymo_toy'],
                [2, 1, 0.1, 1, 1, 0, 'SLIM_with_prior_2', 'waymo_toy', 'waymo_toy'],
                [2, 1, 0.1, 0, 0, 0, 'SLIM', 'waymo_trn', 'waymo_toy'],
                [2, 1, 0.1, 1, 1, 0, 'SLIM_with_prior', 'waymo_trn', 'waymo_toy'],


                [2, 1, 0.1, 0, 0, 0, 'SLIM', 'argo2_trn', 'argo2_toy'],
                [2, 1, 0.1, 1, 1, 0, 'SLIM_with_prior', 'argo2_trn', 'argo2_toy'],
                [2, 1, 0, 0, 0, 0, 'MITTAL', 'argo2_trn', 'argo2_toy'],

                [2, 1, 0.1, 0, 0, 0, 'SLIM', 'sk_trn', 'sk_toy'],
                [2, 1, 0.1, 1, 1, 0, 'SLIM_with_prior', 'sk_trn', 'sk_toy'],
                [2, 1, 0, 0, 0, 0, 'MITTAL', 'sk_trn', 'sk_toy'],
                ]

# 'flow from point 1 should not be in freespace of time 2, if inside, increase the distance from it'!!!

if __name__ == "__main__":
    cfg = {'x_max' : 35.0,  # orig waymo 85m
           'x_min' : -35.0,
           'y_max' : 35.0,
           'y_min' : -35.0,
           'z_max' : 3.0,
           'z_min' : 0.3,   # as in slim - remove 30 cm above init ground
           # 'z_min' : -3.0,
           'grid_size' : 512,   # slim 640
           'point_features' : 6,
           'background_weight' : 0.1,
           'learning_rate' : 0.0001,
           'weight_decay' : 0.0000,
           'use_group_norm' : True, ##  # tmp, change to false when training to unlock batch norm
           'BS' : 10,
           'm_thresh' : 0.1,
           }

    cfg['cell_size'] = np.abs(2 * cfg['x_min'] / cfg['grid_size'])

    exp_nbr = sys.argv[1]

    chosen_exp = comb_exp[int(exp_nbr)]
    w_nn = chosen_exp[0]
    w_c = chosen_exp[1]
    w_a = chosen_exp[2]
    w_p = chosen_exp[3]
    w_f = chosen_exp[4]
    w_m = chosen_exp[5]
    exp_name = chosen_exp[6]
    trn_dataset_name = chosen_exp[7]
    val_dataset_name = chosen_exp[8]



    max_epoch = 100
    device = get_device()

    print("RUNNING EXP ----- \n ", chosen_exp)


    model = FastFlow3DModelScatter(n_pillars_x=cfg['grid_size'], n_pillars_y=cfg['grid_size'],
                                   background_weight=cfg['background_weight'], point_features=cfg['point_features'],
                                   learning_rate=cfg['learning_rate'],
                                   use_group_norm=cfg['use_group_norm']).cuda()


    optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

    # EXP NAME
    exp_folder = os.path.expanduser("~") + '/data/fastflow/' + exp_name + '_' + trn_dataset_name

    # Add exp types
    current_runs = sorted(glob.glob(exp_folder + "*"))
    if len(current_runs) > 0:
        exp_folder + str(len(current_runs))

    for fold in [exp_folder + '/model', exp_folder + '/inference']:
        os.makedirs(fold, exist_ok=True)

    # Saving config info
    with open(exp_folder + '/exp_config.txt', 'w') as f:
        f.writelines(str(chosen_exp)[1:-1] + '\n')
        f.close()

    # DATASET
    flow_dataset = SceneFlowLoader(name_of_dataset=trn_dataset_name, cfg=cfg)   # TMP Change
    val_flow_dataset = SceneFlowLoader(name_of_dataset=val_dataset_name, cfg=cfg)

    trn_dataloader = flow_dataset.return_dataloader(batch_size=cfg['BS'], num_workers=0, shuffle=True)
    val_dataloader = val_flow_dataset.return_dataloader(batch_size=cfg['BS'], num_workers=0, shuffle=False)



    max_iter = len(trn_dataloader.dataset)
    max_val_iter = len(val_dataloader.dataset)

    # calculated_class_weights = flow_dataset.calculate_CE_weights()
    # print("Class Weights: ", calculated_class_weights)
    # ce_weights = torch.tensor((0, calculated_class_weights[0], calculated_class_weights[1]), dtype=torch.float,device=device)
    ce_weights = torch.tensor((0., 1, 100.), dtype=torch.float,device=device)   # not used now

    # focal_loss = FocalLoss_Image(gamma=2, ce_weights=ce_weights)     # gamma 0 ---> CE
    CE_mos = torch.nn.CrossEntropyLoss(weight=ce_weights, reduction='none')

    art_loss = Artificial_label_loss()

    metric = Motion_Metric()
    trn_metric = Motion_Metric()

    current_iou = 0
    total_iter = 0
    accum_loss = []

    model = model.cuda()

    error_p_i_flow, artificial_loss, static_flow_loss, dynamic_flow_loss, flow_magnitude_loss, fekal_loss = 0,0,0,0,0,0

    # exp_name = "SLIM_1_waymo_toy"
    # model_paths = sorted(glob.glob(f"{os.path.expanduser('~')}/data/fastflow/{exp_name}/model/*.pth"))

    # last_model_path = model_paths[-1]
    # model.load_state_dict(torch.load(last_model_path, map_location='cpu'))
    # model = model.cuda()
    # validate_fastflow(model, val_dataloader, verbose=True)

    # sys.exit('Only validation')

    for epoch in range(max_epoch):

        model = model.train()
        loss_list = []
        epoch_loss = []

        # validate_fastflow(model, val_dataloader)

        for idx, batch in enumerate(trn_dataloader):

            # flip point cloud here
            temporal_flip = np.random.choice(2,1)

            if temporal_flip and False:
                curr, prev = batch
            else:
                prev, curr = batch

            flow_label = prev[5]

            prev_pts, prev_prior, prev_mask = prev[0].cuda(), prev[1].cuda(), prev[2].cuda()
            curr_pts, curr_prior, curr_mask = curr[0].cuda(), curr[1].cuda(), curr[2].cuda()

            prev_batch = flow_dataset.create_pillar_batch_gpu(prev_pts, prev_mask)
            curr_batch = flow_dataset.create_pillar_batch_gpu(curr_pts, curr_mask)

            flow, mos = model(prev_batch, curr_batch)



            # mos is current and flow is previous to current. As is done in SLIM
            target_grid = curr[3].to(device).long()


            # Art label loss
            p_i = (prev_batch[0][..., :3] + prev_batch[0][..., 3:6])
            p_j = (curr_batch[0][..., :3] + curr_batch[0][..., 3:6])

            epe = eval_flow(flow_label, flow)

            print_str = f"Epoch: {epoch:03d}, EPE: {epe} "

            # prior_mos_j = construct_batched_cuda_grid(p_j, curr_prior, cfg, device)


            if w_f > 0:
                nbr_elements = target_grid.size()[0] * target_grid.size()[1] * target_grid.size()[2]
                grid_fekal_loss = CE_mos(mos, target_grid)
                                # dynamic Weights                                       # Summed loss over all predictions
                dyn_loss = torch.sum(target_grid == 1) / nbr_elements * torch.sum(grid_fekal_loss[target_grid == 2]) / nbr_elements
                stat_loss = torch.sum(target_grid == 2) / nbr_elements * torch.sum(grid_fekal_loss[target_grid == 1]) / nbr_elements

                fekal_loss = dyn_loss + stat_loss
                print_str += f"Focal: {w_f * fekal_loss:.6f}, dyn: {dyn_loss:.4f}, stat: {stat_loss:.4f} "

            if w_m > 0:
                flow_magnitude = torch.linalg.vector_norm(flow, dim=2)
                # print(flow_magnitude)
                flow_magnitude_loss = flow_magnitude[flow_magnitude > 5].mean()  # doplnit progresivni vahu tam kde nechci
                # flow_magnitude_loss = flow_magnitude.mean()
                # print(flow_magnitude, flow_magnitude.shape)
                print_str += f"Magnitude: {w_m * flow_magnitude_loss:.3f}, "

            ### NN loss
            if w_nn > 0:
                error_p_i_flow, nearest_flow = NN_loss(p_i + flow, p_j)
                print_str += f"error NN: {w_nn * error_p_i_flow.item():.3f}, "


            ### Artificial Label Loss
            if w_a > 0:
                artificial_loss = art_loss(p_i, mos, p_j, error_p_i_flow, nearest_flow)
                print_str += f"Art: {w_a * artificial_loss.item():.3f}, " \


            ### Prior Label Loss
            if w_p > 0:
                prior_label = prev[1]

                flow_magnitude = torch.linalg.vector_norm(flow, dim=2)

                Cosine_similarity = torch.nn.CosineSimilarity(dim=2)

                

                static_flow = flow[prev_prior == 0]
                dynamic_flow = flow[(prev_prior == 1) & (flow_magnitude < cfg['m_thresh'])]

                # stat_weight = (len(dynamic_flow) + 1) / (len(dynamic_flow) + len(static_flow))
                # dyn_weight = (len(static_flow) + 1) / (len(dynamic_flow) + len(static_flow))

                dyn_weight = 100 # to get from 0.1 ** 2 to 1
                stat_weight = 0.1 * dyn_weight

                # dim = 1 because it already flattens the vector
                static_flow_loss = stat_weight * torch.linalg.vector_norm(static_flow, dim=1).mean()
                # Maximize dynamic flow of till m_thresh
                dynamic_flow_loss = dyn_weight * ((torch.linalg.vector_norm(dynamic_flow, dim=1) - cfg['m_thresh']) ** 2).mean() # 0, m_thresh

                if len(dynamic_flow) == 0:
                    dynamic_flow_loss = 0   # just for printing zero instead of nan

                print_str += f"S Prior Flow: {w_p * static_flow_loss:.3f}, " \
                             f"D Prior Flow: {w_p * dynamic_flow_loss:.3f}, "


            if w_p > 0 or w_a > 0 or w_nn > 0 or w_m > 0 or w_f > 0:
                loss = w_nn * error_p_i_flow +\
                       w_a * artificial_loss +\
                       w_p * static_flow_loss +\
                       w_p * dynamic_flow_loss +\
                       w_m * flow_magnitude_loss +\
                       w_f * fekal_loss +\
                       torch.linalg.vector_norm(flow[..., 2]).mean()  # min flow_z to zero (maybe helps)

                # print(loss.item())
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            forward_flow = flow.detach()  # to close computational graph before cycle flow


            ### Cycle consistency flow  # separated for larger batch_size in gpu memory
            if w_c > 0:
                with torch.no_grad():
                    # truncated flow!
                    p_i_flow = p_i + forward_flow
                    new_range = torch.sqrt((p_i_flow[..., :3] ** 2).sum(2))
                    p_i_flow[new_range > cfg['x_max']] = 0

                    # p_i_flow = torch.cat((p_i_flow, prev[0][..., -1][..., None].cuda()), dim=2).to(torch.float)  # return intensity

                    N_pts = p_i.shape[1]
                    BS = p_i.shape[0]
                    anchored_p_j = p_j.flatten(0,1)[nearest_flow.flatten()].reshape(BS, N_pts, 3)

                    # Average of anchored NN from next time and p_i with flow
                    cycle_p_ij = (p_i_flow + anchored_p_j) / 2


                    cycle_input = flow_dataset.create_pillar_batch_gpu(cycle_p_ij, prev_mask)


                cycle_flow, cycle_mos = model(cycle_input, prev_batch)

                cycle_error = torch.linalg.vector_norm(p_i_flow[..., :3] + cycle_flow - p_i, dim=2).mean() # This is actually the right way!


                # todo add reconstruction loss in future

                loss = cycle_error
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                print_str += f"Cycle: {w_c * cycle_error.item():.3f}, "


            # print_gpu_memory()

            # epoch_loss.append(loss.item())
            # running_loss = np.mean(epoch_loss)
            # print_str += f'Running loss: {running_loss:.3f}'
            print(print_str)

            p_j_mos = transfer_from_batched_cuda_grid(p_j, torch.argmax(mos, dim=1), cfg, device)
            # p_j_mos = transfer_from_batched_cuda_grid(p_j, label_grid, cfg, device)

            total_iter += 1
            # accum_loss.append(loss.item())

            # Saving
            if total_iter % 1 == 0:
                file_path = exp_folder + f'/inference/{epoch}_{idx:06d}'
                save_one_frame(p_i, prev_prior, p_j, curr_prior, flow, p_j_mos, file_path)
                # save_one_frame(p_i, prev_prior, p_j, curr_prior, flow, torch.argmax(mos, dim=1), file_path)

                model_save_path = f"{exp_folder}/model/{epoch:03d}_{total_iter}_weights.pth"
                torch.save(model.state_dict(), model_save_path)

        validate_fastflow(model, val_dataloader)
        model_save_path = f"{exp_folder}/model/{epoch:03d}_{total_iter}_weights.pth"
        torch.save(model.state_dict(), model_save_path)



        # todo do not calculate chamfer on under height treshold besides the preanotated road?
        # todo chamfer_loss = chamfer_loss[chamfer_loss < chamfer_threshold]
        # todo truncate flow?



        # statistika objektu
        # naucit jen na spravnych dynamickych labelu, pokud je spatne, tak distribuce deleni je spatna

        # 1) vezmu GT a vymaskuju dynamicky i staticky ve svych labelech, pak naucim - odstranim noise
        # 2) Distribuce, histogram
        # 3) Podle semantickych labelu najit distribuci zachycenych seg classu
        # 4) prah segmentaci? - variabilni
        # 5) clustering vic bodu v objektech, oddelit silnici
        # fine-tune model

        # choose frames with dynamic only
        # uci se to samply, dokaze se prefitovat na prior labely
        # dynamicke body se najdou podle prior labelu, staticke take
        # problem je, ze dynamicke body jsou vsude okolo take a staticke body to neregularizuji. Mozna waymo bude davat vetsi smysl

        # z obou duvodu, co rikali kluci to pravdepodobne nejde
        # nefunguje to standalone, ale melo by pomoct pri cele pipeline ---> connect with motion flow
        # Loss Art should have lower weight because we need first the chamfer to assign flow, then use it for class
        # v chamferu by asi nemela byt ta zeme - jestli to nepujde, tak odfiltruj podle z_var_road


 # 'flow from point 1 should not be in freespace of time 2, if inside, increase the distance from it'.
