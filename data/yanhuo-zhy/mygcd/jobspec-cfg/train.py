import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import MyModel, DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups, unsupervised_contrastive_loss, supervised_contrastive_loss
from vision_transformer import vit_base
from scipy.optimize import linear_sum_assignment as linear_assignment

# def scale_factor(epoch):
#     return (0.5)**epoch +1

# def scale_factor(epoch):
#     return math.floor((0.7)**epoch * 40 + 10)

# def scale_factor1(epoch):
#     if epoch < 50:
#         num_values_to_replace = 1 + (epoch * 19) // 49  # Linearly increase from 1 to 20 over 50 epochs
#     else:
#         num_values_to_replace = 20
#     return num_values_to_replace

# def scale_factor2(epoch):
#     if epoch < 50:
#         num_values_to_replace = 1 + (epoch * 9) // 49  # Linearly increase from 1 to 20 over 50 epochs
#     else:
#         num_values_to_replace = 10
#     return num_values_to_replace 

def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )


    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # # inductive
    # best_test_acc_lab = 0
    # # transductive
    # best_train_acc_lab = 0
    # best_train_acc_ubl = 0 
    # best_train_acc_all = 0
    # mymodel = MyModel(200, 256, 100).to('cuda')
    # feature_model = MyModel(num_classes=args.mlp_out_dim, feature_dim=256,K=len(train_loader.dataset)//args.batch_size*args.batch_size).to(args.device)
    
    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        student.train()
        # proj, labels, mask, pseduo = [], [], [], []
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out = student(images)
                teacher_out = student_out.detach()
                ######
                # proj.append(student_proj.chunk(2)[0])
                # labels.append(class_labels)
                # mask.append(mask_lab)
                # pseduo.append((student_out / 0.1).chunk(2)[0])
                # #####

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                # cluster_loss = cluster_criterion(student_out.chunk(2)[0], epoch)
                
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss
                # else:

                # _size = teacher_out.size(0)
                # weights = torch.cat([torch.ones(100), torch.ones(100) * scale_factor(epoch+1)], dim=0)
                # weights = weights.unsqueeze(0).expand(_size, -1).to(args.device)
                # # 对teacher_logits进行加权
                # weighted_logits = teacher_out * weights
                # cluster_loss = cluster_criterion(student_out, weighted_logits, epoch)
                # avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                # me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                # cluster_loss += args.memax_weight * me_max_loss

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                sup_con_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                sup_con_proj = torch.nn.functional.normalize(sup_con_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(sup_con_proj, labels=sup_con_labels)

                #-----------------------------------Mine Representation Learning Loss----------------------------------#
                # # Calculate self-supervised contrastive loss for all the samples
                # contrastive_loss = unsupervised_contrastive_loss(features=student_proj, device=args.device)

                # # Calculate supervised contrastive loss for the labeled samples
                # sup_con_proj = student_proj.chunk(2)[0][mask_lab]
                # sup_con_proj = torch.nn.functional.normalize(sup_con_proj, dim=-1)
                # sup_con_labels = class_labels[mask_lab]
                # sup_con_loss = supervised_contrastive_loss(sup_con_proj, sup_con_labels, device=args.device)

                #------------------------------------negative loss---------------------------------------------------#
                # _, max_indices = torch.max(student_out, dim=1)
                # # 创建一个和student_out形状相同的全零张量negative_label
                # # negative_label = torch.zeros_like(student_out)
                # mask = torch.zeros_like(student_out, dtype=torch.bool)
                # # 根据每一行的最大值所在位置来动态地确定k，并获取每一行的最小k个值的索引
                # for i, max_index in enumerate(max_indices):
                #     if max_index < 100:
                #         k = scale_factor1(epoch)
                #     else:
                #         k = scale_factor2(epoch)
                #     _, min_indices = torch.topk(student_out[i], k, largest=False)
                #     # negative_label[i, min_indices] = 1
                #     mask[i, min_indices] = True
                # # 计算交叉熵损失
                # selected_elements = (1 - student_out)[mask]
                # # print(mask)
                # # print(selected_elements)
                # # s = torch.sigmoid(selected_elements)
                # # print(s)

                # loss_fn = nn.BCEWithLogitsLoss()
                # neg_loss = loss_fn(selected_elements, torch.ones_like(selected_elements).float())

                # ------------------------------------SimMatch loss---------------------------------------------------#
                # u_bank   = feature_model.u_bank.clone().detach()
                # u_labels = feature_model.u_labels.clone().detach()
            
                # feature_w, feature_s = student_proj.chunk(2)[0], student_proj.chunk(2)[1]
                # logits_w, logits_s = student_out.chunk(2)[0], student_out.chunk(2)[1]
                # prob_w = F.softmax(logits_w, dim=1)

                # simmatrix_w = feature_w @ u_bank.T
                # relation_w = F.softmax(simmatrix_w / 0.1, dim=-1)
                # simmatrix_s = feature_s @ u_bank.T
                # relation_s = F.softmax(simmatrix_s / 0.1, dim=-1)
                # # mask for old
                # # mask_for_old = class_labels < 98

                # # ee_loss = F.kl_div(relation_s.log(), relation_w, reduction='batchmean')
                # ee_loss = torch.sum(-relation_s.log() * relation_w.detach(), dim=1).mean()
                # # ee_loss = F.kl_div(relation_s.log(), relation_w, reduction='batchmean')
                # # kl_div_w_s = F.kl_div(relation_w.log(), relation_s, reduction='batchmean')
                # # ee_loss = 0.5 * (kl_div_s_w + kl_div_w_s)

                # # mask_for_new = class_labels >= 98
                # # _, max_indices = torch.max(logits_s, dim=1)
                # # mask_for_new = max_indices >= 98
                # nn_qu = relation_s @ u_labels
                # ne_loss = torch.sum(-nn_qu.log() * prob_w.detach(), dim=1).mean()

                # feature_model.update_unlabel_bank(feature_w, prob_w)

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                # pstr += f'ee_loss: {ee_loss.item():.4f} '
                # pstr += f'ne_loss: {ne_loss.item():.4f} '

                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                # if epoch > 0:
                    # loss += ee_loss
                    # loss += ne_loss
                    # loss += (1 - args.sup_weight) * ne_loss + args.sup_weight * ee_loss
                # if epoch > 0:
                #     regular_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                #     regularloss = mymodel.compute_loss(regular_proj, student_out/0.1, scale_factor(epoch))
                #     loss += regularloss
                #     pstr += f'regularloss: {regularloss.item():.4f} '
                
            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        ####
        # proj = torch.cat(proj,dim=0)
        # labels = torch.cat(labels,dim=0)
        # mask = torch.cat(mask,dim=0)
        # pseduo = torch.cat(pseduo,dim=0)
        # mymodel.reset_feature_bank()
        # mymodel.update_feature_bank(proj.detach(), labels, mask, pseduo)
        # print(mymodel.feature_counts)
        ####
        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        # args.logger.info('Testing on disjoint test set...')
        # all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)


        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        # args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

        # all_acc, old_acc, new_acc, pstr_old, pstr_new, rank_95_old, rank_95_new = test(student, unlabelled_train_loader)

        # args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        # args.logger.info('Old Distribution & Rank-95: {} | Rank-95: {:.2f}'.format(pstr_old, rank_95_old))
        # args.logger.info('New Distribution & Rank-95: {} | Rank-95: {:.2f}'.format(pstr_new, rank_95_new))

        # Step schedule
        exp_lr_scheduler.step()

        # save_dict = {
        #     'model': student.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'epoch': epoch + 1,
        # }

        # torch.save(save_dict, args.model_path)
        # args.logger.info("model saved to {}.".format(args.model_path))

        # if old_acc_test > best_test_acc_lab:
        #     
        #     args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
        #     args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        #     
        #     torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
        #     args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
        #     
        #     # inductive
        #     best_test_acc_lab = old_acc_test
        #     # transductive            
        #     best_train_acc_lab = old_acc
        #     best_train_acc_ubl = new_acc
        #     best_train_acc_all = all_acc
        # 
        # args.logger.info(f'Exp Name: {args.exp_name}')
        # args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')


def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc

# def mysoftmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)
# # test2.0
# def get_image_info(image_id, preds, targets, ind_map, logits_list):
#     # Get the correct label and predicted label for the given image_id
#     correct_label = targets[image_id]
#     predicted_label = preds[image_id]
#     logits = logits_list[image_id]

#     # Get the mapping from ind_map
#     mapped_predicted_label = ind_map[predicted_label]

#     # Check if the mapped prediction is correct
#     is_correct = correct_label == mapped_predicted_label

#     # Calculate the proportion of the mapped_predicted_label in logits
#     # mapped_predicted_label_proportion = logits[mapped_predicted_label] / logits.sum()
#     logits_softmax = mysoftmax(logits)
#     mapped_predicted_label_proportion = logits_softmax[mapped_predicted_label] / logits_softmax.sum()

#     # Get the rank of the mapped_predicted_label in logits
#     rank_dict = {label: rank for rank, label in enumerate(logits.argsort()[::-1])}
#     mapped_predicted_label_rank = rank_dict[mapped_predicted_label]

#     return {
#         "correct_label": correct_label,
#         "predicted_label": predicted_label,
#         "mapped_predicted_label": mapped_predicted_label,
#         "is_correct": is_correct,
#         "mapped_predicted_label_proportion": mapped_predicted_label_proportion,
#         "mapped_predicted_label_rank": mapped_predicted_label_rank
#     }

# def test(model, test_loader):

#     model.eval()

#     preds, targets, logits_list = [], [], []
#     mask = np.array([])
#     for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
#         images = images.cuda(non_blocking=True)
#         with torch.no_grad():
#             _, logits = model(images)
#             logits_list.append(logits.cpu().numpy())
#             preds.append(logits.argmax(1).cpu().numpy())
#             targets.append(label.cpu().numpy())
#             mask = np.append(mask, np.array([True if x.item() in range(100) else False for x in label]))

#     logits_list = np.concatenate(logits_list, axis=0)
#     preds = np.concatenate(preds)
#     targets = np.concatenate(targets)
    
#     # 预测精度
#     mask = mask.astype(bool)
#     targets = targets.astype(int)
#     preds = preds.astype(int)

#     old_classes_gt = set(targets[mask])
#     new_classes_gt = set(targets[~mask])

#     assert preds.size == targets.size
#     D = max(preds.max(), targets.max()) + 1
#     w = np.zeros((D, D), dtype=int)
#     for i in range(preds.size):
#         w[preds[i], targets[i]] += 1

#     ind = linear_assignment(w.max() - w)
#     ind = np.vstack(ind).T

#     ind_map = {j: i for i, j in ind}
#     ind_match = {j: i for i, j in ind}
#     total_acc = sum([w[i, j] for i, j in ind])
#     total_instances = preds.size

#     total_acc /= total_instances

#     old_acc = 0
#     total_old_instances = 0
#     for i in old_classes_gt:
#         old_acc += w[ind_map[i], i]
#         total_old_instances += sum(w[:, i])
#     old_acc /= total_old_instances

#     new_acc = 0
#     total_new_instances = 0
#     for i in new_classes_gt:
#         new_acc += w[ind_map[i], i]
#         total_new_instances += sum(w[:, i])
#     new_acc /= total_new_instances

#     # 打印预测错误样本真实标签的信息
#     incorrect_ranks_new = []
#     incorrect_ranks_old = []

#     for image_id in range(len(targets)):
#         info = get_image_info(image_id, preds, targets, ind_match, logits_list)
#         if not info["is_correct"]:
#             if targets[image_id] > 99:
#                 incorrect_ranks_new.append(info["mapped_predicted_label_rank"])
#             else:
#                 incorrect_ranks_old.append(info["mapped_predicted_label_rank"])

#     def get_proportions(ranks):
#         counts = [
#             sum(1 for rank in ranks if rank < threshold) for threshold in [10, 20, 50, 100, 150, 180]
#         ]
#         total = len(ranks)
#         return [count / total for count in counts]

#     proportions_new = get_proportions(incorrect_ranks_new)
#     proportions_old = get_proportions(incorrect_ranks_old)

#     pstr_new = 'New:'.join([f'<{threshold}: {prop:.4f} ' for threshold, prop in zip([10, 20, 50, 100, 150, 180], proportions_new)])
#     pstr_old = 'Old:'.join([f'<{threshold}: {prop:.4f} ' for threshold, prop in zip([10, 20, 50, 100, 150, 180], proportions_old)])

#     # Calculate rank_95
#     # all_incorrect_ranks = incorrect_ranks_new + incorrect_ranks_old
#     # rank_95 = np.percentile(all_incorrect_ranks, 95)
#     rank_95_old = np.percentile(incorrect_ranks_old, 95)
#     rank_95_new = np.percentile(incorrect_ranks_new, 95)

#     return total_acc, old_acc, new_acc, pstr_old, pstr_new, rank_95_old, rank_95_new



# class MyModel(torch.nn.Module):
#     def __init__(self, class_num, feature_size, max_features_per_class):
#         super(MyModel, self).__init__()
#         self.class_num = class_num
#         self.feature_size = feature_size
#         self.max_features_per_class = max_features_per_class
        
#         # 使用buffer来保存特征和类别计数
#         self.register_buffer('feature_bank', torch.zeros(class_num, max_features_per_class, feature_size))
#         self.register_buffer('feature_counts', torch.zeros(class_num, dtype=torch.long))

#     def reset_feature_bank(self):
#         """重置特征库在每个epoch开始时调用"""
#         self.feature_bank.zero_()
#         self.feature_counts.zero_()

#     def update_feature_bank(self, proj, label, mask, pseudo_label):
#         # 对于有标签的数据
#         labeled_features = proj[mask]
#         labeled_labels = label[mask]
#         for feat, lbl in zip(labeled_features, labeled_labels):
#             if self.feature_counts[lbl] < self.max_features_per_class:
#                 self.feature_bank[lbl, self.feature_counts[lbl]] = feat
#                 self.feature_counts[lbl] += 1

#         # 对于无标签的数据
#         unlabeled_features = proj[~mask]
#         probs = F.softmax(pseudo_label[~mask], dim=1)
#         max_probs, predictions = probs.max(1)
        
#         # 获取后100类
#         valid_classes = list(range(self.class_num - 100, self.class_num))
        
#         for feat, prob, pred in zip(unlabeled_features, max_probs, predictions):
#             if prob > 0.7 and pred.item() in valid_classes:
#                 if self.feature_counts[pred] < self.max_features_per_class:
#                     self.feature_bank[pred, self.feature_counts[pred]] = feat
#                     self.feature_counts[pred] += 1
                    
#     # def compute_loss(self, proj, pseudo_label, k):
#     #     _, class_indices = torch.topk(pseudo_label, k, largest=False, dim=1)
#     #     batch_size = proj.size(0)
#     #     losses = torch.zeros(batch_size).to(proj.device)

#     #     for i in range(batch_size):
#     #         distances = []
#     #         for class_idx in class_indices[i]:
#     #             class_count = self.feature_counts[class_idx].item()
#     #             if class_count > 0:
#     #                 class_features = self.feature_bank[class_idx, :class_count]
#     #                 distance_matrix = torch.sqrt((proj[i] - class_features).pow(2)).sum(dim=1)
#     #                 distances.append(distance_matrix)

#     #         if distances:
#     #             total_distances = torch.cat(distances)
#     #             loss_for_i = (1.0 / (total_distances + 1e-8)).sum()
#     #             losses[i] = loss_for_i

#     #     return losses.mean()
#     def compute_loss(self, proj, pseudo_label, k):
#         _, class_indices = torch.topk(pseudo_label, k, largest=False, dim=1)
#         batch_size = proj.size(0)

#         # Create a tensor to hold all distances
#         all_distances = torch.zeros(batch_size, k).to(proj.device)

#         for idx, class_set in enumerate(class_indices):
#             class_counts = self.feature_counts[class_set]
#             valid_mask = class_counts > 0
#             valid_classes = class_set[valid_mask]
            
#             # Initialize a tensor to hold distances for the current item in the batch across all valid classes
#             distances_for_idx = torch.zeros(len(valid_classes)).to(proj.device)
            
#             for j, valid_class in enumerate(valid_classes):
#                 # Extract features for the valid class based on class_counts
#                 valid_class_features = self.feature_bank[valid_class, :self.feature_counts[valid_class]]
                
#                 # Expand dimensions for broadcasting
#                 expanded_proj = proj[idx].unsqueeze(0).expand_as(valid_class_features)

#                 distance = torch.sqrt((expanded_proj - valid_class_features).pow(2).sum(dim=-1))
#                 distances_for_idx[j] = distance.sum()

#             all_distances[idx, valid_mask] = distances_for_idx

#         # Calculate loss
#         losses = (1.0 / (all_distances.sum(dim=-1) + 1e-8)).sum()
        
#         return losses / batch_size



# def test(model, test_loader, epoch, save_name, args):
#     model.eval()

#     preds, targets = [], []
#     mask = np.array([])

#     # 打开log.txt文件以追加模式
#     with open('/wang_hp/zhy/SimGCD-main/test/novel_sharpen_log1.txt', 'a') as f:
#         # 写入当前epoch的分隔行
#         f.write(f"****epoch {epoch}*******\n")
#         # 写入标题行
#         f.write("True Label, Predicted Label, Prediction Value, Max Old Class Value, Max New Class Value, Ratio (Old/New)\n")

#         for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
#             images = images.cuda(non_blocking=True)
#             with torch.no_grad():
#                 _, logits = model(images)

#                 # 在此处写入每个样本的信息
#                 for i in range(logits.size(0)):
#                     true_label = label[i].item()
#                     predicted_label = logits[i].argmax().item()
#                     prediction_value = logits[i][predicted_label].item()

#                     # 获取旧类和新类的最大预测值
#                     max_old_class_value = logits[i][:100].max().item()
#                     max_new_class_value = logits[i][100:].max().item()

#                     # 计算旧类和新类的最大预测值的比值
#                     ratio = max_old_class_value / max_new_class_value

#                     # 写入每个样本的信息
#                     f.write(f"{true_label}, {predicted_label}, {prediction_value:.4f}, {max_old_class_value:.4f}, {max_new_class_value:.4f}, {ratio:.4f}\n")

#                 preds.append(logits.argmax(1).cpu().numpy())
#                 targets.append(label.cpu().numpy())
#                 mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

#     preds = np.concatenate(preds)
#     targets = np.concatenate(targets)

#     all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
#                                                     T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
#                                                     args=args)

#     return all_acc, old_acc, new_acc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2b'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=False)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=1, type=int)
    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--seed_num', default=0, type=int)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    setup_seed(args.seed_num)
    device = torch.device('cuda:0')
    args = get_class_splits(args)
    args.device = device

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['baseline'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    # backbone = vit_base()
    # backbone.load_state_dict(torch.load('/wang_hp/zhy/gcd-task/pretrained/DINO/dino_vitbase16_pretrain.pth'))
    
    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
                print(f'Finetuning layer {name}')

    
    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    
    # test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
    #                                   batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    # train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
    # print("len of dataset:",len(train_loader.dataset))
    # print(len(train_loader.dataset)//args.batch_size*args.batch_size)
    train(model, train_loader, None, test_loader_unlabelled, args)
