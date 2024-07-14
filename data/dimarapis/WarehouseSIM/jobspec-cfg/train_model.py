import argparse

import torch.nn as nn
import torch.optim as optim
import torch.utils.data.sampler as sampler
import matplotlib.pyplot as plt

from models.model_ResNet import MTLDeepLabv3, MTANDeepLabv3, ResNetSingle
from models.model_SegNet import SegNetSplit, SegNetMTAN, SegNetSingle
from models.model_DDRNet import DualResNetMTL, BasicBlock, DualResNetSingle
from models.model_GuideDepth import GuideDepth
from models.model_EdgeSegNet import EdgeSegNet


from utils import *
#from dataloader import DecnetDataloader
from tqdm import tqdm
from autolambda_code import SimWarehouse, NYUv2
import visualizer
from omegaconf import OmegaConf
#import segmentation_models_pytorch as smp 
import wandb
import yaml

from torch.utils.data import DataLoader

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def min_max_sanity_check(returned_data_dict):
    #print(returned_data_dict["rgb"].to(device))
    print(f'rgb {torch_min_max(returned_data_dict["rgb"])}')
    print(f'depth {torch_min_max(returned_data_dict["depth"])}')
    print(f'semantic {torch_min_max(returned_data_dict["semantic"])}')
    print(f'normals {torch_min_max(returned_data_dict["normals"])}')
 
 
def depth_colorize(depth):
    cmap3 = plt.cm.turbo
    depth = (depth - np.min(depth)) / ((np.max(depth) - np.min(depth)) + 1e-6)
    depth_colorized = 255 * cmap3(np.squeeze(depth))[:, :, :3]  # H, W, C    
    return depth_colorized.astype('uint8')

def depth_colorize_gt(depth): 
    cmap3 = plt.cm.turbo 
    depth_x = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) 
    depth_colorized = 255 * cmap3(np.squeeze(depth_x))[:, :, :3]  # H, W, C 
    return depth_colorized.astype('uint8'), np.min(depth),np.max(depth)

def depth_colorize_pred(depth,min_val,max_val): 
    cmap3 = plt.cm.turbo 
    depth_x = (depth - min_val) / (max_val - min_val) 
    depth_colorized = 255 * cmap3(np.squeeze(depth_x))[:, :, :3]  # H, W, C 
    return depth_colorized.astype('uint8')

def rgb_visualizer(image):
    #print(np.min(image), np.max(image))
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = 255 * image
    #print(np.min(image), np.max(image))

    rgb = np.transpose(image, (1, 2, 0))
    return rgb.astype('uint8')

def torch_min_max(data):
    minmax = (torch.min(data.float()).item(),torch.mean(data.float()).item(),torch.median(data.float()).item(),torch.max(data.float()).item())
    return minmax

def train_sweep():

    with wandb.init():
        wb_config = wandb.config
        for k, v in wb_config.items():
            ks = k.split(".")
            opt[ks[0]][ks[1]] = v
        #print(opt)
        training(opt)


def training(opt):
    
    #Initialize weights and biases logger
    if opt.wandb.t_logger:
        print("Started logging in wandb")
        wandb_config = OmegaConf.to_container(opt, resolve=True, throw_on_missing=True)
        wandb.init(project=opt.wandb.project_name,entity=opt.wandb.entity,
            name='{}_{}_{}'.format(opt.data.dataset[0],opt.network.backbone,opt.network.tasks),
            config = wandb_config)

    model_name = opt.network.backbone 
    dataset_name = opt.data.dataset

    # create logging folder to store training weights and losses
    if not os.path.exists('logging'):
        os.makedirs('logging')

    # define model, optimiser and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
    ############################### 
    #BRANCH_1: MTL vs Single Task
    ###############################

    if opt.network.tasks == 'all':
            
        if opt.data.dataset == 'sim_warehouse':
            train_tasks = {'semantic': 23, 'depth': 1, 'normals': 3}
            pri_tasks = {'semantic': 23, 'depth': 1, 'normals': 3}
        elif opt.data.dataset == 'nyuv2':
            train_tasks = {'semantic': 13, 'depth': 1 , 'normals': 3}
            pri_tasks = {'semantic': 13, 'depth': 1, 'normals': 3}
            
            #train_tasks = {'depth': 1, 'semantic': 13, 'normals': 3}
            #pri_tasks = {'depth': 1, 'semantic': 13, 'normals': 3}
            
        #train_tasks = create_task_flags('all', opt.data.dataset, with_noise=False)
        network = opt.network.backbone + 'MTL_' + opt.network.mtl_architecture
        print(network)
        
        ############################### 
        #UTILS CREATE NETWORK=
        ###############################
        
        if network == 'ResNetMTL_Split':
            model = MTLDeepLabv3(train_tasks, opt.data.dataset).to(device)
        elif network == 'ResNetMTL_MTAN':
            model = MTANDeepLabv3(train_tasks).to(device)
        elif network == "SegNetMTL_Split": #OK 
            model = SegNetSplit(train_tasks,opt.data.dataset).to(device)
        elif network == "SegNetMTL_MTAN":
            model = SegNetMTAN(train_tasks).to(device)
        #elif network == "EdgeSegNet":
        #    model = EdgeSegNet(train_tasks).to(device)
        #elif network == "GuidedDepth":
        #    model = GuideDepth(train_tasks).to(device) 
        elif network == "DDRNetMTL_Split": #OK
            model = DualResNetMTL(BasicBlock, [2, 2, 2, 2], train_tasks, opt.data.dataset, planes=32, spp_planes=128, head_planes=64).to(device)
        #elif network == "Segmentation":
        #    model = smp.Unet(
        #        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        #        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #        classes=23,                      # model output channels (number of classes in your dataset)
        #    ).to(device)     
        else:
            raise ValueError 
        
        ###############################
    else:
        if opt.network.tasks == 'semantic':
            if opt.data.dataset == 'sim_warehouse':
                train_tasks = {'semantic': 23}
                pri_tasks = {'semantic': 23}
            elif opt.data.dataset == 'nyuv2':
                train_tasks = {'semantic': 13}
                pri_tasks = {'semantic': 13}
        elif opt.network.tasks == 'depth':
            train_tasks = {'depth': 1}
            pri_tasks = {'depth': 1}
        elif opt.network.tasks == 'normals':
            train_tasks = {'normals': 3}
            pri_tasks = {'normals': 3}  
            
        network = opt.network.backbone + 'Single'
        print(network)
            
        if network == "SegNetSingle":
            model = SegNetSingle(train_tasks,opt.data.dataset).to(device)
        elif network == "ResNetSingle":
            model = ResNetSingle(train_tasks, opt.data.dataset).to(device)
        elif network == "DDRNetSingle":
            model = DualResNetSingle(BasicBlock, [2, 2, 2, 2], train_tasks, opt.data.dataset, planes=32, spp_planes=128, head_planes=64).to(device)
            #model.load_state_dict(torch.load('models/DDRNet23s_imagenet.pth'), strict= False)
        
    #pri_tasks = create_task_flags(opt.network.tasks, opt.data.dataset, with_noise=False)
    #print(pri_tasks)
    train_tasks_str = ''.join(task.title() + ' + ' for task in train_tasks.keys())[:-3]
    pri_tasks_str = ''.join(task.title() + ' + ' for task in pri_tasks.keys())[:-3]
    print('Dataset: {} | Training Task: {} | Primary Task: {} in Multi-task / Auxiliary Learning Mode with {}'
        .format(opt.data.dataset.title(), train_tasks_str, pri_tasks_str, opt.network.backbone.upper()))
    print('Applying Multi-task Methods: Weighting-based: {} + Gradient-based: {}'
        .format(opt.network.weighting.title(), opt.network.grad_method))
    # define new or load excisting model and optimizer 

    total_epoch = opt.training.epochs
    saving_epoch = 0

    # choose task weighting here
    if opt.network.weighting == 'uncert':
        logsigma = torch.tensor([-0.7] * len(train_tasks), requires_grad=True, device=device)
        params = list(model.parameters()) + [logsigma]
        logsigma_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)

    if opt.network.weighting in ['dwa', 'equal']:
        T = 2.0  # temperature used in dwa
        lambda_weight = np.ones([total_epoch, len(train_tasks)])
        #print('lambdaweight',lambda_weight)
        params = model.parameters()
        

    #UNIVERSAL OPTIMIZERS AND LR SCHEDULER
    #CHANGE
    optimizer_o = opt.training.optimizer_o
    learning_rate = opt.training.learning_rate
    if optimizer_o == 'sgd':
        optimizer = optim.SGD(params, lr=learning_rate, weight_decay=opt.training.weight_decay)
    elif optimizer_o == 'adam':
        optimizer = optim.Adam(params, lr=opt.training.learning_rate, weight_decay=opt.training.weight_decay)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[15,20,25], gamma=0.1)


    if opt.training.t_checkpoint:
        #CHANGE
        checkpoint = torch.load(f"models/model_{model_name}_{dataset_name}_{opt.training.o_checkpoint}")#        path = f"models/model_{model_name}_{dataset_name}_epoch{index}.pth"
        #checkpoint = torch.load(f"models/model_{model_name}_{dataset_name}.pth")
        model.load_state_dict(checkpoint["model_state_dict"]) 

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    print("\nSTEP. Loading datasets...")
    #model.load_state_dict(torch.load('models/DDRNet23s_imagenet.pth'), strict=True)

    if opt.data.dataset == 'sim_warehouse':
        '''
        batch_size = opt.batch_size
        train_loader = DataLoader(DecnetDataloader('dataset/sim_warehouse/train/datalist_train_warehouse_sim.list', split='train'),batch_size=batch_size,num_workers=0, shuffle=True)#num_workers=0 otherwise there is an error. Need to see why
        test_loader = DataLoader(DecnetDataloader('dataset/sim_warehouse/test/datalist_test_warehouse_sim.list', split='eval'),batch_size=1)
        '''
        dataset_path = os.path.join(opt.data.path,opt.data.dataset)
        dataset_path = 'dataset/sim_warehouse'
        batch_size = opt.data.batch_size 
        train_set = SimWarehouse(root=dataset_path, train=True, augmentation=True)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        test_set = SimWarehouse(root=dataset_path, train=False)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False
        )    
        
    elif opt.data.dataset == 'nyuv2':
        #dataset_path = 'dataset/nyuv2'
        dataset_path = os.path.join(opt.data.path,opt.data.dataset)
        batch_size = opt.data.batch_size 
        train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        test_set = NYUv2(root=dataset_path, train=False)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False
        )    
    else:
        raise ValueError
        
    #Visualize data for sanity check
    train_sample = next(iter(train_loader))  
    test_sample = next(iter(test_loader))


    data_sample = train_sample

    print(f"Data sanity check. RGB.shape: {data_sample['rgb'].shape},\tDepth.shape {data_sample['depth'].shape},\
        \tSemantic.shape {data_sample['semantic'].shape},\tNormals.shape {data_sample['normals'].shape}")
    #print(test_sample)#

    #sanity_train_target = {task_id: data_sample[task_id].to(device) for task_id in train_tasks.keys()}
    #print('sanity_train_target',sanity_train_target.shape)
    # Train and evaluate multi-task network
    train_batch = len(train_loader)
    test_batch = len(test_loader)

    #print(train_batch, test_batch)


    #train_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.data.dataset)
    #test_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.data.dataset)#, include_mtl=True)


    train_metric = OriginalTaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.data.dataset)
    if opt.network.tasks == 'all':
        test_metric = OriginalTaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.data.dataset, include_mtl=True)
    else: 
        test_metric = OriginalTaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.data.dataset)
        
    # load loss and initialize index/epoch
    if opt.training.t_checkpoint:
        print("Loading checkpoint...")
        loss = checkpoint["loss"]
        index = checkpoint["epoch"] + 1
    else:
        index = 0


    while index < total_epoch:
        model.train()
        force_cudnn_initialization()
        for i,multitaskdata in ((enumerate(tqdm(train_loader)))):
        #if True:
            #multitaskdata = next(iter(train_loader))   
            #print(i,len(train_loader))
            image = multitaskdata['rgb'].to(device)
            #print(multitaskdata)
            train_target = {task_id: multitaskdata[task_id].to(device) for task_id in train_tasks.keys()}
            #print(train_target)
            import torchvision.transforms.functional as ttf

            #if gt.shape[2] == 720:
            semantic_gt = multitaskdata['semantic'].to(device)
            resized_semantic_gt = ttf.resize(semantic_gt, size=(360,640)).squeeze(1)
            #print(pred.shape,gt.shape)
            #gt = gt.squeeze(1)
            #print(pred.shape,gt.shape)

            train_target['semantic'] = resized_semantic_gt
            optimizer.zero_grad()
            #print(image.shape)
            train_pred = model(image)
            
            #print(f'train_pred_shape {train_pred[0]}')
            #print(torch.min(train_pred[0]), torch.max(train_pred[0]))

            #if i == 0:
                #print(multitaskdata['file'])
                #min_max_sanity_check(multitaskdata)
                #print(f'prediction {torch_min_max(train_pred[0])}')
            
            #print(train_tasks)
            if opt.network.tasks == 'all':
                train_loss = [compute_loss_ole(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
                train_loss_tmp = [0] * len(train_tasks)
            else:
                #print('getting in now')
                #print(f'train_pred_shape {train_pred.shape}')
                #print(f'train_target[task_id].shape {train_target["semantic"].shape}')
                

                train_loss = [compute_loss_ole(train_pred, train_target[task_id], task_id) for task_id in train_tasks.keys()]
                train_loss_tmp = [0] * len(train_tasks)


            if opt.network.weighting in ['equal', 'dwa']:
                train_loss_tmp = [w * train_loss[i] for i, w in enumerate(lambda_weight[index])]#torch.tensor([1, 2, 3]))]#

            if opt.network.weighting == 'uncert':
                train_loss_tmp = [1 / (2 * torch.exp(w)) * train_loss[i] + w / 2 for i, w in enumerate(logsigma)]


        
            loss = sum(train_loss_tmp)
            #print(type(loss))
            loss.backward()
            optimizer.step()

            #print(optimizer)
            
            
            if opt.network.tasks == 'all':
                train_metric.update_metric(train_pred, train_target, train_loss)
                
            else:
                train_metric.update_single_metric(train_pred, train_target, train_loss[0])
            pass  
        train_str, train_losses = train_metric.compute_metric()
        #print('train_losses',train_losses[opt.network.tasks][index][1])
        
        train_metric.reset()

        model.eval()

        with torch.no_grad():
            for i_test, multitaskdatatest in enumerate(tqdm(test_loader)):
                #multitaskdatatest = next(iter(test_loader))   
                image = multitaskdatatest['rgb'].to(device)
                test_target = {task_id: multitaskdatatest[task_id].to(device) for task_id in train_tasks.keys()}
                test_pred = model(image)
                '''
                if opt.network.tasks == 'depth':
                    gt = multitaskdatatest['depth'].to(device).squeeze(1)
                    gt_image,gt_min_val,gt_max_val = depth_colorize_gt(gt.cpu().numpy())
                    prediction = test_pred[0]
                    prediction = depth_colorize_pred(prediction.cpu().numpy(),gt_min_val,gt_max_val)
                elif opt.network.tasks == 'semantic':
                    gt = multitaskdatatest['semantic'].to(device).squeeze(1)
                    gt_image = depth_colorize(gt.cpu().numpy())
                    prediction = F.softmax(test_pred[0], dim=0).argmax(0)
                    prediction = depth_colorize(prediction.cpu().numpy())
                elif opt.network.tasks == 'normals':
                    gt = multitaskdatatest['normals'].to(device).squeeze(1).squeeze(0)
                    gt_image = rgb_visualizer(gt.cpu().numpy())
                    #print(gt.shape)
                    prediction = rgb_visualizer(test_pred[0].cpu().numpy())
                    #prediction = F.softmax(test_pred[0], dim=0).argmax(0)
                elif opt.network.tasks == 'all':
                    #gt_depth = multitaskdatatest['depth'].to(device).squeeze(1)
                    gt_image = depth_colorize(multitaskdatatest['depth'].to(device).squeeze(1).cpu().numpy())
                    prediction = depth_colorize(test_pred[1].cpu().numpy())
                    gt_semantic = depth_colorize(multitaskdatatest['semantic'].to(device).squeeze(1).cpu().numpy())           
                    prediction_semantic = depth_colorize(F.softmax(test_pred[0].squeeze(0), dim=0).argmax(0).cpu().numpy())       
                    gt_normals = rgb_visualizer(multitaskdatatest['normals'].to(device).squeeze(1).squeeze(0).cpu().numpy())
                    prediction_normals = rgb_visualizer(test_pred[2].squeeze(0).cpu().numpy())
                
                if i_test == 0:
                    img_list_0= []
                    image_rgb_0 = wandb.Image(image.permute(0,2,3,1).cpu().numpy(), caption="RGB_0")
                    image_gt_0 = wandb.Image(gt_image, caption="GT_0")
                    image_pred_0 = wandb.Image(prediction, caption="Pred_0")
                    img_list_0.append(image_rgb_0)
                    img_list_0.append(image_gt_0)
                    img_list_0.append(image_pred_0)
                    if opt.network.tasks == 'all':
                        image_gt_0_1 = wandb.Image(gt_semantic, caption="GT_0_1")
                        image_pred_0_1 = wandb.Image(prediction_semantic, caption="Pred_0_1")
                        image_gt_0_2 = wandb.Image(gt_normals, caption="GT_0_2")
                        image_pred_0_2 = wandb.Image(prediction_normals, caption="Pred_0_2")
                        img_list_0.append(image_gt_0_1)
                        img_list_0.append(image_pred_0_1)
                        img_list_0.append(image_gt_0_2)
                        img_list_0.append(image_pred_0_2)
                        
                    
                if i_test == 10:
                    img_list_50= []
                    image_rgb_50 = wandb.Image(image.permute(0,2,3,1).cpu().numpy(), caption="RGB_50")
                    image_gt_50 = wandb.Image(gt_image, caption="GT_50")
                    image_pred_50 = wandb.Image(prediction, caption="Pred_50")
                    img_list_50.append(image_rgb_50)
                    img_list_50.append(image_gt_50)
                    img_list_50.append(image_pred_50)
                    if opt.network.tasks == 'all':
                        image_gt_50_1 = wandb.Image(gt_semantic, caption="GT_50_1")
                        image_pred_50_1 = wandb.Image(prediction_semantic, caption="Pred_50_1")
                        image_gt_50_2 = wandb.Image(gt_normals, caption="GT_50_2")
                        image_pred_50_2 = wandb.Image(prediction_normals, caption="Pred_50_2")
                        img_list_50.append(image_gt_50_1)
                        img_list_50.append(image_pred_50_1)
                        img_list_50.append(image_gt_50_2)
                        img_list_50.append(image_pred_50_2)
                if i_test == 20:
                    img_list_100= []
                    image_rgb_100 = wandb.Image(image.permute(0,2,3,1).cpu().numpy(), caption="RGB_100")
                    image_gt_100 = wandb.Image(gt_image, caption="GT_100")
                    image_pred_100 = wandb.Image(prediction, caption="Pred_100")
                    img_list_100.append(image_rgb_100)
                    img_list_100.append(image_gt_100)
                    img_list_100.append(image_pred_100)    
                    if opt.network.tasks == 'all':
                        image_gt_100_1 = wandb.Image(gt_semantic, caption="GT_100_1")
                        image_pred_100_1 = wandb.Image(prediction_semantic, caption="Pred_100_1")
                        image_gt_100_2 = wandb.Image(gt_normals, caption="GT_100_2")
                        image_pred_100_2 = wandb.Image(prediction_normals, caption="Pred_100_2")
                        img_list_100.append(image_gt_100_1)
                        img_list_100.append(image_pred_100_1)
                        img_list_100.append(image_gt_100_2)
                        img_list_100.append(image_pred_100_2)        
                if i_test == 30:
                    img_list_150= []
                    image_rgb_150 = wandb.Image(image.permute(0,2,3,1).cpu().numpy(), caption="RGB_150")
                    image_gt_150 = wandb.Image(gt_image, caption="GT_150")
                    image_pred_150 = wandb.Image(prediction, caption="Pred_150")
                    img_list_150.append(image_rgb_150)
                    img_list_150.append(image_gt_150)
                    img_list_150.append(image_pred_150)
                    if opt.network.tasks == 'all':
                        image_gt_150_1 = wandb.Image(gt_semantic, caption="GT_150_1")
                        image_pred_150_1 = wandb.Image(prediction_semantic, caption="Pred_150_1")
                        image_gt_150_2 = wandb.Image(gt_normals, caption="GT_150_2")
                        image_pred_150_2 = wandb.Image(prediction_normals, caption="Pred_150_2")
                        img_list_150.append(image_gt_150_1)
                        img_list_150.append(image_pred_150_1)
                        img_list_150.append(image_gt_150_2)
                        img_list_150.append(image_pred_150_2)

                '''
                if opt.network.tasks == 'all':
                    test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
                    test_metric.update_metric(test_pred, test_target, test_loss)
                    #print(test_loss)
                else:
                    test_loss = [compute_loss(test_pred, test_target[task_id], task_id) for task_id in test_target.keys()]
                    test_metric.update_single_metric(test_pred, test_target, test_loss[0])
                    #print(test_loss)

                i_test +=1

        test_str, metric = test_metric.compute_metric()
        test_metric.reset()
        
        
        scheduler.step()
        #print(optimizer)
        print(test_str[0])
        #print(test_metric.metric)
        print('Entering evaluation phase...')
        print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
            .format(index, train_str, test_str, opt.network.tasks.title(), test_metric.get_best_performance(opt.network.tasks)))

        if opt.network.weighting in ['dwa', 'equal']:
            dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                    'weight': lambda_weight}

            print(get_weight_str(lambda_weight[index], train_tasks))

        if opt.network.weighting == 'uncert':
            logsigma_ls[index] = logsigma.detach().cpu()
            dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                    'weight': logsigma_ls}

            print(get_weight_str(1 / (2 * np.exp(logsigma_ls[index])), train_tasks))

        np.save('logging/mtl_dense_{}_{}_{}_{}_{}_{}_.npy'
                .format(opt.network.backbone, opt.data.dataset, opt.network.tasks, opt.network.weighting, opt.network.grad_method, seed), dict)
        np.save('logging/mtl_dense_{}_{}_{}_{}_{}_{}_.txt'
                .format(opt.network.backbone, opt.data.dataset, opt.network.tasks, opt.network.weighting, opt.network.grad_method, seed), dict)

        # Save full model
        if opt.network.tasks == 'all':
            current_test_metric = metric['all'][index]
            #print('current_loss', current_loss)
        else: 
            current_test_metric = metric[opt.network.tasks][index][1]
            print('current_loss', current_test_metric)
            
            
        #current_loss = test_metric.get_best_performance(opt.network.tasks)
        
        if index == 0:
            best_test_str = current_test_metric
            
        if opt.wandb.t_logger:
            
            var_exists = 'img_list_0' in locals() or 'img_list_0' in globals()
            if not var_exists: 
                print("Variable does not exist")
                img_list_0, img_list_50, img_list_100, img_list_150 = [],[],[],[]
        
            if opt.network.tasks == 'all':            
                wandb.log({'training_loss_depth':train_losses['depth'][index][1],'training_loss_semantic':train_losses['semantic'][index][1],
                        'training_loss_normals':train_losses['normals'][index][1],'test_metric':current_test_metric, 'Images_0':img_list_0,    
                        'Images_50':img_list_50, 'Images_100': img_list_100,'Images_150': img_list_150}, step = index)
                #wandb.log({'train_loss':train_metric.metric[index], 'test_loss':test_metric.metric[index]}, step = index)
            elif opt.network.tasks == 'semantic':
                wandb.log({'training_loss':train_losses[opt.network.tasks][index][1],'test_metric':current_test_metric, 'Images_0':img_list_0,    
                        'Images_50':img_list_50, 'Images_100': img_list_100,'Images_150': img_list_150}, step = index)
            elif opt.network.tasks == 'depth':
                wandb.log({'training_loss':train_losses[opt.network.tasks][index][1],'test_metric':current_test_metric, 'Images_0':img_list_0,    
                        'Images_50':img_list_50, 'Images_100': img_list_100,'Images_150': img_list_150}, step = index)
            elif opt.network.tasks == 'normals':
                wandb.log({'training_loss':train_losses[opt.network.tasks][index][1],'test_metric':current_test_metric, 'Images_0':img_list_0,    
                        'Images_50':img_list_50, 'Images_100': img_list_100,'Images_150': img_list_150}, step = index)



        print(optimizer_o,learning_rate, optimizer)
        if opt.network.tasks == "depth" or opt.network.tasks == "normals":
            if current_test_metric <= best_test_str:
                best_test_str = current_test_metric

                save_model = True
            else:
                save_model = False
        else:
            if current_test_metric >= best_test_str:
                best_test_str = current_test_metric

                save_model = True
            else:
                save_model = False
                
        if save_model:
            file = f"models/model_{model_name}_{dataset_name}_epoch{saving_epoch}.pth"
            if os.path.exists(file):
                os.remove(f"models/model_{model_name}_{dataset_name}_epoch{saving_epoch}.pth")
            saving_epoch = index
            
            print("Saving full model")
            path = f"models/model_{model_name}_{dataset_name}_epoch{index}.pth"
            device = torch.device("cuda")
            model.to(device)
            torch.save({
                'epoch': index,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                }, path)
            #break
        print(current_test_metric,best_test_str)

        index += 1

    print("Training complete")
    
    
if __name__ == "__main__":

    opt = OmegaConf.load('configs/config.yaml')

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    
    if opt.reproduceability.t_seed:
        seed = opt.reproduceability.o_seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
    if opt.sweep.sweep_t:
        sweep_config = yaml.load(
            open(opt.sweep.sweep_o, "r"), Loader=yaml.FullLoader
        )
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project="MTLwarehouse-Sweep",
            entity=opt.wandb.entity,
        )
        wandb.agent(sweep_id, function=train_sweep, count=15)
        
    else:
        training(opt)
