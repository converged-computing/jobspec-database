#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import time
import datetime
import torch
torch.set_num_threads(1)
import pickle 

from utils.train import *
from utils.load_data import *
from utils.log import TrainLogger
from models import replay, detect, ewc
from models.losses import *
from models import trainer
from models.model import D2STGNN
from torch_geometric.utils import to_dense_batch, k_hop_subgraph
import yaml
import setproctitle
import logging
import networkx as nx
import gc

# os.environ['CUDA_VISIBLE_DEVICES']='-1'

def dataloaderEveryYears(dataset_name, load_pkl, data_dir, config, year, dataset_type):
    if load_pkl:
        t1   = time.time()
        try:
            dataloader  = pickle.load(open('./output/dataloader_' + dataset_name + '_' + str(year)\
                 + '.pkl', 'rb'))
        except:
            batch_size  = config['model_args']['batch_size']
            if dataset_type == 'BAST-Stream':
                dataloader  = load_dataset(data_dir, batch_size, batch_size, 
                    batch_size*4, dataset_name, two_way=True, year=str(year))
            else:
                dataloader  = load_dataset(data_dir, batch_size, batch_size, 
                    batch_size*4, dataset_name, year=str(year))
            gc.collect()
            pickle.dump(dataloader, open('./output/dataloader_' + dataset_name + '_' + str(year)\
                 + '.pkl', 'wb'))
        t2  = time.time()
        logging.info("Load dataset: {:.2f}s...".format(t2-t1))
    else:
        t1   = time.time()
        batch_size  = config['model_args']['batch_size']
        if dataset_type == 'BAST-Stream':
            dataloader  = load_dataset(data_dir, batch_size, batch_size, 
                batch_size, dataset_name, two_way=True, year=str(year))
        else:
            dataloader  = load_dataset(data_dir, batch_size, batch_size, 
                batch_size, dataset_name, year=str(year))
        pickle.dump(dataloader, open('./output/dataloader_' + dataset_name + '_' + str(year)\
                 + '.pkl', 'wb'))
        t2  = time.time()
        logging.info("Load dataset: {:.2f}s...".format(t2-t1))
    scaler          = dataloader['scaler']
    
    # if dataset_name == 'PEMS04' or dataset_name == 'PEMS08' or dataset_name == 'BAST':  # traffic flow
    if dataset_name == 'PEMS04' or dataset_name == 'PEMS08':  # traffic flow
        _min = pickle.load(open("{0}/min.pkl".format(data_dir), 'rb'))
        _max = pickle.load(open("{0}/max.pkl".format(data_dir), 'rb'))
    else:
        _min = None
        _max = None
    
    t1   = time.time()
    if dataset_type == "Pems3-Stream":
        adj_mx, adj_ori = load_adj(
            config['data_args']['adj_data_path'] + year + '_adj.npz', 
            config['data_args']['adj_type'],
            is_npz=config['data_args']['is_npz'])
    elif dataset_type == 'BAST-Stream':
        adj_mx, adj_ori = load_adj(
            config['data_args']['adj_data_path'] + 'adj_BAST_' + year + '.npz', 
            config['data_args']['adj_type'],
            is_npz=config['data_args']['is_npz'])
    t2  = time.time()
    logging.info("Load adjacent matrix: {:.2f}s...".format(t2-t1))
    return dataloader, scaler, _min, _max, adj_mx, adj_ori

def trainAYear(model, resume_epoch, optim_args, engine, dataloader, train_time, val_time, device,
                model_name,  _max, _min, early_stopping, save_path_resume, scaler, dataset_name, args):
    batch_num   = resume_epoch * len(dataloader['train_loader'])    
    if args.cur_year > args.begin_year and args.strategy == 'incremental':
        model = loadpremodel(engine.model, args.pre_model, args)
        if args.ewc:
            logging.info("[*] EWC! lambda {:.6f}".format(args.ewc_lambda))
            engine.model = ewc.EWC(engine.model, args, args.ewc_lambda, args.ewc_strategy)
            ewc_loader = dataloader['train_loader']
            engine.model.register_ewc_params(ewc_loader, engine.loss, device)
    for epoch in range(resume_epoch + 1, optim_args['epochs']):
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        # train a epoch
        time_train_start    = time.time()
        # current_learning_rate = engine.lr_scheduler.get_last_lr()[0]
        current_learning_rate = engine.optimizer.param_groups[0]['lr']
        train_loss = []
        train_mape = []
        train_rmse = []
        dataloader['train_loader'].shuffle()    # traing data shuffle when starting a new epoch.
        totaliter = 0
        avgmae = 0.0
        for itera, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            if torch.cuda.is_initialized() and itera%10 == 0:
                torch.cuda.empty_cache()
            if args.cur_year > args.begin_year and args.strategy == 'incremental':
                x = x[:, :, args.subgraph, :].reshape(x.shape[0], x.shape[1], len(args.subgraph), x.shape[3])
                y = y[:, :, args.subgraph, :].reshape(y.shape[0], y.shape[1], len(args.subgraph), y.shape[3])
            totaliter += 1
            trainx          = data_reshaper(x, device)
            trainy          = data_reshaper(y, device)
            # print(trainx.min(), trainx.max(), trainy.min(), trainy.max())
            mae, mape, rmse = engine.train(trainx, trainy, args, batch_num=batch_num, _max=_max, _min=_min)
            print("train : {0}: {1}".format(itera, mae), end='\r')
            avgmae += mae
            train_loss.append(mae)
            train_mape.append(mape)
            train_rmse.append(rmse)
            batch_num += 1
            # break
        logging.info("train : {0}: {1}".format(epoch, avgmae/totaliter))
        time_train_end      = time.time()
        train_time.append(time_train_end - time_train_start)

        current_learning_rate = engine.optimizer.param_groups[0]['lr']

        # record history loss
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
# ===================================================== Validation ================================================================= #
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        time_val_start      = time.time()
        mvalid_loss, mvalid_mape, mvalid_rmse, = engine.eval(device, dataloader, model_name, args,
                        _max=_max, _min=_min)
        if engine.if_lr_scheduler:
            engine.lr_scheduler.step(mvalid_loss)
        # mvalid_loss, mvalid_mape, mvalid_rmse, = 0,0,0
        time_val_end        = time.time()
        val_time.append(time_val_end - time_val_start)

        curr_time   = str(time.strftime("%d-%H-%M", time.localtime()))
        log = 'Current Time: ' + curr_time + ' | Epoch: {:03d} | Train_Loss: {:.4f} | Train_MAPE: {:.4f} |\
         Train_RMSE: {:.4f}  \n | Valid_Loss: {:.4f} | Valid_RMSE: {:.4f} | Valid_MAPE: {:.4f} | LR: {:.6f}'
        logging.info(log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, 
                mvalid_rmse, mvalid_mape, current_learning_rate))
        if args.cur_year > args.begin_year and args.strategy == 'incremental':
            with torch.no_grad():
                model_weight = engine.model.state_dict()
                for name, param in args.full_model.named_parameters():
                    if name in ['module.node_emb_u', 'module.node_emb_d']:
                        tmp_emb = param.clone().cuda()
                        try:
                            tmp_emb[args.subgraph] = model_weight[name].cuda()
                        except:
                            tmp_emb[args.subgraph] = model_weight['model.' + name].cuda()
                        param.copy_(tmp_emb)
                    else:
                        try:
                            param.copy_(model_weight['model.' + name].clone())
                        except:
                            param.copy_(model_weight[name].clone())
            early_stopping(mvalid_loss, args.full_model)
        else:
            early_stopping(mvalid_loss, model)
        if early_stopping.early_stop:
            logging.info('Early stopping!')
            break
# =============================================================== Test ================================================================= #
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        if args.cur_year > args.begin_year and args.strategy == 'incremental':
            engine.test(args.full_model, save_path_resume, device, dataloader, scaler, model_name, args,
                _max=_max, _min=_min, loss=engine.loss, dataset_name=dataset_name)
        else:
            # break
            engine.test(model, save_path_resume, device, dataloader, scaler, model_name, args,
                _max=_max, _min=_min, loss=engine.loss, dataset_name=dataset_name)
        # break

def loadpremodel(model, premodelpth, args):
    premodel = torch.load(premodelpth)
    args.full_model = torch.nn.DataParallel(args.full_model)
    prefix = ''
    if args.cur_year > args.begin_year+1 and args.strategy == 'incremental' and args.ewc:
        prefix = ''
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ['module.node_emb_u', 'module.node_emb_d']:
                try:
                    pre_node_len = premodel[prefix + name].size(0)
                    borrow_idx = torch.LongTensor([i for i in args.subgraph if i < pre_node_len])
                    tmp_emb = param.clone()
                    tmp_emb[:borrow_idx.size(0)] = premodel[prefix + name][borrow_idx]
                    param.copy_(tmp_emb)
                except:
                    tmpname = str(name).replace('module.','')
                    pre_node_len = premodel[tmpname].size(0)
                    borrow_idx = torch.LongTensor([i for i in args.subgraph if i < pre_node_len])
                    tmp_emb = param.clone()
                    tmp_emb[:borrow_idx.size(0)] = premodel[tmpname][borrow_idx]
                    param.copy_(tmp_emb)
            else:
                try:
                    param.copy_(premodel[prefix + name])
                except:
                    tmpname = str(name).replace('module.','')
                    param.copy_(premodel[tmpname])
    if args.cur_year > args.begin_year and args.strategy == 'incremental':
        with torch.no_grad():
            for name, param in args.full_model.named_parameters():
                if name in ['module.node_emb_u', 'module.node_emb_d']:
                    try:
                        pre_node_len = premodel[prefix + name].size(0)
                        tmp_emb = param.clone()
                        if pre_node_len > param.size(0):
                            tmp_emb = premodel[prefix + name][:param.size(0)]
                        else:  
                            tmp_emb[:pre_node_len] = premodel[prefix + name]
                        param.copy_(tmp_emb)
                    except:
                        pre_node_len = premodel[name.replace('module.','')].size(0)
                        tmp_emb = param.clone()
                        if pre_node_len > param.size(0):
                            tmp_emb = premodel[name.replace('module.','')][:param.size(0)]
                        else:  
                            tmp_emb[:pre_node_len] = premodel[name.replace('module.','')]
                        param.copy_(tmp_emb)
                else:
                    try:
                        param.copy_(premodel[prefix + name])
                    except:
                        param.copy_(premodel[name.replace('module.','')])
    return model

def main(**kwargs):
    set_config(0)
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', type=str, default='Pems3-Stream', help='Dataset name.')
    parser.add_argument('--dataset', type=str, default='BAST-Stream', help='Dataset name.')
    parser.add_argument('--stream', type=int, default=0, help='Dataset name.')
    args = parser.parse_args()
    config_path = "configs/" + args.dataset + ".yaml"

    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        
    data_dir        = config['data_args']['data_dir']
    dataset_name    = config['data_args']['data_dir'].split("/")[-1]

    device          = torch.device(config['start_up']['device'])
    timestr = '{0:_%Y-%m-%d__%H_%M_%S}'.format(datetime.datetime.now())
    load_pkl        = config['start_up']['load_pkl']
    model_name      = config['start_up']['model_name']

    begin_year      = config['start_up']['begin_year']
    end_year        = config['start_up']['end_year']
    vars(args)['begin_year'] = begin_year
    vars(args)['end_year']   = end_year
    vars(args)['graph_path']      = config['data_args']['adj_data_path']
    vars(args)['detect_strategy'] = config['start_up']['detect_strategy']
    vars(args)['replay_strategy'] = config['start_up']['replay_strategy']
    vars(args)['num_hops']        = config['start_up']['num_hops']
    vars(args)['strategy']        = config['start_up']['strategy']
    vars(args)['ewc']             = config['start_up']['ewc']
    vars(args)['ewc_strategy']    = config['start_up']['ewc_strategy']
    vars(args)['ewc_lambda']      = config['start_up']['ewc_lambda']
    save_path_logger= './output/' + config['start_up']['model_name'] + "_Stream_" + str(begin_year) + "_" \
                + str(end_year) + "_" + dataset_name + timestr + '_Strategy_' + str(args.strategy) \
                + '_detect_' + str(config['start_up']['detect']) + ".log"    
    logging.basicConfig(filename=save_path_logger, level=logging.INFO)  
    setproctitle.setproctitle("{0}.{1}@S22".format(model_name, dataset_name))
# ================================ Hyper Parameters ================================= #
    # model parameters
    model_args  = config['model_args']
    model_args['device']        = device
    model_args['dataset']       = dataset_name
    vars(args)['device']        = device

    
# ============================= Model =========================================================== #
    # log
    logger  = TrainLogger(model_name, dataset_name)
    save_path_logger= './output/' + config['start_up']['model_name'] + "_Stream_" \
        + dataset_name + timestr + ".log"    
    # begin training:
    train_time  = []    # training time
    val_time    = []    # validate time
    last_save_path = ''
    
# ========================== load dataset, adjacent matrix, node embeddings ====================== #
    for i in range(begin_year, end_year+1):
        train_time  = []    # training time
        val_time    = []    # validate time
        vars(args)['cur_year'] = i
        if args.dataset == 'Pems3-Stream' or args.dataset == 'BAST-Stream':
            data_dir_year = data_dir + '_' + str(i)
            print('current year:', data_dir_year)
        dataloader, scaler, _min, _max, adj_mx, adj_ori = dataloaderEveryYears(
            dataset_name=dataset_name, load_pkl=load_pkl, data_dir=data_dir_year, 
            config=config, year=str(i), dataset_type=args.dataset)
        graph = nx.from_numpy_matrix(adj_ori)
        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = i
        # training strategy parametes
        model_args['num_nodes']     = adj_mx[0].shape[0]
        model_args['adjs']          = [torch.tensor(i).to(device) for i in adj_mx]
        model_args['adjs_ori']      = torch.tensor(adj_ori).to(device)
        optim_args                  = config['optim_args']
        optim_args['cl_steps']      = optim_args['cl_epochs'] * len(dataloader['train_loader'])
        optim_args['warm_steps']    = optim_args['warm_epochs'] * len(dataloader['train_loader'])
        save_path       = './output/' + config['start_up']['model_name'] + "_Stream_" \
                            + str(i)+ "_" + dataset_name + '_Strategy_' + str(args.strategy) \
                            + '_detect_' + str(config['start_up']['detect']) + timestr + ".pt"             # the best model
        save_path_resume= './output/' + config['start_up']['model_name'] + "_Stream_" \
                            + str(i)+ "_" + dataset_name + '_Strategy_' + str(args.strategy) \
                            + '_detect_' + str(config['start_up']['detect']) + timestr + "_resume.pt"      # the resume model

        logger.print_model_args(model_args, ban=['adjs', 'adjs_ori', 'node_emb'])
        logger.print_optim_args(optim_args)
        logging.info("Whole trainining iteration is " + str(len(dataloader['train_loader'])))
        if i == begin_year or args.strategy == 'retrain':
            print('retrain/init model')
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            if i>begin_year:
                del model
            model   = D2STGNN(**model_args).to(device)
            engine  = trainer(scaler, model, **optim_args)
        if i > args.begin_year and args.strategy == "incremental":
            vars(args)['pre_model'] = last_save_path
            node_list = list()
            if config['start_up']['increase']:
                if args.dataset == 'BAST-Stream':
                    alivedict = pickle.load(open('datasets/sensor_graph/alives.pkl','rb'))
                    alive_cur = alivedict[str(i)]
                    alive_pre = alivedict[str(i-1)]
                    increased = [alive_cur.index(i) for i in alive_cur if i not in alive_pre]
                    node_list.extend(increased)
                else:
                    cur_node_size = np.load(
                        config['data_args']['adj_data_path'] + str(i) + '_adj.npz')['x'].shape[0]
                    pre_node_size = np.load(
                        config['data_args']['adj_data_path'] + str(i-1) + '_adj.npz')['x'].shape[0]
                    node_list.extend(list(range(pre_node_size, cur_node_size)))
                # print('increase', len(node_list))
                
            if config['start_up']['detect']:
                pre_data = np.load(
                     data_dir + '_' + str(i-1) + '/' + str(i-1)+".npz")["x"]
                cur_data = np.load(
                     data_dir_year + '/' + str(i)+".npz")["x"]
                if args.dataset == 'BAST-Stream':
                    pre_graph = np.array(list(nx.from_numpy_matrix(np.load(
                        config['data_args']['adj_data_path'] + 
                        'adj_BAST_' + str(i-1) + '.npz')["x"]).edges)).T
                    cur_graph = np.array(list(nx.from_numpy_matrix(np.load(
                        config['data_args']['adj_data_path'] + 
                        'adj_BAST_' + str(i) + '.npz')["x"]).edges)).T
                else:   
                    pre_graph = np.array(list(nx.from_numpy_matrix(np.load(
                        config['data_args']['adj_data_path'] 
                        + str(i-1) + '_adj.npz')["x"]).edges)).T
                    cur_graph = np.array(list(nx.from_numpy_matrix(np.load(
                        config['data_args']['adj_data_path'] 
                        + str(i) + '_adj.npz')["x"]).edges)).T
                vars(args)["topk"] = int(0.01*args.graph_size)
                influence_node_list = detect.influence_node_selection(
                    model, args, pre_data, cur_data, pre_graph, 
                    cur_graph, timeinday=model_args['time_in_day'])
                node_list.extend(list(influence_node_list))
                # print('detect', len(node_list))
            
            if config['start_up']['replay']:
                # int(0.2*args.graph_size)- len(node_list)
                vars(args)["replay_num_samples"] = int(0.09*args.graph_size)
                logging.info(
                    "[*] replay node number {}".format(args.replay_num_samples))
                replay_node_list = replay.replay_node_selection(
                    args, dataloader, model)
                node_list.extend(list(replay_node_list))
                # print('replay', len(node_list))

            node_list = list(set(node_list))
            if len(node_list) > int(0.1*args.graph_size):
                node_list = random.sample(node_list, int(0.1*args.graph_size))
            
            # Obtain subgraph of node list
            if args.dataset == 'BAST-Stream':
                cur_graph = np.array(list(nx.from_numpy_matrix(np.load(
                        config['data_args']['adj_data_path'] + 
                        'adj_BAST_' + str(i) + '.npz')["x"]).edges)).T
                edge_list = list(nx.from_numpy_matrix(np.load(
                    config['data_args']['adj_data_path'] + 
                        'adj_BAST_' + str(i) + '.npz')["x"]).edges)
            else:
                cur_graph = np.array(list(nx.from_numpy_matrix(
                    np.load(config['data_args']['adj_data_path'] 
                    + str(i) + '_adj.npz')["x"]).edges)).T
                edge_list = list(nx.from_numpy_matrix(
                    np.load(config['data_args']['adj_data_path'] 
                    + str(i) + '_adj.npz')["x"]).edges)
            graph_node_from_edge = set()
            for (u, v) in edge_list:
                graph_node_from_edge.add(u)
                graph_node_from_edge.add(v)
            node_list = list(set(node_list) & graph_node_from_edge)
            # print(len(node_list))
            if len(node_list) != 0:
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(
                    node_list, num_hops=args.num_hops, edge_index=torch.LongTensor(cur_graph), 
                    relabel_nodes=True)
                vars(args)["subgraph"] = subgraph
                vars(args)["subgraph_edge_index"] = subgraph_edge_index
                vars(args)["mapping"] = mapping
                logging.info("number of increase nodes:{}, nodes after {} hop:{}, \
                        total nodes this year {}".format
                        (len(node_list), args.num_hops, args.subgraph.size()[0], args.graph_size))
            vars(args)["node_list"]    = np.asarray(node_list)
            vars(args)["full_model"]   = D2STGNN(**model_args)
            graph = nx.Graph()
            graph.add_nodes_from(range(args.subgraph.size(0)))
            graph.add_edges_from(args.subgraph_edge_index.numpy().T)
            adj = nx.to_numpy_array(graph)
            adj_mx, adj_ori = load_adj(adj, config['data_args']['adj_type'],
                                is_npz=False, is_arr=True)
            model_args['num_nodes']     = adj_mx[0].shape[0]
            model_args['adjs']          = [torch.tensor(i).to(device) for i in adj_mx]
            model_args['adjs_ori']      = torch.tensor(adj_ori).to(device)
            model = D2STGNN(**model_args).to(device)
            # print(args.subgraph[mapping].size())
            # print(args.subgraph[mapping].size())
            # print(args.subgraph)
            # print(args.subgraph.size())
            # print(adj_ori.shape)
            engine  = trainer(scaler, model, **optim_args)
# ========================== Train ============================================================== #       
        # training init: resume model & load parameters
        early_stopping = EarlyStopping(optim_args['patience'], save_path)
        mode = config['start_up']['mode']
        assert mode in ['test', 'resume', 'scratch']
        resume_epoch = 0
        if mode == 'test':
            if args.cur_year > args.begin_year and args.strategy == 'incremental':
                tmp_timestr = '_2023-01-01__17_38_08'
                save_path       = './output/' + config['start_up']['model_name'] + "_Stream_" \
                                + str(i)+ "_" + dataset_name + '_Strategy_' + str(args.strategy) \
                                + '_detect_' + str(config['start_up']['detect']) + tmp_timestr + ".pt" 
                args.full_model = load_model(args.full_model, save_path)        # resume best
                engine.test(args.full_model, save_path_resume, device, dataloader, scaler, 
                    model_name, args, year=str(i),  _max=_max, _min=_min, 
                    loss=engine.loss, dataset_name=dataset_name) 
                continue
        else:
            if mode == 'resume':
                resume_epoch = config['start_up']['resume_epoch']
                model = load_model(model, save_path_resume)
            else:       # scratch
                resume_epoch = 0
        
        batch_num   = resume_epoch * len(dataloader['train_loader'])     # batch number (maybe used in schedule sampling)

        engine.set_resume_lr_and_cl(resume_epoch, batch_num)
# =============================================================== Training ================================================================= #   
        if mode != 'test':
            # continue
            trainAYear(model=model, resume_epoch=resume_epoch, optim_args=optim_args, engine=engine,
                    dataloader=dataloader, train_time=train_time, val_time=val_time, device=device,
                model_name=model_name,  _max=_max, _min=_min, early_stopping=early_stopping, 
                save_path_resume=save_path_resume, scaler=scaler, dataset_name=dataset_name, args=args)
            logging.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
            logging.info("Average Inference Time: {:.4f} secs/epoch".format(np.mean(val_time)))
        else:
            engine.test(model, save_path_resume, device, dataloader, 
                scaler, model_name, args, year=str(i), save=False, _max=_max, _min=_min, 
                loss=engine.loss, dataset_name=dataset_name)
        last_save_path = save_path

if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end   = time.time()
    logging.info("Total time spent: {0}".format(t_end - t_start))
