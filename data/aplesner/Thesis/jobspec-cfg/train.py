from logging import warning
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import library.functions.misc_helper_functions as misc_helpers
from library.functions import model_parameters
import library.data.data_helper_functions as data_helpers
#from library.base_models import tree_models, linear_models
#from library.cnn import cnn_models
import time
import os

if __name__ == '__main__':

    args = misc_helpers.parse_args()
    pretrained_model = False
    if args.pretrained_model != "no":
        pretrained_model_params = torch.load(os.path.join(args.model_savepoint_folder, args.pretrained_model))
        args_pretrained_model = pretrained_model_params["args"]
        
        args.in_features = args_pretrained_model.in_features
        args.out_features = args_pretrained_model.out_features
        args.activation_function = args_pretrained_model.activation_function
        args.parameter_settings = args_pretrained_model.parameter_settings
        args.use_dropout = args_pretrained_model.use_dropout
        args.use_batch_normalization = args_pretrained_model.use_batch_normalization

        pretrained_model = True

    # Hyperparameters - Data
    device = args.device
    print("Device:", device)
    
    params = {'batch_size': args.batch_size,
              # 'shuffle': True,
              'drop_last': False}

    labels = [
        'LivLongSxD1 [mm]', 'LivLongDxD1 [mm]', 'AllinSxD1 [mm]', 'AllinDxD1 [mm]',
        'LivLongSxD2 [mm]', 'LivLongDxD2 [mm]', 'AllinSxD2 [mm]', 'AllinDxD2 [mm]', 
        'LivLongSxD3 [mm]', 'LivLongDxD3 [mm]', 'AllinSxD3 [mm]', 'AllinDxD3 [mm]']

    # Load hyperparameters
    hyperparameters, model = model_parameters.parameters(args)

    torch.random.manual_seed(args.seed)
    net = model(hyperparameters).to(device=args.device, dtype=args.data_type)
    if pretrained_model:
        net.load_state_dict(pretrained_model_params["model_state_dict"])

    print("Hyperparameters:")
    for (key, value) in hyperparameters.items():
        print("\t", key, " : ", value, sep="")
    print(net)

    if args.use_cv and not args.training_data_file.endswith("_v2_training.pt"):
        dataset, train_idx, val_idx = data_helpers.load_data_training(args=args, net=net)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **params)
        val_loader = torch.utils.data.DataLoader(dataset, sampler=val_sampler, **params)
    else:
        train_dataset, val_dataset = data_helpers.load_data_training(args=args, net=net)
        train_loader = torch.utils.data.DataLoader(train_dataset, **params)
        val_loader = torch.utils.data.DataLoader(val_dataset, **params)

    # Setup save files
    t = time.localtime()
    run_name = type(net).__name__+"_" + args.training_data_file[:-12] + "_{:0>2}-{:0>2}-{}_{:0>2}:{:0>2}_{}".format(
        t.tm_mday, t.tm_mon, t.tm_year, t.tm_hour, t.tm_min, args.run_id)
    loss_log = "results/epoch_logs/" + run_name + ".log"

    writer_signed = SummaryWriter('/work1/s174505/Thesis/runs/' + run_name + '_signed')#.add_hparams(hyperparameters)
    writer_unsigned = SummaryWriter('/work1/s174505/Thesis/runs/' + run_name + '_unsigned')#.add_hparams(hyperparameters)

    # Hyperparameters - optimiser and loss function
    if hyperparameters["optimizer"] == "SGD":
        print("SGD")
        optimiser = torch.optim.SGD( net.parameters(), lr=100*hyperparameters["lr"], 
            weight_decay=hyperparameters["weight_decay"])
    elif hyperparameters["optimizer"] == "nesterov": 
        print("nesterov")
        optimiser = torch.optim.SGD( net.parameters(), lr=10*hyperparameters["lr"], 
            nesterov=True, momentum=args.momentum, weight_decay=hyperparameters["weight_decay"])
    elif hyperparameters["optimizer"] == "AdamW": 
        print("AdamW")
        optimiser = torch.optim.AdamW( net.parameters(), lr=0.7*hyperparameters["lr"], 
            weight_decay=hyperparameters["weight_decay"])
    elif hyperparameters["optimizer"] == "AMSGrad": 
        print("AMSGrad")
        optimiser = torch.optim.Adam( net.parameters(), lr=2*hyperparameters["lr"], 
            amsgrad=True, weight_decay=hyperparameters["weight_decay"])
    elif hyperparameters["optimizer"] == "Adadelta":
        print("Adadelta")
        optimiser = torch.optim.Adadelta( net.parameters(), lr=100*hyperparameters["lr"]) 
    elif hyperparameters["optimizer"] == "Adagrad":
        print("Adagrad")
        optimiser = torch.optim.Adagrad( net.parameters(), lr=10*hyperparameters["lr"]) 
    elif hyperparameters["optimizer"] == "Adamax":
        print("Adamax")
        optimiser = torch.optim.Adamax( net.parameters(), lr=2*hyperparameters["lr"]) 
    elif hyperparameters["optimizer"] == "ASGD":
        print("ASGD")
        optimiser = torch.optim.ASGD( net.parameters(), lr=100*hyperparameters["lr"]) 
    elif hyperparameters["optimizer"] == "NAdam":
        print("NAdam")
        optimiser = torch.optim.NAdam( net.parameters(), lr=0.7*hyperparameters["lr"]) 
    elif hyperparameters["optimizer"] == "RAdam":
        print("RAdam")
        optimiser = torch.optim.RAdam( net.parameters(), lr=hyperparameters["lr"]) 
    elif hyperparameters["optimizer"] == "RMSprop":
        print("RMSprop")
        optimiser = torch.optim.RMSprop( net.parameters(), lr=0.7*hyperparameters["lr"]) 
    elif hyperparameters["optimizer"] == "Rprop":
        print("Rprop")
        optimiser = torch.optim.Rprop( net.parameters(), lr=2*hyperparameters["lr"]) 
    elif hyperparameters["optimizer"] == "Adam":
        print("Adam")
        optimiser = torch.optim.Adam( net.parameters(), lr=hyperparameters["lr"], 
            weight_decay=hyperparameters["weight_decay"])
    elif hyperparameters["optimizer"] == "LBFGS":
        print("LBFGS")
        optimiser = torch.optim.LBFGS( net.parameters(), lr=100*hyperparameters["lr"]) 
    else:
        optimiser = None
        warning("No optimizer chosen")
    
    if pretrained_model:
        if pretrained_model_params["parameters"]["optimizer"] == hyperparameters["optimizer"]:
            optimiser.load_state_dict(pretrained_model_params["optimizer_state_dict"])


    criterion = torch.nn.MSELoss(reduction="mean")

    # Training hyperparameters
    max_epochs = args.max_epochs

    train_batches = len(train_loader)
    val_batches = len(val_loader)

    print("Train batches:", train_batches, "\tValidation batches:", val_batches)

    # Use LR scheduler
    if args.use_lr_schedule:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[250,400], gamma=0.40)

    torch.backends.cudnn.benchmark = True
    epoch_start_time = time.time()
    min_loss_seen = torch.inf
    last_save = time.time()
    for epoch in range(max_epochs):
        if epoch%5 == 0:
            print("Starting epoch:", epoch)

        train_error_epoch = [None] * len(train_loader)
        
        # Train loop
        net.train()
        for idx, (x, y) in enumerate(train_loader):
            def closure():
                if torch.is_grad_enabled():
                    optimiser.zero_grad()
                yhat = net(x)
                batch_mse = criterion(yhat, y) 
                if batch_mse.requires_grad:
                    batch_mse.backward()
                return batch_mse
            optimiser.step(closure)

            train_error_epoch[idx] = net(x) - y
        train_error_epoch = torch.concat(train_error_epoch, 2)

        # Validation loop
        val_mse_epoch = torch.zeros((args.out_features, val_batches))

        net.eval()
        with torch.no_grad():
            val_error_epoch = torch.concat([net(x) - y for x,y in val_loader], 2)
            
        misc_helpers.log_epoch_loss(loss_log=loss_log, epoch=epoch, 
            train_loss=train_error_epoch, val_loss=val_error_epoch, labels=labels, 
            writer_signed=writer_signed, writer_unsigned=writer_unsigned)

        if (epoch+1)%100 == 0:
            print("Epochs time: {:.2f} s".format(time.time() - epoch_start_time))
            epoch_start_time = time.time()
        
        if args.use_lr_schedule:
            scheduler.step()
        
        val_me = np.mean(np.abs(val_error_epoch.detach().cpu().numpy()))
        # End of final epoch
        if epoch == max_epochs - 1:
            misc_helpers.save_model_and_optim(epoch, net, val_error_epoch, optimiser, hyperparameters, args, t, final=True)
        elif val_me < min_loss_seen and (time.time() - last_save) > 120:
            last_save = time.time()
            min_loss_seen = val_me
            misc_helpers.save_model_and_optim(epoch, net, val_error_epoch, optimiser, hyperparameters, args, t, final=False)


