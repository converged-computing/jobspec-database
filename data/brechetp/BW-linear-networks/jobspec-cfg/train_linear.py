import torch
import utils
import plot
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import pandas as pd
import csv
import math
import argparse
import sys
import datetime
import re
import random
import pickle
from collections import defaultdict
import traceback  # for tracebacks of exceptions

# the default arguments
DEFAULTS = {
    "output": "results/",
    "name": "",
    "vary-name": "depth/smin",
    "depth": 10,
    "width": 20,
    "zdim": 20,
    "xdim": 20,
    "learning-rate": 1e-5,
    "time": 5,
    "num-iter": 1000,
    "init": "close",  # initialize close to the solution
    "init-scheme": "balance-force",
    "gamma": 1,
    "std": 0.1,
    "smin": -1,  # no restriction on the minium singular value of the target
    "tau": 0,
    "grad-comp": "manual", # one of manual, backprop A, backprop B
    "grad-rank": "full",  # one of full, adapt
    "save-dataset": True,
    "dataset-fname": None,
    "new-dataset": False,
    "rank-target": None,
    "load-dataset": True,
    "loss": "BW",  # the loss to consider, BW or Fro
    "samp-mode":"eigen",  # the sampling modality for the target , eigen or singular values
    "samp-distrib": "zipf",  # the distribution used to sample the target
    "save-every": 0.1,  # save every 10% iterations
    "cvg-test": True,
}


def train(args):

# args = dict()  # in order to save the arguments
    # args.update(kwargs)  # update the defaults with the input arguments
    redo = False

    depth= args.depth     # number of matrices in the network
    zdim= m = din =  args.zdim
    xdim= n = dout = args.xdim
    width= args.width




# dataset_fname = "data/1gaussian/dim-20-N-500_1.pkl"
    dataset_fname = args.dataset_fname
    save_datset = args.save_dataset
    new_dataset = args.new_dataset


    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpu_index = args.gpu_index % num_gpus if args.gpu_index is not None else random.choice(range(num_gpus))
    else:
        gpu_index = 0
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu', gpu_index)
    grad_rank=  args.grad_rank      # one of "adapt", "full"

    lr= args.learning_rate


    tau = args.tau #1e-1  # regularization parameters
    smin = args.smin

    init= args.init

    # balance = args.balance
    ischeme = args.init_scheme
    gamma= args.gamma  #(1-1/(depth))
    if args.std is not None:
        std = args.std
    else:
        if ischeme == "balance-force":  # initialization is performed on the end-to-end matrix
            std = zdim ** (-gamma / 2)  # take the output dimension as width
        else:
            std = width ** (-gamma / 2)

    loss_name = args.loss  # either BW or Fro
# gamma = 1



    vary_name = None
    fullname = args.name
    if args.vary_name is not None:
        vary_name = utils.get_name(args, parser)
        fullname = os.path.join(args.name, vary_name)

    if args.debug:
        args.output = os.path.join(args.output, "debug")

    # fullname = args.nam
    outroot  = args.output
    OUTPATH = os.path.join(args.output, fullname)
    # os.makedirs(OUTPATH, exist_ok=True)


    os.makedirs(OUTPATH, exist_ok=True)
    PLOTDIR = os.path.join(OUTPATH, "plot")
    os.makedirs(PLOTDIR, exist_ok=True)

# Data generation for the samples
    trainset = utils.get_toy_trainset(
        dataset_fname,
        new_dataset,
        save_datset,
        xdim,
        smin=smin,
    samp_distrib=args.samp_distrib,
    samp_mode = args.samp_mode)

    args.dataset_fname = trainset.fname


    # save the target eigenvalues to a plot
    plt.figure()
    plt.plot(trainset.Lambda, '+', label="Eigenvalues target")
    plt.title(fullname)
    plt.legend()
    fname = os.path.join(PLOTDIR, "target.pdf")
    plt.savefig(fname=fname)
    plt.close()

    # save the configuration
    with open(os.path.join(OUTPATH, 'config.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=0, sort_keys=True)

    # logfile = open(os.path.join(OUTPATH, "logs.txt"), 'a') # in roder to write to the log file
    resf = open(os.path.join(OUTPATH, 'results.csv'), 'a')  # to log the values as csv
    logf = open(os.path.join(OUTPATH, 'logs.txt'), 'a')
    errf = open(os.path.join(OUTPATH, 'err.txt'), 'a')

    if not args.debug:
        sys.stdout = logf
        sys.stderr = errf

# print('Init....')
    start = datetime.datetime.now()
    print("start at ", start)

        # smin = 15)
# testset
    Lambda, Omega = trainset.Lambda, trainset.Omega

# trainloader = DataLoader(
        # trainset,
        # batch_size=batch_size,
        # shuffle=False)  # no SGD

# the eigenvalue decomposition of Sigma0
    rank_target = xdim
    r = min(zdim, xdim, rank_target)  # minimum
    # Lambda, Omega = torch.linalg.eigh(Sigma0)
    Sigma0= trainset.Sigma.to(device)
    R0 = trainset.Root.to(device)
    # Sigma0 = (Omega @ (Lambda.view(-1, 1) * Omega.t())).to(device)
    # R0 = (Omega @ (Lambda.view(-1, 1).sqrt() * Omega.t())).to(device)
    # Lambda and Omega are already in the descending order
    # Lambda = Lambda.flip([0])  # flip the values (descending order)
    # Omega = Omega.flip([1])  # flip along the columns to match the eigenvalues
    sminstar = Lambda[r-1].sqrt().item() # minimum eigenvalue of R0
    # smin2star = Lambda[dmin-1].sqrt().item()
    erank_R0 = utils.compute_erank(R0)

    #init_lowrank = False  # if true, will initialize the network to a lower rank (e.g. 1)

# takes the minimum between tau and the smallest eigenvalue
    tau = min(tau, Lambda[r-1].item())
    print("value of tau:", tau)

    if init == "close":
        init_kwargs = {"target": Sigma0, "Lambda": Lambda, "Omega": Omega, "scale": 10.95, "tau": tau}
    else:
        init_kwargs = {"std":  std}  # will be useful only if init == "normal"

# model construction  and initialization
    model = utils.create_network(
        "const",
        depth,
        din,
        width,
        dout,
        init_name=init,
        init_scheme=ischeme,
        save=True,
        **init_kwargs,
    )
    print("Model: ", model)
    S = torch.linalg.svdvals(model.end_to_end())
    plt.plot(S.detach().cpu().numpy()**2, '+', label="Eigenvalues")
    plt.suptitle(fullname)
    plt.title("Eigenvalues of W at initialization")
    plt.legend()
    fname = os.path.join(PLOTDIR, "model-init.pdf")
    plt.savefig(fname=fname)
    plt.close()

    model.to(device)
    dmin = min(model.widths)  # bottleneck
    lambda_dmin = Lambda[dmin-1]  # last eigenvalue that we can possibly learn
    #full_row_rank = dmin == xdim

    #if init_lowrank:
     #   model.init_lowrank()

    Sigma0np = Sigma0.detach().cpu().numpy()
    R0np = R0.detach().cpu().numpy()  # put the square root on the cpu

# Loss of the network
    def LossBW(Sigma, tau=tau, mask=None, Sigma0=Sigma0, R0=R0, with_grad=False):
        """use the differentiable sqrtm function"""
        with torch.set_grad_enabled(with_grad):

            #global Sigma0, R0
            Sigma0 = Sigma0.to(Sigma.device)

            n = Sigma.size(0)
            #if mask is None:
            #    mask = tau == 0
            if R0 is None:
                R0 = utils.sqrtm(Sigma0)
                R0 = (R0 + R0.T)/2
            else:
                R0 = R0.to(Sigma.device)
            # assert sp.linalg.issymmetric(R0), "R0 not symmetric"
            # assert sp.linalg.issymmetric(Sigma0), "Sigma0 not symmetric"
            I = torch.eye(n).to(device)
            M = R0 @ (Sigma + tau*I) @ R0
            M = (M + M.T)/2
            # assert sp.linalg.issymmetric(M), "M not symmetric"
            # return torch.trace(Sigma + tau* I + Sigma0 - 2*(utils.sqrtm(R0 @ (Sigma + tau * I) @ R0, mask=mask)))
            return max(torch.trace(Sigma + tau*I + Sigma0 - 2* utils.sqrtm(M)), 0)

    def LossFro(Sigma, tau=tau, mask=None, Sigma0=Sigma0, R0=R0np):

        #global Sigma0, R0
        n = Sigma.size(0)
        Sigma0 = Sigma0.to(Sigma.device)
        #if mask is None:
        #    mask = tau == 0
        return torch.linalg.norm(0.5*(Sigma - Sigma0))**2
        # if R0 is None:
            # R0 = sp.linalg.sqrtm(Sigma0)
            # R0 = (R0 + R0.T)/2
        # # assert sp.linalg.issymmetric(R0), "R0 not symmetric"
        # # assert sp.linalg.issymmetric(Sigma0), "Sigma0 not symmetric"
        # I = np.eye(n)
        # M = R0 @ (Sigma + tau*I) @ R0
        # M = (M + M.T)/2
        # # assert sp.linalg.issymmetric(M), "M not symmetric"
        # # return torch.trace(Sigma + tau* I + Sigma0 - 2*(utils.sqrtm(R0 @ (Sigma + tau * I) @ R0, mask=mask)))
        # return max(np.trace(Sigma + tau*I + Sigma0 - 2* sp.linalg.sqrtm(M).real), 0)
# value at initialization
    if args.loss == "BW":
        Loss = LossBW
    else:
        Loss = LossFro

    W = model.end_to_end()

    Sigma = (W @ W.T)

    #c = min(math.sqrt(lambda_dmin), sigma_r)  # assumption: the value during training won't be smaller than those two exreme points
    smin2star = math.sqrt(lambda_dmin)  # simply the lowest singluar value  for W
    c = smin2star - math.sqrt(Loss(Sigma, tau))
    if args.init == "close":  # reduce the radius of initilization so that we close enough to the target (smin - c away)
        while c < 0:
            print("c :", c)
            init_kwargs["scale"] *= 0.95
            model.balance_weights(utils.init_close, **init_kwargs)
            W = model.end_to_end()
            Sigma = (W @ W.T)
            c = smin2star - math.sqrt(Loss(Sigma, tau))
    else:
        c = max(c, smin2star)  # take a positive c, but not so meaningful

    R = utils.sqrtm(Sigma) # square root of Sigma
    X = R0 @ R   # product of the square roots
# compute the maximal value for the c parameter (should be > 0)
    V1, S, V2t = torch.linalg.svd(X, full_matrices=False)
    V = V1 @ V2t # the unitary matrix in the polar decomposition of X
    U0, S0, V0h = torch.linalg.svd(W, full_matrices=False)  # SVD at initialization
    sigma_r = S0[dmin-1]  # already in descending order

    cmax = sigma_r - (R + R0@V).norm()  # smin is the smallest singular value for the target projected to the set of linear functions we can learn using the network
    cmin = sigma_r - (R - R0@V).norm()


    vals = defaultdict(float)
    # vals['c'] = c  # minimum singular value the network can learn (with more than one layer)
    vals['error'] = False
    vals['smin'] = smin  # minimum singular value if the network was one layer
    vals['smin*'] = sminstar  # minimum singular value if the network was one layer
    vals['smin**'] = smin2star

    # prediction for gradient flow
    Loss0 = Loss(Sigma, tau).item()  # value at initialization
    C_0 = 2*(Loss0 + Lambda.sum())  # upper bound for the norm of the weights
    K = math.sqrt(tau * Lambda[-1]) / (2 * C_0 ** 2)  # strongly-convex constant
    N = model.depth
    Pred_C = (-8*N*c**((2*(2*N-1))/N) * K).item()
    OPT =  Omega[:, :dmin] @ ((Lambda[:dmin].view(-1, 1) - tau) * (Omega[:, :dmin].t()))
    OPT = (OPT + OPT.T) / 2  # symmetric matrix

    L_OPT = Loss(OPT.to(device), tau)
    L_OPT_theo = (Lambda + tau - 2*(tau*Lambda).sqrt())[dmin:].sum().item() # smallest eigenvalues
    vals['Loss0'] = Loss0
    vals['C_0'] = C_0
    vals['K'] = K
    vals['Pred_C'] = Pred_C
    vals['L_OPT'] = L_OPT
    vals['L_OPT_theo'] = L_OPT_theo
    vals['c'] = c
    print("Diff in L_OPT: ", (L_OPT - L_OPT_theo))

    print("erank target (R0):", erank_R0)
    print("c, cmin, cmax: ", c, cmin.item(), cmax.item())
    print("K, C_0, Pred_C, L_OPT:", K.item(), C_0.item(), Pred_C, L_OPT)


    # prediction for gradient descent

    M = math.sqrt(C_0)
    eta_1 = c ** 2 / (8 * M * math.sqrt(Loss0))
    eta_3 = 1 / (4 * N * c ** ( 2 * (N-1) / N))
    try:
        Delta = 2**(N+1) / (c ** (2*N)) * N**2 * M**((4*N-3) / N) * Lambda[0].sqrt() + 8 *N * (N-1) * M**((3*N-4)/N) * (M **(1/N) + Lambda.sum())
        eta_2 = N * c ** (2 * (N-1) / N) / (2 * Delta)
        eta = min(eta_1, eta_2, eta_3)
    except ZeroDivisionError as e:
        traceback.print_exc()
        print("continue...")
        Delta = eta_2 = np.nan
        eta = min(eta_1, eta_3)

    # lr = min(eta_1, eta_3, lr)
    lr = lr
    if args.num_iter is not None:
        num_iter= args.num_iter
    else:
        num_iter = int(args.time / lr + 0.5)

    if args.cvg_test:  # if we stop trainng after convergence test
        save_every = int(args.save_every / lr + 0.5)  # take save_every as
    else:
        save_every = int(args.save_every) if (args.save_every  >= 1 or args.save_every <= 0) else int(num_iter * (args.save_every) +0.5)

    print("Delta, M:", Delta, M)
    print("eta_1, eta_2, eta_3: ", eta_1, eta_2, eta_3)

    print("eta:", eta)

    vals['Delta'] = Delta
    vals['M'] = M
    vals['eta_1'] = eta_1
    vals['eta_2'] = eta_2
    vals['eta_3'] = eta_3
# SGD but will perform full batch gradient
    optimizer = optim.SGD(model.parameters(), lr=lr)

    print("Optimizer: ", optimizer)

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)  # halfs the learning rate every step size
# model = models.LinearNetwork(widths)  # the actual model


# # The training of the network



    def write_gradient(model):
        """Compute and write the gradients to the parameters"""

        grads = model.compute_grads(R0, loss=loss_name, tau=tau)
        for i, p in enumerate(model.parameters()):
            p.grad = grads[i]  # / NS  # first time the gradient is None
        return grads


    x_fixed = trainset.sample(100)  # 100 points for plotting
    z_fixed = torch.randn(100*1, model.din)  # zero centered
    prev_n = niter = 0
    y_fixed_0 = model(z_fixed.to(device))  # original generated data


# T = (num_iter - start_iter) // len(trainloader)

# W = model.end_to_end()
# rank_W = torch.linalg.matrix_rank(W).item()
# m = len(trainloader.dataset)  # number of points in the training data
# samples = []
# while niter <  + start_n:
# for ep in range(start_ep, num_iter + start_ep):
# The computation of the loss

    # numerical comparison of the gradient as computed with the closed-form
    # solution and the gradient computed with the backpropagation
    def check_gradient(gen):
        # check the gradient computed in closed form
        # has to check for every parameter? first check for the end-to-end
        # matrix
        grads = gen.compute_grads(R0, loss=loss_name, tau=tau, mode=grad_rank)
        max_diff = 0
        for i, p in enumerate(gen.parameters()):
            max_diff = max(max_diff, (p.grad - grads[i]).norm(p='fro').item())
        return max_diff


    def compute_gan_loss(model, loss_name=loss_name):
        #global R0, Sigma0, n, m
        W = model.end_to_end()
        # R0 = R0.to(W.device)  # can't change the local variables inside the
        # Sigma0 = Sigma0.to(W.device)
        loss_val = None
        if loss_name == "BW":
            if tau > 0:
                A = R0 @ W @ (R0 @ W).t() + tau * Sigma0  # regularization strength
                sqrtA = utils.sqrtm(A)
                loss_val = W.norm(p='fro').pow( 2) + tau*n + R0.norm(p='fro').pow(2) - 2*torch.trace(sqrtA)
            elif tau == 0.:  # can compute the loss based on the SVD of R0 @ W
                r = min(n, m)
                U, S, Vh = torch.linalg.svd(R0.mm(W), full_matrices=False)
                loss_val = W.norm(p='fro').pow( 2) + R0.norm(p='fro').pow(2) - 2*S[:r].sum()
        elif loss_name == "Fro":
            loss_val = (W.mm(W.t()) - Sigma0).norm(p='fro').pow(2)
        else:
            return NotImplementedError()

        return loss_val



    # all the scalar quantities
    columns = pd.Index(['time', 'loss', 'balance', 'norm grad',
                        'distance values', 'U - Omega', "WW' - Sigma0", "max diff grad", "rank W", "erank W",
                        "diff loss to OPT", "diff loss to OPT theo", "wass. dist to OPT", "euc. dist to OPT"])

    df = pd.DataFrame(columns=columns, index=pd.Index([], name="niter"))
    df = df.astype(float)
    df = df.astype({"rank W": int})

    fields = [df.index.name] + list(df.columns)

    csv_writer = csv.DictWriter(resf, fieldnames = fields)  # the writer to the csv file
    csv_writer.writeheader()

    # the dataframe for the 2D cosine values
    df_cos = pd.DataFrame(columns=["cos(U, Omega)"])
    df_cos = df_cos.astype(object)
    df_cvg = pd.DataFrame(columns=pd.Index(np.arange(1, dmin+1), name="index"))
    # stats = {'lossA': [], 'lossB': [], 'balance': [], 'norm grad': [],


    def get_checkpoint():
        '''Get current checkpoint'''

        checkpoint = {
            'model': model.state_dict(),
            # 'df': df,
            # 'args' : args,
            'optimizer': optimizer.state_dict(),
            # 'lr_scheduler':lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'niter': niter,
                    }

        return checkpoint

# def save_checkpoint(checkpoint=None, name=None, fname=None):
    def save_checkpoint(prev_n=0, checkpoint=None):
        '''Save checkpoint to disk'''


        # global prev_n # OUTPATH, ep, prev_n, niter

        CHKPTDIR = os.path.join(OUTPATH, "checkpoints")
        os.makedirs(CHKPTDIR, exist_ok=True)
        fname_prev_checkpoint = os.path.join(CHKPTDIR, f"{prev_n}.pth")
        if os.path.isfile(fname_prev_checkpoint):
            os.remove(fname_prev_checkpoint)

        # name=f"{niter}.pth"

        fname = os.path.join(CHKPTDIR, f"{niter}.pth")

        # if fname is None:
            # fname = os.path.join(OUTPATH, name + '.pth')

        if checkpoint is None:
            checkpoint = get_checkpoint()

        torch.save(checkpoint, fname)
        return niter

            # 'cos(U, Omega[idx])':[], "cos(U, Omega)": [], "distance values": [],
    def save_dfs():
        '''Save dataframes to disk'''

        name=f"dfs.pkl"

        fname = os.path.join(OUTPATH, name)
        dfs = {'quant': df, 'vals': vals, "cos": df_cos, "cvg": df_cvg, 'trainset': trainset}#, "balance": df_balance, "jacob_svd":df_jacob_svd}

        with open(fname, "wb") as _f:
            pickle.dump(dfs, _f)
        return


    converged = False
    try:
        while not converged:
        # for niter in mtqdm(range(num_iter)):
            model.zero_grad()
            stats = dict()  # for the current iteration
            # the keys in stats have to be the same as the headaer of the CSV file
            # the different losses computed with sqrtm (lossA) or SVD (lossB)
            # grad_comp specifies how the gradient is computed (backprop, etc,...)
            loss_val = compute_gan_loss(model, loss_name=loss_name)
            # backpropagation is performed if grad_comp is backprop, depending on
            # the loss that was computed (lossB is preferred to lossA)
            write_gradient(model)  # NS is to divide by the total number of samples

            # the balance of the layers
            balance = model.compute_balance()

            grad = model.get_gradient()  # the gradient used in the update of the model, flattened
            # perform one step after all the samples have been seen
            stats['rank W'] = model.compute_rank()
            stats['erank W'] = model.compute_erank()  # the effective rank (exponential entropy of the sv distribution)
            optimizer.step()  # update the models parameters

            # save_this = (niter % save_every == 0) or (last_iter)
            stats['loss'] = loss_val.item()
            stats['balance'] = max(balance) if balance else 0 # take the max over all the layers
            stats['norm grad'] = grad.norm().item()
            stats['time'] = niter * lr

            # if niter >= 30000:
            # scheduler.step()


            W = model.end_to_end().detach().cpu()
            # compare the current estimate to the expression of critical points
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            # n, k = U.size()
            n = W.size(0)
            k = dmin #bottlneck dimension #torch.linalg.matrix_rank(W)  # rank of W
            distance_values = (S[:k].pow(2) - (Lambda - tau)[:k])
            cos =  (Omega.view(n, -1, 1) * U.view(n, 1, -1)).sum(dim=0).detach().numpy()
            df_cos.loc[niter, 'cos(U, Omega)'] = cos  # the cosine 2d data to the dataframe
            # if U.size() == Omega.size():
                # stats['cos(U, Omega)'][niter] = (U.flatten().dot(Omega.flatten()) / (U.norm() * Omega.norm())).item()
                # stats['norm(U-Omega)'][niter] = (U-Omega).norm().item()
            # stats['norm(U-Omega[idx])'][niter] = UOmega_norm
            df_cvg.loc[niter, :] = distance_values
            stats['distance values'] = distance_values.norm().item()
            stats["WW' - Sigma0"] = (W@W.t() + tau*torch.eye(xdim) - Sigma0.cpu()).norm().detach().item()
            Sigma = W @ W.t()
            smin_W = torch.linalg.svdvals(W)[dmin-1]
            R = utils.sqrtm(Sigma)
            X = R0.cpu() @ R  # on the cpu
            V1, S, V2t = torch.linalg.svd(X, full_matrices=False)  # the SVD to compute the polar decomposition
            V = V1 @ V2t
            # stats['upper bound']  = (R + R0 @ V).norm().pow(2).item()
            # stats['lower bound'] = (R - R0 @ V).norm().pow(2).item()

            # stats['c diff'] = (smin_W - c).item()
            # stats['cmin'] = (smin - (R - R0@V).norm()).item()
            # stats['cmax'] = (smin - (R + R0@V).norm()).item()
            stats['diff loss to OPT'] = (loss_val - L_OPT).abs().item()
            stats['diff loss to OPT theo'] = (loss_val - L_OPT_theo).abs().item()
            stats['euc. dist to OPT'] = (W.mm(W.t()) - OPT).norm().item()
            stats['wass. dist to OPT'] = LossBW(W.to(device).mm(W.T.to(device)), Sigma0=OPT, R0=None, tau=tau).detach().cpu().item()

            # converged |= stats['wass. dist to OPT'] <= EPSILON

            df.loc[niter, :] = stats  # save the data to the data frame
            csv_writer.writerow(stats)  # to write the csv file

            niter += 1
            converged = niter > num_iter  # regular test
            delta_iter = int(0.1/lr+0.5)  # interation delta to test convergence of the learning
            if args.cvg_test:  # test the convergence
                if len(df) >= delta_iter:
                    converged |= df['diff loss to OPT'].iloc[-delta_iter] < 1e-4
            # last_iter = niter >= num_iter
            save_this = (save_every > 0 and niter % save_every == 0) or (converged)

            if save_this:
                save_dfs()  # write the dataframes to disk
                prev_n = save_checkpoint(prev_n)


                t = df.index.to_numpy()
                prediction_GF  = Loss0 * np.exp(Pred_C * t)
#
                # Plotting

                fig_scatter, axes_scatter = plt.subplots(1, 2, figsize=(10, 5))
                fname = os.path.join(PLOTDIR, "scatter.pdf")
                plot.scatter_samples(
                    None,
                    x_fixed,
                    y_fixed_0,
                    axtitle="niter 0",
                    title="Target and generated samples (2D slice)",
                    fig=fig_scatter,
                    ax=axes_scatter[0])
                y_fixed = model(z_fixed.to(device))
                axtitle = "niter {}".format(niter)
                plot.scatter_samples(
                    None,
                    x_fixed,
                    y_fixed,
                    axtitle=axtitle,
                    fig=fig_scatter,
                    ax=axes_scatter[1])
# plt.show()
                plt.tight_layout()
                fig_scatter.savefig(fname)

                fig = plt.figure()
                # plt.plot(df['loss A'], label="loss A")
                plt.plot(df['loss'], label=['loss'])
                plt.legend()
                ax = plt.gca()
                ax.ticklabel_format(style="sci", useMathText=True)
                plt.title(fullname)
                #f save:
                fname = os.path.join(PLOTDIR, "loss.pdf")
                fig.savefig(fname, bbox_inches="tight")
                # plt.plot(df["upper bound"], label="upper bound")
                # plt.plot(df["lower bound"], label="lower bound")
# plt.plot(df['lossB'], label="loss B")
# plt.show()

                # plt.figure()
                # plt.plot(df["cmin"], label="cmin")
                # plt.plot(df["cmax"], label="cmax")
                # plt.legend()

                # plt.figure()
                # plt.plot(df['norm grad'], label="norm grad")
                # plt.legend()

                fig = plt.figure()
                plt.plot(df['rank W'], label="rank W")
                plt.plot(df['erank W'], label="erank W")
                plt.xlabel("niter")
                ax = plt.gca()
                ax.ticklabel_format(style="sci", useMathText=True)
                xmin, xmax = plt.xlim()
                plt.hlines(erank_R0, xmin, xmax,  colors=['g'], linestyle="dashed", label="erank R0")
                plt.legend()
                plt.title(fullname)
                # if save:
                fname = os.path.join(PLOTDIR, "ranks.pdf")
                fig.savefig(fname, bbox_inches="tight")

                fig = plt.figure()
                plt.plot(df["diff loss to OPT"], label="diff loss to OPT")
                plt.plot(df["diff loss to OPT theo"], label="diff loss to OPT theo")
                plt.plot(df["euc. dist to OPT"], label="euc. dist to OPT")
                plt.plot(df["wass. dist to OPT"], label="wass. dist to OPT")
                plt.plot(prediction_GF, label="prediction GF")
                plt.xlabel("niter")
                plt.legend()
                plt.yscale('log')
                fig.tight_layout()
                plt.title(fullname)
                # if save:
                fname = os.path.join(PLOTDIR, "diff_to_opt.pdf")
                fig.savefig(fname, bbox_inches="tight")

                # fig = plt.figure()
                # plt.plot(df["c diff"], label="c diff")
                # plt.legend()
                # fig.tight_layout()
                # plt.title(name)
                # if save:
                    # fname = os.path.join(PLOTDIR, "c_diff.pdf")
                    # fig.savefig(fname, bbox_inches="tight")
                fig, ax = plt.subplots()
                fname = os.path.join(PLOTDIR, "dist_eigvals.pdf")
                df_cvg.plot(ax=ax)
                fig.savefig(fname, bbox_inches="tight")



                if len(df_cos['cos(U, Omega)']) > 1 and niter > 0:
                    fname = os.path.join(PLOTDIR, "cos_matrix.pdf")
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        # # todo: update with the dataframe instead of the list
                    # else:  # only plot the last iteration
                    # the last iteration as a 2d plot
                    hm = axes[0].imshow(df_cos['cos(U, Omega)'][niter-1], vmin=-1, vmax=1, aspect='auto')
                    axes[1].plot(df['distance values'], label="Eigenvalues distance")
                    axes[1].set_xlabel("niter")
                    axes[1].legend()

                    axes[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))  # force integer ticks
                    axes[0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))  # force integer ticks
# axes[0].set_xticks(range(1, min(n, m)+1))
# axes[0].set_yticks(range(1, n+1))
                    divider = make_axes_locatable(axes[0])  # in order to scale the colorbar
                    cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(hmap, ax=axes[0], location='right')
                    plt.colorbar(hm, cax=cax)

                    fig.tight_layout()

                    fname = os.path.join(PLOTDIR, "cos_matrix.pdf")
                    fig.savefig(fname, bbox_inches="tight")

                plt.close('all')
# axes[1].plot(stats['distance values'], label="distance values")
# axes[1].plot(stats["norm(U-Omega)"], label="norm(U-Omega)")
# axes[1].plot(stats["norm(U-Omega[idx])"], label="norm(U-Omega[idx])")
# axes[1].legend()

                # fig, axes = plt.subplots(1, 1, figsize=(5, 5))
                # axes.plot(df["WW' - Sigma0"], label="|WW' + tau*I - Sigma0|")
                # axes.legend()



    # plt.show()

    except Exception as e:
        vals['error'] = True
        traceback.print_exc()


        # log some informations
        # with torch.no_grad():
        # W = model.end_to_end()
        # R = utils.sqrtm(W.mm(W.t()))
        # n, m = W.size()
        # norm_W = W.norm(p='fro').item()
        # # dvc = R0.device
        # # R0 = R0.to(W.device)
        # r = rank_W = torch.linalg.matrix_rank(R0 @ W).item()

        # k = rank_W
        # Uk, Sk, Vhk = torch.linalg.svd(Sigma0, full_matrices=False)
        # SigApp = Uk[:, :k] @ torch.diag(Sk[:k]) @ Vhk[:k, :]

        # # det_WWt = torch.linalg.det(W.mm(W.t())).item()

        # Sigma = W @ W.t() + tau * torch.eye(n)


# end of training


    W = model.end_to_end()
    Sigma = W @ W.t()
    eig = torch.linalg.eigvalsh(Sigma)
    plt.plot(trainset.Lambda, '+', label="Eigenvalues target")
    plt.plot(eig.flip(0).detach().cpu().numpy(), 'x', label="Eigenvalues model")
    plt.title(fullname)
    plt.legend()
    fname = os.path.join(PLOTDIR, "target.pdf")
    plt.savefig(fname=fname)
    plt.close()


    print("end at ", datetime.datetime.now())
    total_time = datetime.timedelta(seconds=(datetime.datetime.now()-start).total_seconds())
    vals["total time"] = total_time
    save_dfs()
    print("total time:", total_time)
    logf.close()
    errf.close()
    resf.close()

    return df


if __name__ == "__main__":

    class ConfigAction(argparse.Action):

        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            if nargs is not None:
                raise ValueError("nargs is not allowed")
            super().__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string):
            with open(values) as fconf:  # open the conf file to read  and copy the parameters
                data = json.load(fconf)
                for key,val in data.items():
                    setattr(namespace, key, val)
            setattr(namespace, self.dest, values)




    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default=DEFAULTS["output"])
    parser.add_argument("-n", "--name", default=DEFAULTS["name"])
    parser.add_argument("-vn", "--vary-name", default=DEFAULTS["vary-name"])
    parser.add_argument("-lr", "--learning-rate",  type=float, default=DEFAULTS["learning-rate"], help="the learning rate for optimization")
    parser.add_argument("-d", "--depth",  type=int, default = DEFAULTS["depth"],  help="")
    parser.add_argument("-w", "--width",  type=int, default = DEFAULTS["width"], help="")
    parser.add_argument("-x", "--xdim",  type=int, default = DEFAULTS["xdim"], help="")
    parser.add_argument("-z", "--zdim",  type=int, default = DEFAULTS["zdim"], help="")
    grp_iter = parser.add_mutually_exclusive_group(required=False)
    grp_iter.add_argument("-N", "--num-iter",  type=int, help="force number of iterations")
    # grp_iter.add_argument("-M", "--max-iter", type=int,  help="maximum number of iterations, use the convergence test to stop training")
    # grp_iter.add_argument("-M", "--max-time", type=int,  help="maximum number of iterations, use the convergence test to stop training")
    grp_iter.add_argument("-T", "--time",  type=float,  default=DEFAULTS["time"], help="absolute time of training")

    grp_iter_mode = parser.add_mutually_exclusive_group(required=False)
    grp_iter_mode.add_argument("--cvg-test", action="store_true", help="use a convergence test to stop the training")
    grp_iter_mode.add_argument("--no-cvg-test", dest="cvg_test", action="store_false", help="only use the number of iterations to stop the training")

    parser.add_argument("-i", "--init",   choices=utils.INITIALIZATIONS.keys(), default=DEFAULTS["init"], help="the initalization function")
    parser.add_argument("-is", "--init-scheme",  default=DEFAULTS["init-scheme"], choices=("balance", "ortho", "balance-force", None), help="the initialization scheme")
    parser.add_argument("-gma", "--gamma",   type=float, default=DEFAULTS["gamma"], help="Gamma value for initialization of the weights")
    parser.add_argument("--std",   type=float, help="standard deviation for the weights")
    parser.add_argument("-t", "--tau",  type=float, default=DEFAULTS["tau"],  help="")
    parser.add_argument("-s", "--smin", type=float,   default=DEFAULTS["smin"], help="negative value for no effect, >= 0 for setting minimum singular value of the target")
    # parser.add_argument("-st", "--vmax", type=float,   help="")

    parser.add_argument("--rank-target",    type=int, help="")
    parser.add_argument("-gc", "--grad-comp", choices=("manual", "backprop A", "backprop B"), default=DEFAULTS["grad-comp"])
    parser.add_argument("-gr", "--grad-rank", choices=("full", "adapt"), default=DEFAULTS["grad-rank"])
    parser.add_argument("-l", "--loss", choices=("BW", "Fro"), default=DEFAULTS["loss"])

    parser.add_argument("--dataset-fname")
    grp_newds = parser.add_mutually_exclusive_group(required=False)
    grp_newds.add_argument("--new-dataset", action="store_true")
    grp_newds.add_argument("--load-dataset", nargs="?", type=str, default=DEFAULTS["load-dataset"], const="")

    parser.add_argument("-sm", "--samp-mode", choices=("eig", "sing"), default=DEFAULTS["samp-mode"], help="sample either singular values or eigenvalues of the target")
    parser.add_argument("-sd", "--samp-distrib", choices=("random", "zipf"), default=DEFAULTS["samp-distrib"], help="the way the sampling for the target is performed")

    grp_saveds = parser.add_mutually_exclusive_group(required=False)
    grp_saveds.add_argument("--save-dataset", action="store_true")
    grp_saveds.add_argument("--no-save-dataset", dest="save_dataset", action="store_false")

    parser.add_argument("--save-every", type=float, default=DEFAULTS["save-every"], help="the number of iterations to save after, or frequency when between 0 and 1")

    grp_debug = parser.add_mutually_exclusive_group(required=False)
    grp_debug.add_argument("--debug", action="store_true")
    grp_debug.add_argument("--no-debug", dest="debug", action="store_false")



    gp_dvc = parser.add_mutually_exclusive_group(required=False)
    gp_dvc.add_argument("--cpu", action="store_true", help="force use cpu")
    gp_dvc.add_argument("--cuda", dest="cpu", action="store_false", help="use cuda if available")

    parser.add_argument('-igpu', "--gpu-index", type=int, help="gpu index to use")

    parser.add_argument("--config", type=str, action=ConfigAction, help="a previous config file to run again")

    parser.set_defaults(save_dataset=True, new_dataset=False, debug=False, cpu=False, cvg_test=DEFAULTS["cvg-test"])

    args = parser.parse_args()

    train(args)
