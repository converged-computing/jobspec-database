import torch
import models.losses as losses
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
from utils.fid_scores import fid_pytorch
#from utils.miou_scores import miou_pytorch
from utils.Metrics import metrics

import config


#--- read options ---#
opt = config.read_arguments(train=True)
print("number of gpus: ", torch.cuda.device_count())
#--- create utils ---#
timer = utils.timer(opt)
visualizer_losses = utils.losses_saver(opt)
losses_computer = losses.losses_computer(opt)
dataloader,dataloader_supervised, dataloader_val = dataloaders.get_dataloaders(opt)
print('data successfully loaded')
im_saver = utils.image_saver(opt)
fid_computer = fid_pytorch(opt, dataloader_val)
metrics_computer = metrics(opt, dataloader_val)
#miou_computer = miou_pytorch(opt,dataloader_val)

#--- create models ---#
model = models.Unpaired_model(opt)
model = models.put_on_multi_gpus(model, opt)
utils.load_networks(opt, model)
#--- create optimizers ---#
optimizerG = torch.optim.Adam(model.module.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
optimizerD = torch.optim.Adam(model.module.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
optimizerDu = torch.optim.Adam(model.module.netDu.parameters(), lr=5*opt.lr_d, betas=(opt.beta1, opt.beta2))
# optimizerDe = torch.optim.Adam(model.module.wavelet_decoder.parameters(), lr=5*opt.lr_d, betas=(opt.beta1, opt.beta2))
# optimizerDe2 = torch.optim.Adam(model.module.wavelet_decoder2.parameters(), lr=5*opt.lr_d, betas=(opt.beta1, opt.beta2))



def loopy_iter(dataset):
    while True :
        for item in dataset :
            yield item

#--- the training loop ---#
already_started = True
start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))
if opt.model_supervision != 0 :
    supervised_iter = loopy_iter(dataloader_supervised)
for epoch in range(start_epoch, opt.num_epochs):
    for i, data_i in enumerate(dataloader):
        if not already_started and i < start_iter:
            continue
        already_started = True
        cur_iter = epoch*len(dataloader) + i
        image, label = models.preprocess_input(opt, data_i)  ####
        # [bs, 3, 256, 256] [bs, 38, 256, 256]
        #--- generator unconditional update ---#
        model.module.netG.zero_grad()
        loss_G, losses_G_list = model(image, label, "losses_G", losses_computer)
        loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
        loss_G.backward()
        optimizerG.step()

        # --- generator conditional update ---#
        if opt.model_supervision != 0 :
            supervised_data = next(supervised_iter)
            p_image, p_label = models.preprocess_input(opt,supervised_data)
            model.module.netG.zero_grad()
            p_loss_G, p_losses_G_list = model(image, label, "losses_G_supervised", losses_computer)
            p_loss_G, p_losses_G_list = p_loss_G.mean(), [loss.mean() if loss is not None else None for loss in p_losses_G_list]
            p_loss_G.backward()
            optimizerG.step()
        else:
            p_loss_G, p_losses_G_list = torch.zeros((1)), [torch.zeros((1))]


        #--- discriminator update ---#
        model.module.netD.zero_grad()
        loss_D, losses_D_list = model(image, label, "losses_D", losses_computer)
        loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
        loss_D.backward()
        optimizerD.step()

        #--- unconditional discriminator update ---#
        model.module.netDu.zero_grad()
        # model.module.wavelet_decoder.zero_grad()
        # model.module.wavelet_decoder2.zero_grad()
        loss_Du, losses_Du_list = model(image, label, "losses_Du", losses_computer)
        loss_Du, losses_Du_list = opt.reg_every*loss_Du.mean(), [loss.mean() if loss is not None else None for loss in losses_Du_list]
        loss_Du.backward()
        optimizerDu.step()
        # optimizerDe.step()
        # optimizerDe2.step()

        #--- generator psuedo labels updates ---@



        # --- unconditional discriminator regulaize ---##
        if i % opt.reg_every == 0:
            model.module.netDu.zero_grad()
            loss_reg, losses_reg_list = model(image, label, "Du_regulaize", losses_computer)
            loss_reg, losses_reg_list = loss_reg.mean(), [loss.mean() if loss is not None else None for loss in losses_reg_list]
            loss_reg.backward()
            optimizerDu.step()
        else :
            loss_reg, losses_reg_list = torch.zeros((1)), [torch.zeros((1))]

        #--- stats update ---#
        if not opt.no_EMA:
            utils.update_EMA(model, cur_iter, dataloader, opt)
        if cur_iter %opt.freq_print == 0:
            im_saver.visualize_batch(model, image, label, cur_iter)
            timer(epoch, cur_iter)
        if cur_iter % opt.freq_save_ckpt == 0:
            utils.save_networks(opt, cur_iter, model)
        if cur_iter % opt.freq_save_latest == 0:
            utils.save_networks(opt, cur_iter, model, latest=True)
        if cur_iter % opt.freq_fid == 0 and cur_iter > 0:
            is_best = fid_computer.update(model, cur_iter)
            metrics_computer.update_metrics(model, cur_iter)
            if is_best:
                utils.save_networks(opt, cur_iter, model, best=True)
        #_ = miou_computer.update(model,cur_iter)   #  do not have miou model yet
        visualizer_losses(cur_iter, losses_G_list+p_losses_G_list+losses_D_list+losses_Du_list+losses_reg_list)


#--- after training ---#
utils.update_EMA(model, cur_iter, dataloader, opt, force_run_stats=True)
utils.save_networks(opt, cur_iter, model)
utils.save_networks(opt, cur_iter, model, latest=True)
is_best = fid_computer.update(model, cur_iter)
metrics_computer.update_metrics(model, cur_iter)

#if is_best:
    #utils.save_networks(opt, cur_iter, model, best=True)

print("The training has successfully finished")


