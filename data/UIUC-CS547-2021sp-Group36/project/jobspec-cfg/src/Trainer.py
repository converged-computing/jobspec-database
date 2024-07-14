import time
import os, signal

import torch
import torch.nn
import torch.optim

import wandb

import data.ImageLoader as ImageLoader
import LossFunction
import models

class Trainer(object):
    def __init__(self, model,
            dataloader:ImageLoader.TripletSamplingDataLoader,
            validation_set:ImageLoader.TripletSamplingDataLoader,
            g=1.0,
            verbose=True,
            lr=0.0001,
            weight_decay=0.0001,
            device=None):
        """
        """
        self.model = model
        self.dataloader = dataloader
        self.validation_set = validation_set
        self.g = g
        self.loss_fn = LossFunction.LossFunction(self.g)
        self.accuracy_function = LossFunction.TripletAccuracy()
        
        self.device = device
        
        #FREEZING (search other files.)
        #This should really be done automatically in the optimizer. Not thrilled with this.
        #only optimize parameters that we want to optimize
        optim_params = [p for p in self.model.parameters() if p.requires_grad]
        
        #Optimization
        self.optimizer = torch.optim.Adam(optim_params, lr=lr, weight_decay=weight_decay) #TODO: not hardcoded
        self.lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        
        self.total_epochs = 0
        
        #Various intervals
        self.lr_interval = 200
        
        #Logging
        self.verbose = verbose #log to terminal too?
        self.batch_log_interval = 1
        self.checkpoint_interval = 50 #additional checkpoints in number of batches
        self.monitor_norms = False
        
        #For clean interruption
        self.running = False
        self.previous_handler = None
    
    def handle_sigint(self, signum, frame):
        self.running = False
    
    def create_checkpoint(self):
        if self.verbose:
            print("Creating checkpoint")
        model_file = os.path.join(wandb.run.dir, "model_state.pt")
        trainer_file = os.path.join(wandb.run.dir,"trainer.pt")
        
        torch.save(self.model.state_dict(), model_file)
        wandb.save(model_file)
        
        #This will save the entire dataset and everything.
        #We need to implement a state_dict kind of thing for the trainer.
        #torch.save(self, trainer_file)
        #wandb.save(trainer_file)
    
    _ = """
    @classmethod
    def load_checkpoint(cls,base_dir=None):
        raise Exception("Currently unavailable")
        if None == base_dir:
            base_dir = wandb.run.dir
        
        trainer_file = os.path.join(base_dir,"trainer.pt")
        unpickled_trainer = torch.load(model_file)
        return unpickled_trainer
    """
    def crossval(self):
        self.model.eval()
        total_validation_loss = 0.0
        total_seen = 0
        with torch.no_grad():
            for batch_idx, ((Qs,Ps,Ns),l) in enumerate(self.validation_set):
                
                if self.device is not None:
                    Qs = Qs.to(self.device)
                    Ps = Ps.to(self.device)
                    Ns = Ns.to(self.device)
                
                Q_emb = self.model(Qs).detach()
                P_emb = self.model(Ps).detach()
                N_emb = self.model(Ns).detach()
                
                total_validation_loss += float(self.accuracy_function(Q_emb, P_emb, N_emb))
                total_seen += int(len(l))
        
        total_validation_loss /= float(total_seen)
        total_validation_loss = 1.0 - total_validation_loss
        
        if self.verbose:
            print("Crossval_error {}".format(total_validation_loss))
        wandb.log({"epoch_val_error":total_validation_loss},step=wandb.run.step)
        
        return total_validation_loss
    
    def norm_logging(self, q_emb,p_emb,n_emb):
        
        with torch.no_grad():
            mqn = float(torch.norm(q_emb.detach(),dim=1).mean())
            mpn = float(torch.norm(p_emb.detach(),dim=1).mean())
            mnn = float(torch.norm(n_emb.detach(),dim=1).mean())
        overall_mean_norms = float((mqn + mpn + mnn)/3.0)
        
        if self.verbose:
            print("mean Q,P,N norms {:.5f} {:.5f} {:.5f} ".format(mqn, mpn, mnn))
        wandb.log({"embedding_mean_l2_norm":overall_mean_norms},commit=False,step=wandb.run.step)
        
        return overal_mean_norms
    
    def log_lr(self):
        lr = 0.0
        n_seen = 0
        for param_group in self.optimizer.param_groups:
            lr += param_group["lr"]
            n_seen += 1
        
        lr = lr/n_seen
        if self.verbose:
            print("LR : {}".format(lr))
        wandb.log({"current_lr":lr},commit=False,step=wandb.run.step)

    
    def train_one_batch(self, one_batch,batch_idx=None):
        ((Qs,Ps,Ns),l) = one_batch
        
        if self.device is not None:
            Qs = Qs.to(self.device)
            Ps = Ps.to(self.device)
            Ns = Ns.to(self.device)
        
        Q_embedding_vectors = self.model(Qs)
        P_embedding_vectors = self.model(Ps)
        N_embedding_vectors = self.model(Ns)
        
        batch_loss = self.loss_fn(Q_embedding_vectors, P_embedding_vectors, N_embedding_vectors)
        batch_loss.backward()
        
        batch_loss = float(batch_loss)
        
        ## DEBUG: Monitor Norms
        if self.monitor_norms:
            overall_mean_norms = norm_logging(Q_embedding_vectors, P_embedding_vectors, N_embedding_vectors)
        
        return batch_loss
    
    def train(self, n_epochs):
        self.running = True
        #install signal handler for clean interruption
        self.previous_handler = signal.signal(signal.SIGINT, self.handle_sigint)
        
        for _ in range(n_epochs):
            #clean stoppign on signal
            if not self.running:
                break
            self.total_epochs += 1
            
            self.log_lr()
            
            epoch_average_batch_loss = 0.0;
            batchgroup_average_batch_loss = 0.0;
            
            for batch_idx, one_batch in enumerate(self.dataloader):
                
                #clean stopping on signal
                if not self.running:
                    break
                
                batch_start_time = time.time() #Throughput measurement
                
                #======ONE TRAINING STEP =========
                self.model.train(True)
                self.optimizer.zero_grad()
                
                batch_loss = self.train_one_batch(one_batch,batch_idx)
                
                self.optimizer.step()
                #=====TRAINING STEP DONE, SOME LOGGIN AND THINGS======
                
                #cumulative
                epoch_average_batch_loss += float(batch_loss)
                batchgroup_average_batch_loss += float(batch_loss)
                
                #creates a step
                wandb.log({"batch_loss":float(batch_loss)},commit=False)
                #DEBUG LOG loss to terminal #TODO: Can wandb echo to terminal?
                if self.verbose and 0 == batch_idx % self.batch_log_interval:
                    print("batch ({}) loss {:.5f}".format(batch_idx,
                                                            float(batch_loss))
                        )
                #TODO: Add proper logging
                
                
                #CHECKPOINTING (epochs are so long that we need to checkpoitn more freqently)
                if 0 != batch_idx and 0 == batch_idx%self.checkpoint_interval:
                    self.create_checkpoint()
                
                #LEARNING SCHEDULE
                if 0 != batch_idx and self.lr_interval > 0 and 0 == batch_idx%self.lr_interval:
                    batchgroup_average_batch_loss /= self.lr_interval
                    self.lr_schedule.step(batchgroup_average_batch_loss)
                    batchgroup_average_batch_loss = 0.0
                    
                    #Any logging of LR rate
                    self.log_lr()
                
                #TODO: Any per-batch logging
                #END of loop over batches
                batch_end_time = time.time() #Throughput measurement
                batch_time_per_item = float(batch_end_time-batch_start_time)/len(one_batch) #Throughput measurement
                #Commit wandb logs for this batch
                if self.verbose:
                    print("time per item: {}".format(batch_time_per_item))
                wandb.log({"time_per_item":batch_time_per_item},commit=True, step=wandb.run.step)
                
                
            self.model.train(False)
            
            #Until now, this was actually total batch loss
            epoch_average_batch_loss /= batch_idx
            wandb.log({"epoch_average_batch_loss":epoch_average_batch_loss
                        },step=wandb.run.step)
            #TODO: log LR for this epoch.
            
            #TODO: any logging
            #TODO: any validation checking, any learning_schedule stuff.
            if self.running and 0 == self.total_epochs % self.lr_interval:
                self.model.eval()
                
                #TODO: blah. Too slow.
                #self.lr_schedule.step(epoch_average_batch_loss)
            
            #CROSSVALIDATION
            if self.running and None != self.validation_set:
                self.crossval()
                        
                        
            
            #Also save a checkpoint after every epoch and upon cancellation
            self.create_checkpoint()
        
        #END of train
        #restore the previous interrupt handler
        signal.signal(signal.SIGINT, self.previous_handler)
        
        return self.running #should_continue

if __name__ == "__main__":
    import os
    import torch
    import torchvision
    
    import warnings
    warnings.warning("Training entrypoint has moved to train_main.py")
    
    #testing
    run_id = wandb.util.generate_id()
    #TODO: Move to a main script and a bash script outside this program.
    wandb_tags = ["debug"]
    wandb.init(id=run_id,
                resume="allow",
                entity='uiuc-cs547-2021sp-group36',
                project='image_similarity',
                group="debugging",
                tags=wandb_tags)
    if wandb.run.resumed:
        print("Resuming...")
    
    print("create model")
    model = models.create_model("LowDNewModel")
    if wandb.run.resumed:
        print("Resuming from checkpoint")
        model_pickle_file = wandb.restore("model_state.pt")
        model.load_state_dict( torch.load(model_pickle_file.name) )
    wandb.watch(model, log_freq=100) #Won't work if we restore the full object from pickle. :(
    
    
    print("load data")
    all_train = ImageLoader.load_imagefolder("/workspace/datasets/tiny-imagenet-200/")
    train_data, crossval_data, _ = ImageLoader.split_imagefolder(all_train, [0.5,0.1,0.1])
    print("create dataloader")
    tsdl = ImageLoader.TripletSamplingDataLoader(train_data,batch_size=100, num_workers=0)
    tsdl_crossval = ImageLoader.TripletSamplingDataLoader(crossval_data,batch_size=100, num_workers=0,shuffle=False)
    
    print("create trainer")
    test_trainer = Trainer(model, tsdl, tsdl_crossval)
    test_trainer.loss_fn = LossFunction.create_loss("normed")
    
    print("Begin training")
    test_trainer.train(100)
