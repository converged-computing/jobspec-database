import argparse
import importlib
import sys
import numpy as np
import time
import copy

from bunch import Bunch
from sklearn.decomposition import NMF,PCA
from sklearn.cluster import KMeans

from data.data_loader import DataLoader
from utils.utils import set_seed, sample_target_items,stack_csrdata, load_fake_data
from framework import DQN_attack,init_model
from adversary_example_generator_v2 import generate_target_users     

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="generate_attack_args")
parser.add_argument("-dataset", type=str)
parser.add_argument("-att_type", type=str)
parser.add_argument("-ratio", type=float)
parser.add_argument("-pop",type=str)
parser.add_argument("-unroll",type=int)
parser.add_argument("-tag",type=str)
parser.add_argument("-alpha",type=float,default=1.0)
parser.add_argument("-transfer",type=int,default=0)
parser.add_argument("-cluster",type=int,default=0)
#parser.add_argument("-version",type=int,default=1)
config = parser.parse_args()

def get_median(results):
    sorted_result=results
    sorted_result.sort()
    if len(results)%2==1:
        return sorted_result[len(results)//2]
    else:
        return (sorted_result[len(results)//2]+sorted_result[len(results)//2-1])/2

def get_target_users(model,training_dataset,user_amounts,item_amounts,target_item,cluster):
    current_model=model
    recommendation_of_normal_users=model.recommend(training_dataset[:user_amounts],item_amounts)
    user_item_ranks=np.zeros_like(recommendation_of_normal_users)
    for i in range(user_amounts):
        for j in range(item_amounts):
            user_item_ranks[i,recommendation_of_normal_users[i,j]]=j
    rank_of_items=[np.where(row==target_item)[0] for row in recommendation_of_normal_users]
    rank_of_items=np.asarray(rank_of_items)
    rank_of_items=np.reshape(rank_of_items,(1,len(rank_of_items)))[0]
    
    if cluster==0:
        target_item_ranks=[]
        for i in range(len(rank_of_items)):
            target_item_ranks.append((i,rank_of_items[i]))
        target_item_ranks=sorted(target_item_ranks,key=lambda x:x[1])
        real_target_users=target_item_ranks[-user_amounts//3:]
        remain_users=target_item_ranks[:-user_amounts//3]
        real_target_users_id=np.asarray([ele[0] for ele in real_target_users])
        real_target_users_rank=np.asarray([ele[1] for ele in real_target_users])
        remain_users_id=np.asarray([ele[0] for ele in remain_users])
        remain_users_ranks=np.asarray([ele[1] for ele in remain_users])
        sample_users_id=np.random.choice(remain_users_id,size=user_amounts//3,replace=False)
    else:
        cluster_cnt=10
        all_user_records=training_dataset.toarray()
        all_user_ids=np.arange(user_amounts)
        pca=PCA(n_components=100)
        compressed_remain_users=pca.fit_transform(all_user_records)
        kmeans = KMeans(n_clusters=cluster_cnt, random_state=0)
        kmeans.fit(compressed_remain_users)
        cluster_labels = kmeans.labels_
        cluster_avg=[]      
        for i in range(cluster_cnt):
            cluster_data=rank_of_items[cluster_labels==i]
            if len(cluster_data)>=5:
                avg=np.mean(cluster_data)
                cluster_avg.append((i,avg))
        cluster_avg=np.array(cluster_avg)
        
        if cluster==1:
            cluster_id=cluster_avg[np.argmax(cluster_avg[:,1])][0]
        if cluster==-1:
            cluster_id=cluster_avg[np.argmin(cluster_avg[:,1])][0]
        real_target_users_id=all_user_ids[cluster_labels==cluster_id] 
        sample_users_id=np.setdiff1d(all_user_ids,real_target_users_id)

    target_users_rank=np.mean(user_item_ranks[real_target_users_id],axis=0)
    sample_users_rank=np.mean(user_item_ranks[sample_users_id],axis=0)

    real_trigger_item_id=np.argmin(target_users_rank-sample_users_rank)
   
    return real_target_users_id,real_trigger_item_id

def generate(config,revise,trigger_item,n_users,n_items,train_data,test_data,attack_gen_args):

    # Load data.
    #print(config)
    print("dataset={}".format(args.data_path))

    print("attack method={}".format(args.attack_gen_args))
    print("Loading data from {}".format(args.data_path))
    print("n_users: {}, n_items: {}".format(n_users, n_items))
    print("surrogate model={}".format(attack_gen_args.surrogate))



    print("n_users: {}, n_items: {}".format(n_users, n_items))
    print("popularity={}, fake user ratio={}, fake users={}, unroll epochs={}".format(attack_gen_args.target_item_popularity,attack_gen_args.n_fakes,attack_gen_args.n_fake_users, attack_gen_args.unroll_steps))    
    attack_gen_args.trigger_item=None
    if config.att_type=="RandFilter":
        attack_gen_args.click_target=True
    else:
        attack_gen_args.click_target=False
    if config.att_type=="DQN":
        DQN_attack(n_users=n_users,n_items=n_items,train_data=train_data,test_data=test_data,args=attack_gen_args)
    else:
        if revise>0:
            attack_gen_args.trigger_item=np.zeros_like(1)
            attack_gen_args.trigger_item=trigger_item
        adv_trainer_class = attack_gen_args.trainer_class
        adv_trainer = adv_trainer_class(n_users=n_users,n_items=n_items,args=attack_gen_args,attack_ver=revise)
        generated_fake_data=adv_trainer.fit(train_data, test_data)
    return generated_fake_data

def evaluate(args,config,revise,target_items,trigger_item,fake_data,target_users,train_data,test_data):

    attack_eval_args = Bunch(args.attack_eval_args)
    #attack_eval_args.target_items_path=attack_eval_args.target_items_path+popularity+"_"+config.tag+".npz"

    n_fakes = 0

    #if attack_eval_args.fake_data_path:
    if fake_data!=None:
        #fake_data = load_fake_data(attack_eval_args.fake_data_path)
        train_data = stack_csrdata(train_data, fake_data)
        n_fakes = fake_data.shape[0]        
        #print("Statistics of fake data: n_fakes={}, avg_clicks={:.2f}".format(n_fakes, fake_data.sum(1).mean()))
        

    # Evaluate victim model performance.
    # evaluate output HR20
    for victim_args in attack_eval_args.victims:
        victim_args = Bunch(victim_args)
        victim_args.tag=config.tag
        trainer_class = victim_args.model["trainer_class"]
        trainer = trainer_class(n_users=n_users+n_fakes,
                                n_items=n_items,
                                args=victim_args)
        trainer.fit(train_data, test_data)
        # Load target items and evaluate attack performance.
        #target_items = np.load(attack_eval_args.target_items_path)['target_items']
        result=trainer.validate_target_users(train_data, test_data, target_items,target_users)
        target_item_HR=result["TargetHR@20"]
        trigger_items=np.asarray([trigger_item])
        result=trainer.validate_target_users(train_data, test_data, trigger_items,target_users)   
        trigger_item_HR=result["TargetHR@20"]
    return target_item_HR,trigger_item_HR    
    
if __name__ == "__main__":
    args = importlib.import_module(config.config_file)
    args.seed=int(time.time())
    set_seed(args.seed, args.use_cuda)
    args.data_path="./data/"+config.dataset
    if config.att_type=="WRMF_ALS":
        args.attack_gen_args=args.attack_gen_args_wrmf_als 
    elif config.att_type=="WRMF_SGD":
        args.attack_gen_args=args.attack_gen_args_wrmf_sgd
    elif config.att_type=="RandFilter":
        args.attack_gen_args=args.attack_gen_args_randfilter
    elif config.att_type=="DQN":
        args.attack_gen_args=args.attack_gen_args_dqn
    else:
        raise Error("Attack method error.")

    attack_gen_args = Bunch(args.attack_gen_args)
    if config.pop=="head":
        attack_gen_args.target_item_popularity="head"
    elif config.pop=="upper":
        attack_gen_args.target_item_popularity="upper_torso"
    elif config.pop=="lower":
        attack_gen_args.target_item_popularity="lower_torso"
    elif config.pop=="tail":
        attack_gen_args.target_item_popularity="tail"
    else:
        raise Error("Popularity type error.")

    attack_gen_args.unroll_steps=config.unroll
    attack_gen_args.tag=config.tag
    if config.alpha>=0:
        attack_gen_args.alpha=config.alpha
    if config.alpha<0:
        attack_gen_args.alpha=-1.0/config.alpha
    print(attack_gen_args.alpha)
    
    data_loader = DataLoader(path=args.data_path)
    n_users, n_items = data_loader.n_users, data_loader.n_items
    attack_gen_args.n_fakes=config.ratio/100
    attack_gen_args.n_fake_users = int(n_users * attack_gen_args.n_fakes)
    org_train_data = data_loader.load_train_data()
    org_test_data = data_loader.load_test_data()
    print(n_users,n_items,attack_gen_args.n_fake_users)
    
    train_data_first_half=org_train_data[1::2,:]
    train_data_second_half=org_train_data[::2,:]
    test_data_first_half=org_test_data[1::2,:]
    test_data_second_half=org_test_data[::2,:]
    if config.transfer==0:
        train_data=train_data_first_half
        test_data=test_data_first_half
        n_users=train_data_first_half.get_shape()[0]
    else:
        train_data=org_train_data
        test_data=org_test_data

    args.seed=int(time.time())
    set_seed(args.seed, args.use_cuda)
    target_items = sample_target_items(train_data,n_samples=attack_gen_args.n_target_items,popularity=attack_gen_args.target_item_popularity,use_fix=attack_gen_args.use_fixed_target_item,output_dir=attack_gen_args.output_dir,tag=attack_gen_args.tag)
    attack_gen_args.target_items = target_items
    model=init_model(attack_gen_args,n_users,n_items,train_data,test_data)

    targetHR_results_revisit=[]
    triggerHR_results_revisit=[]
    targetHR_results_trigger=[]
    triggerHR_results_trigger=[]
    targetHR_results_clean=[]
    triggerHR_results_clean=[]

    for i in range(10):
        revised_target_users,trigger_item=get_target_users(model,train_data,n_users,n_items,target_items[0],config.cluster)
        attack_gen_args.target_users=revised_target_users
        print("target item={},trigger item={}".format(target_items,trigger_item))
        #if config.version==-1:
        fake_user_data=None
        config.config_file="evaluate_attack_args"
        evaluate_args=importlib.import_module(config.config_file)
        targetHR,triggerHR=evaluate(evaluate_args,config,-1,attack_gen_args.target_items,trigger_item,fake_user_data,attack_gen_args.target_users,train_data,test_data)
        targetHR_results_clean.append(targetHR*100)
        triggerHR_results_clean.append(triggerHR*100)
        print("------------clean {} targetHR={:.4f} triggerHR={:.4f}-------------".format(i+1,targetHR*100,triggerHR*100))
        #if config.version>=0:
        config.config_file="generate_attack_args"
        fake_user_data=generate(config,0,trigger_item,n_users,n_items,train_data,test_data,attack_gen_args)
        config.config_file="evaluate_attack_args"
        evaluate_args=importlib.import_module(config.config_file)
        targetHR,triggerHR=evaluate(evaluate_args,config,0,attack_gen_args.target_items,trigger_item,fake_user_data,attack_gen_args.target_users,train_data,test_data)
        targetHR_results_revisit.append(targetHR*100)
        triggerHR_results_revisit.append(triggerHR*100)
        print("------------revisit {} targetHR={:.4f} triggerHR={:.4f}-------------".format(i+1,targetHR*100,triggerHR*100))
        config.config_file="generate_attack_args"
        fake_user_data=generate(config,1,trigger_item,n_users,n_items,train_data,test_data,attack_gen_args)
        config.config_file="evaluate_attack_args"
        evaluate_args=importlib.import_module(config.config_file)
        targetHR,triggerHR=evaluate(evaluate_args,config,1,attack_gen_args.target_items,trigger_item,fake_user_data,attack_gen_args.target_users,train_data,test_data)
        targetHR_results_trigger.append(targetHR*100)
        triggerHR_results_trigger.append(triggerHR*100)
        print("------------trigger {} targetHR={:.4f} triggerHR={:.4f}-------------".format(i+1,targetHR*100,triggerHR*100))
        
    print(targetHR_results_clean,targetHR_results_revisit,targetHR_results_trigger)
    print(triggerHR_results_clean,triggerHR_results_revisit,triggerHR_results_trigger)
    print("clean median results: target item HR={:.4f},trigger item HR={:.4f}".format(get_median(targetHR_results_clean),get_median(triggerHR_results_clean)))
    avg_target=sum(targetHR_results_clean)/len(targetHR_results_clean)
    avg_trigger=sum(triggerHR_results_clean)/len(triggerHR_results_clean)
    print("clean average results: target item HR={:.4f},trigger item HR={:.4f}".format(avg_target,avg_trigger))
    print("revisit median results: target item HR={:.4f},trigger item HR={:.4f}".format(get_median(targetHR_results_revisit),get_median(triggerHR_results_revisit)))
    avg_target=sum(targetHR_results_revisit)/len(targetHR_results_revisit)
    avg_trigger=sum(triggerHR_results_revisit)/len(triggerHR_results_revisit)
    print("revisit average results: target item HR={:.4f},trigger item HR={:.4f}".format(avg_target,avg_trigger))
    print("trigger median results: target item HR={:.4f},trigger item HR={:.4f}".format(get_median(targetHR_results_trigger),get_median(triggerHR_results_trigger)))
    avg_target=sum(targetHR_results_trigger)/len(targetHR_results_trigger)
    avg_trigger=sum(triggerHR_results_trigger)/len(triggerHR_results_trigger)
    print("trigger average results: target item HR={:.4f},trigger item HR={:.4f}".format(avg_target,avg_trigger))