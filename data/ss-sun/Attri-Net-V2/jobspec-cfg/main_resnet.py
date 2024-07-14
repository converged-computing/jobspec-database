from solvers.resnet_solver import resnet_solver
# from solvers.attrinet_solver import task_switch_solver
# from solvers.attrinet_solver_g_withoutcl import task_switch_solver
from solvers.attrinet_solver_energyloss import task_switch_solver
# from solvers.attrinet_solver_energyloss_with_psydoMask import task_switch_solver
from solvers.bcosnet_solver import bcos_resnet_solver
import logging
from experiment_utils import init_seed, init_experiment, init_wandb
from train_utils import prepare_datamodule
import argparse

def str2bool(v):
    return v.lower() in ('true')




def resnet_get_parser():
    parser = argparse.ArgumentParser()

    # Experiment configuration.
    parser.add_argument('--exp_name', type=str, default='resnet')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--img_mode', type=str, default='gray', choices=['color', 'gray']) # will change to color if dataset is airogs_color

    # Data configuration.
    parser.add_argument('--dataset', type=str, default='chexpert', choices=['chexpert', 'nih_chestxray', 'vindr_cxr'])
    parser.add_argument('--image_size', type=int, default=320, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')

    # Training configuration.
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay')
    parser.add_argument('--manual_seed', type=int, default=42, help='set seed')
    parser.add_argument('--save_path', type=str, default='/mnt/qb/work/baumgartner/sun22/TMI_exps/resnet', help='path of the exp')
    # Testing configuration.
    parser.add_argument('--test_model_path', type=str, default='/mnt/qb/work/baumgartner/sun22/TMI_exps/resnet', help='path of the models')

    # Miscellaneous.
    parser.add_argument('--use_wandb', type=str2bool, default=True)
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to run on the GPU')

    return parser







def prepare_exps(exp_configs):
    if exp_configs.mode == 'train':
        print("training model: ")
        init_experiment(exp_configs)
    init_seed(exp_configs.manual_seed)
    if exp_configs.use_wandb:
        init_wandb(exp_configs)


def main(exp_configs):
    from data.dataset_params import dataset_dict, data_default_params
    prepare_exps(exp_configs)
    print("experiment folder: " + exp_configs.exp_dir)
    datamodule = prepare_datamodule(exp_configs, dataset_dict, data_default_params)
    print(exp_configs)
    # Prepare data loaders and solver.
    data_loader = {}
    data_loader["train"] = datamodule.train_dataloader(batch_size=exp_configs.batch_size, shuffle=True)
    data_loader["valid"] = datamodule.valid_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
    data_loader["test"] = datamodule.test_dataloader(batch_size=exp_configs.batch_size, shuffle=False)
    solver = resnet_solver(exp_configs, data_loader=data_loader)

    if exp_configs.mode == "train":
        print('start training...')
        solver.train()
        print('finish training!')

    if exp_configs.mode == 'test':
        print('start testing....')
        solver.load_model(exp_configs.test_model_path)
        test_auc = solver.test()
        print('finish test!')
        print('test_auc: ', test_auc)





if __name__ == '__main__':

    parser = resnet_get_parser()
    config = parser.parse_args()
    if config.dataset == 'vindr_cxr':
        assert config.epochs > 20

    main(config)

    # datasets = ['chexpert', 'nih_chestxray', 'vindr_cxr', 'skmtea', 'airogs', 'airogs_color', 'vindr_cxr_withBB', 'contam20', 'contam50']
    # config.dataset = datasets[config.dataset_idx]
    # if 'color' in config.dataset:
    #     config.img_mode = 'color'

