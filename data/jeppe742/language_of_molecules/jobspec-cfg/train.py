from torch.optim import Adam,lr_scheduler
import torch
import argparse
from utils.dataloader import QM9Dataset, DataLoader
from layers.transformer import TransformerModel
from layers.bagofwords import BagOfWordsModel, SimpleBagOfWordsModel, BagOfWordsType
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--num_heads', default=3, type=int)
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--num_epochs',  default=100, type=int)
parser.add_argument('--batch_size', default=248, type=int)
parser.add_argument('--edge_encoding', default=1, type=int)
parser.add_argument('--epsilon_greedy', default=0.2, type=float)
parser.add_argument('--num_masks', default=1, type=int)
parser.add_argument('--num_fake', default=0, type=int)
parser.add_argument('--num_same', default=0, type=int)
parser.add_argument('--name_postfix', default='', type=str)
parser.add_argument('--use_cuda', default=True, type=bool)
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--scaffold', default=True, type=bool)
parser.add_argument('--model',choices=['BoN','BoA','Transformer','SimpleBoN', 'SimpleBoA'], default='Transformer')
parser.add_argument('--gamma',default=1, type=float)
parser.add_argument('--bond_order', default=False, type=bool)
parser.add_argument('--dataset', default='zinc', choices=['qm9','zinc'])
args = parser.parse_args()


train_file = f'data/{args.dataset}/adjacency_matrix_train_scaffold.pkl' if args.scaffold else f'data/{args.dataset}/adjacency_matrix_train.pkl'
validation_file = f'data/{args.dataset}/adjacency_matrix_validation_scaffold.pkl' if args.scaffold else f'data/{args.dataset}/adjacency_matrix_validation.pkl'

training = QM9Dataset(data=train_file,
                      num_masks=args.num_masks,
                      epsilon_greedy=args.epsilon_greedy,
                      num_fake=args.num_fake,
                      bond_order=args.bond_order)

train_dl = DataLoader(
    training,
    batch_size=args.batch_size)


# Create multiple validation dlators, one for 25, 50 and 75% masked atoms
val_dls = []
if args.num_fake == 0:
    for masks in range(1, 6):

        val_set = QM9Dataset(data=validation_file, num_masks=masks,bond_order=args.bond_order)
        val_dl = DataLoader(
            val_set,
            batch_size=args.batch_size)
        val_dls.append(val_dl)

if args.num_masks == 0:
    for fakes in range(1, 6):

        val_set = QM9Dataset(data=validation_file, num_fake=fakes,bond_order=args.bond_order)
        val_dl = DataLoader(
            val_set,
            batch_size=args.batch_size)
        val_dls.append(val_dl)


if args.model =='Transformer':
        
    model = TransformerModel(num_layers=args.num_layers,
                                        num_heads=args.num_heads,
                                        embedding_dim=args.embedding_dim,
                                        num_classes=5 if args.dataset=='qm9' else 10,
                                        dropout=args.dropout,
                                        edge_encoding=args.edge_encoding,
                                        use_cuda=args.use_cuda,
                                        name=(
                                            "Transformer2"
                                            f"_num_masks={args.num_masks}"
                                            f"_num_fake={args.num_fake}"
                                            f"_num_same={args.num_same}"
                                            f"_num_layers={args.num_layers}"
                                            f"_num_heads={args.num_heads}"
                                            f"_embedding_dim={args.embedding_dim}"
                                            f"_dropout={args.dropout}"
                                            f"_lr={args.lr}"
                                            f"_edge_encoding={args.edge_encoding}"
                                            f"_epsilon_greedy={args.epsilon_greedy}"
                                            f"_gamma={args.gamma}"
                                            f"_bond_order={args.bond_order}"
                                            f"_dataset={args.dataset}"
                                            f"{args.name_postfix}"
                                        )
                                    )
elif args.model == 'BoA':
    model = BagOfWordsModel(num_layers=args.num_layers,
                                        embedding_dim=args.embedding_dim,
                                        BagOfWordsType=BagOfWordsType.ATOMS,
                                        num_classes=5 if args.dataset=='qm9' else 10,
                                        use_cuda=args.use_cuda,
                                        name=(
                                            "BagOfWords"
                                            f"_num_masks={args.num_masks}"
                                            f"_num_fake={args.num_fake}"
                                            f"_num_same={args.num_same}"
                                            f"_num_layers={args.num_layers}"
                                            f"_embedding_dim={args.embedding_dim}"
                                            f"_lr={args.lr}"
                                            f"_epsilon_greedy={args.epsilon_greedy}"
                                            f"_bow_type={BagOfWordsType.ATOMS}"
                                            f"_dataset={args.dataset}"
                                            f"{args.name_postfix}"
                                        )
                        )
elif args.model == 'BoN':
    model = BagOfWordsModel(num_layers=args.num_layers,
                                        embedding_dim=args.embedding_dim,
                                        BagOfWordsType=BagOfWordsType.NEIGHBOURS,
                                        num_classes=5 if args.dataset=='qm9' else 10,
                                        use_cuda=args.use_cuda,
                                        name=(
                                            "BagOfWords"
                                            f"_num_masks={args.num_masks}"
                                            f"_num_fake={args.num_fake}"
                                            f"_num_same={args.num_same}"
                                            f"_num_layers={args.num_layers}"
                                            f"_embedding_dim={args.embedding_dim}"
                                            f"_lr={args.lr}"
                                            f"_epsilon_greedy={args.epsilon_greedy}"
                                            f"_bow_type={BagOfWordsType.NEIGHBOURS}"
                                            f"_dataset={args.dataset}"
                                            f"{args.name_postfix}"
                                        )
                        )

elif args.model == 'SimpleBoN':
    model = SimpleBagOfWordsModel(num_layers=args.num_layers,
                                        embedding_dim=args.embedding_dim,
                                        BagOfWordsType=BagOfWordsType.NEIGHBOURS,
                                        num_classes=5 if args.dataset=='qm9' else 10,
                                        use_cuda=args.use_cuda,
                                        name=(
                                            "SimpleBagOfNeighbours2"
                                            f"_num_masks={args.num_masks}"
                                            f"_num_fake={args.num_fake}"
                                            f"_num_same={args.num_same}"
                                            f"_num_layers={args.num_layers}"
                                            f"_embedding_dim={args.embedding_dim}"
                                            f"_lr={args.lr}"
                                            f"_epsilon_greedy={args.epsilon_greedy}"
                                            f"_bow_type={BagOfWordsType.NEIGHBOURS}"
                                            f"_dataset={args.dataset}"
                                            f"{args.name_postfix}"
                                        )
                        )

elif args.model == 'SimpleBoA':
    model = SimpleBagOfWordsModel(num_layers=args.num_layers,
                                        embedding_dim=args.embedding_dim,
                                        BagOfWordsType=BagOfWordsType.ATOMS,
                                        num_classes=5 if args.dataset=='qm9' else 10,
                                        use_cuda=args.use_cuda,
                                        name=(
                                            "SimpleBagOfAtoms2"
                                            f"_num_masks={args.num_masks}"
                                            f"_num_fake={args.num_fake}"
                                            f"_num_same={args.num_same}"
                                            f"_num_layers={args.num_layers}"
                                            f"_embedding_dim={args.embedding_dim}"
                                            f"_lr={args.lr}"
                                            f"_epsilon_greedy={args.epsilon_greedy}"
                                            f"_bow_type={BagOfWordsType.ATOMS}"
                                            f"_dataset={args.dataset}"
                                            f"{args.name_postfix}"
                                        )
                        )

def optimizer_fun(param): return Adam(param, lr=args.lr)


if not args.debug:
    wandb.init(project="language-of-molecules-graph", name=model.name)
    wandb.config.update(args)
    wandb.watch(model)

model.train_network(train_dl, val_dls, num_epochs=args.num_epochs, 
                               eval_after_epochs=1,
                               log_after_epochs=1, 
                               optimizer_fun=optimizer_fun, 
                               save_model=True,
                               scheduler_fun=lambda optimizer:lr_scheduler.ExponentialLR(optimizer, args.gamma))
