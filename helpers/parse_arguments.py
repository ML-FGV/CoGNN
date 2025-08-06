from argparse import ArgumentParser

from helpers.dataset_classes.dataset import DataSet
from helpers.model import ModelType
from helpers.classes import Pool
from helpers.encoders import PosEncoder


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", default=DataSet.roman_empire, type=DataSet.from_string,
                        choices=list(DataSet), required=False)
    parser.add_argument("--pool", dest="pool", default=Pool.NONE, type=Pool.from_string,
                        choices=list(Pool), required=False)

    # gumbel
    parser.add_argument("--learn_temp", dest="learn_temp", default=False, action='store_true', required=False)
    parser.add_argument("--temp_model_type", dest="temp_model_type", default=ModelType.LIN,
                        type=ModelType.from_string, choices=list(ModelType), required=False)
    parser.add_argument("--tau0", dest="tau0", default=0.5, type=float, required=False)
    parser.add_argument("--temp", dest="temp", default=0.01, type=float, required=False)

    # optimization
    parser.add_argument("--max_epochs", dest="max_epochs", default=3000, type=int, required=False)
    parser.add_argument("--batch_size", dest="batch_size", default=32, type=int, required=False)
    parser.add_argument("--lr", dest="lr", default=1e-3, type=float, required=False)
    parser.add_argument("--dropout", dest="dropout", default=0.2, type=float, required=False)

    # env cls parameters
    parser.add_argument("--env_model_type", dest="env_model_type", default=ModelType.MEAN_GNN,
                        type=ModelType.from_string, choices=list(ModelType), required=False)
    parser.add_argument("--env_num_layers", dest="env_num_layers", default=3, type=int, required=False)
    parser.add_argument("--env_dim", dest="env_dim", default=128, type=int, required=False)
    parser.add_argument("--skip", dest="skip", default=False, action='store_true', required=False)
    parser.add_argument("--batch_norm", dest="batch_norm", default=False, action='store_true', required=False)
    parser.add_argument("--layer_norm", dest="layer_norm", default=False, action='store_true', required=False)
    parser.add_argument("--dec_num_layers", dest="dec_num_layers", default=1, type=int, required=False)
    parser.add_argument("--pos_enc", dest="pos_enc", default=PosEncoder.NONE,
                        type=PosEncoder.from_string, choices=list(PosEncoder), required=False)

    # policy cls parameters
    parser.add_argument("--act_model_type", dest="act_model_type", default=ModelType.MEAN_GNN,
                        type=ModelType.from_string, choices=list(ModelType), required=False)
    parser.add_argument("--act_num_layers", dest="act_num_layers", default=1, type=int, required=False)
    parser.add_argument("--act_dim", dest="act_dim", default=16, type=int, required=False)

    # reproduce
    parser.add_argument("--seed", dest="seed", type=int, default=0, required=False)
    parser.add_argument('--gpu', dest="gpu", type=int, required=False)

    # dataset dependant parameters 
    parser.add_argument("--fold", dest="fold", default=None, type=int, required=False)

    # optimizer and scheduler
    parser.add_argument("--weight_decay", dest="weight_decay", default=0, type=float, required=False)
    ## for steplr scheduler only
    parser.add_argument("--step_size", dest="step_size", default=None, type=int, required=False)
    parser.add_argument("--gamma", dest="gamma", default=None, type=float, required=False)
    ## for cosine with warmup scheduler only
    parser.add_argument("--num_warmup_epochs", dest="num_warmup_epochs", default=None, type=int, required=False)

    ## args from tunedGNN    
    #parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    #parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')
    parser.add_argument('--model', type=str, default='MPNN')
    # GNN
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7)
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--pre_ln', action='store_true')
    parser.add_argument('--pre_linear', action='store_true')
    parser.add_argument('--res', action='store_true', help='use residual connections for GNNs')
    parser.add_argument('--ln', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--bn', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--jk', action='store_true', help='use JK for GNNs')
    
    # training
    #parser.add_argument('--lr', type=float, default=0.001)
    #parser.add_argument('--weight_decay', type=float, default=5e-4)
    #parser.add_argument('--dropout', type=float, default=0.5)
    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')

    return parser.parse_args()
