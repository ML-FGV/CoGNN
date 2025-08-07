from argparse import ArgumentParser

from helpers.dataset_classes.dataset import DataSet
from helpers.model import ModelType
from helpers.classes import Pool
from helpers.encoders import PosEncoder

from distutils.util import strtobool

def str2bool(x):
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')

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

    # DRGNN
    parser.add_argument('--phantom_grad', type=int, default=0)
    parser.add_argument('--beta_init', type=float, default=0.0)
    parser.add_argument('--gamma_init', type=float, default=1.0)
    parser.add_argument('--tol', type=float, default=1e-5)

    # NSD
    #parser.add_argument('--epochs', type=int, default=1500)
    #parser.add_argument('--lr', type=float, default=0.01)
    #parser.add_argument('--weight_decay', type=float, default=0.0005)
    #parser.add_argument('--sheaf_decay', type=float, default=None)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--min_acc', type=float, default=0.0,
                        help="Minimum test acc on the first fold to continue training.")
    parser.add_argument('--stop_strategy', type=str, choices=['loss', 'acc'], default='loss')

    # Model configuration
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--normalised', dest='normalised', type=str2bool, default=True,
                        help="Use a normalised Laplacian")
    parser.add_argument('--deg_normalised', dest='deg_normalised', type=str2bool, default=False,
                        help="Use a a degree-normalised Laplacian")
    parser.add_argument('--linear', dest='linear', type=str2bool, default=False,
                        help="Whether to learn a new Laplacian at each step.")
    #parser.add_argument('--hidden_channels', type=int, default=20)
    parser.add_argument('--input_dropout', type=float, default=0.0)
    #parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--left_weights', dest='left_weights', type=str2bool, default=True,
                        help="Applies left linear layer")
    parser.add_argument('--right_weights', dest='right_weights', type=str2bool, default=True,
                        help="Applies right linear layer")
    parser.add_argument('--add_lp', dest='add_lp', type=str2bool, default=False,
                        help="Adds fixed high pass filter in the restriction maps")
    parser.add_argument('--add_hp', dest='add_hp', type=str2bool, default=False,
                        help="Adds fixed low pass filter in the restriction maps")
    parser.add_argument('--use_act', dest='use_act', type=str2bool, default=True)
    parser.add_argument('--second_linear', dest='second_linear', type=str2bool, default=False)
    parser.add_argument('--orth', type=str, choices=['matrix_exp', 'cayley', 'householder', 'euler'],
                        default='householder', help="Parametrisation to use for the orthogonal group.")
    parser.add_argument('--sheaf_act', type=str, default="tanh", help="Activation to use in sheaf learner.")
    parser.add_argument('--edge_weights', dest='edge_weights', type=str2bool, default=True,
                        help="Learn edge weights for connection Laplacian")
    parser.add_argument('--sparse_learner', dest='sparse_learner', type=str2bool, default=False)

    # Experiment parameters
    #parser.add_argument('--dataset', default='texas')
    #parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--folds', type=int, default=10)
    #parser.add_argument('--model', type=str, choices=['DiagSheaf', 'BundleSheaf', 'GeneralSheaf', 'DiagSheafODE',
                                                     # 'BundleSheafODE', 'GeneralSheafODE'], default=None)
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--evectors', type=int, default=0, help="Number of Laplacian PE eigenvectors to use.")

    # ODE args
    parser.add_argument('--max_t', type=float, default=1.0, help="Maximum integration time.")
    parser.add_argument('--int_method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
    #parser.add_argument('--step_size', type=float, default=1,
     #                   help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                        help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument("--max_nfe", type=int, default=1000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--no_early", action="store_true",
                        help="Whether or not to use early stopping of the ODE integrator when testing.")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
    parser.add_argument("--max_test_steps", type=int, default=100,
                        help="Maximum number steps for the dopri5Early test integrator. "
                             "used if getting OOM errors at test time")

    return parser.parse_args()
