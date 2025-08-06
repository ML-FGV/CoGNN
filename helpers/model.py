from enum import Enum, auto
from torch.nn import Module
from typing import List, Callable

from models.layers import WeightedGCNConv, WeightedGINConv, WeightedGNNConv, GraphLinear, GPS
from models.mpnns_model import parse_method
from models.drgnn import DRGNN

class ModelType(Enum):
    """
        an object for the different core
    """
    GCN = auto()
    GIN = auto()
    LIN = auto()

    SUM_GNN = auto()
    MEAN_GNN = auto()

    GPS = auto()

    MPNN = auto()

    DRGNN = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return ModelType[s]
        except KeyError:
            raise ValueError()

    def load_component_cls(self):
        if self is ModelType.GCN:
            return WeightedGCNConv
        elif self is ModelType.GIN:
            return WeightedGINConv
        elif self in [ModelType.SUM_GNN, ModelType.MEAN_GNN]:
            return WeightedGNNConv
        elif self is ModelType.LIN:
            return GraphLinear
        elif self is ModelType.GPS:
            return GPS
        elif self is ModelType.MPNN:
            return parse_method
        elif self is ModelType.DRGNN:
            return DRGNN
        else:
            raise ValueError(f'model {self.name} not supported')

    def is_gcn(self):
        return self is ModelType.GCN

    def get_component_list(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, bias: bool,
                           edges_required: bool, gin_mlp_func: Callable, args) -> List[Module]:
        dim_list = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        if self is ModelType.GCN:
            component_list = [self.load_component_cls()(in_channels=in_dim_i, out_channels=out_dim_i, bias=bias)
                              for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        elif self is ModelType.GIN:
            component_list = [self.load_component_cls()(in_channels=in_dim_i, out_channels=out_dim_i, bias=bias,
                                                        mlp_func=gin_mlp_func)
                              for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        elif self in [ModelType.SUM_GNN, ModelType.MEAN_GNN]:
            aggr = 'mean' if self is ModelType.MEAN_GNN else 'sum'
            component_list = [self.load_component_cls()(in_channels=in_dim_i, out_channels=out_dim_i, aggr=aggr,
                                                        bias=bias)
                              for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        elif self is ModelType.LIN:
            assert not edges_required, f'env does not support {self.name}'
            component_list = \
                [self.load_component_cls()(in_features=in_dim_i, out_features=out_dim_i, bias=bias)
                 for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        elif self is ModelType.GPS:
            component_list = [self.load_component_cls()(in_channels=in_dim_i, out_channels=out_dim_i, bias=bias)
                              for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        elif self is ModelType.MPNN:
            component_list = [self.load_component_cls()(args, None, out_dim_i, in_dim_i, device=args.device)
                                for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        elif self is ModelType.DRGNN:
            component_list = [self.load_component_cls()(in_channels=in_dim_i, hidden_channels=out_dim_i,
                                                        out_channels=out_dim_i, dropout=args.dropout,
                                                        phantom_grad=args.phantom_grad, beta_init=args.beta_init,
                                                        gamma_init=args.gamma_init, tol=args.tol)
                              for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        else:
            raise ValueError(f'model {self.name} not supported')
        return component_list
