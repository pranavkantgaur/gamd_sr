'''
1. How to gather evidence for correlation between aggregate edge messages and the corresponding ground truth forces?
    1. Steps:
        1. Load trained GNN model and its class implementation
        2. Load MD files using dataloader
        3. For each MD file:
            1. Save aggregate edge messages from 4rth MPBlock into a dictionary
            2. Save corresponding GT force into that dictionary
        4. Save the resulting dictionary: dict[’filename’][’node_id’][’aggregate_message’]
        5. Load dictionary from the file:
            1. Across all MD files:
                1. Aggregate edge messages are 128 numbers for each node whereas the corresponding force GT is 3 numbers per node → Select top-3 components from aggregate edge message with most variance
                2. For each component:
                    1. fit that component as a function of the corresponding force components  → if the fit is good then *aggregate edge messages are infact linearly related with the corresponding GT forces*
                    2. Plot the fit across all MD files: (just for visualization)
                        1. For each particle, plot that componet on y-axis and plot the value of linear combination on x-axis
                        2. Expected to be a linear if step-1 resulted in a good fit.
2. How to gather evidence for correlation between inter-particle force and the corresponding edge message? 
    1. Seems like can’t be done without heuristics or real simulation/experimental data on inter-particle forces. 
3. How to fit SR over the correlation between particle-pair features and the corresponding edge message?
    1. Steps:
        1. Create a dictionary for recording SR input features across input MD files
        2. For each MD file:
            1. In the 4rth MPBlock, record <node1 posx, posy, posz>, <node2 posx, posy, posz>, <radial distance> as inputs and the corresponding edge message as expected outputs for SR
        3. Save the dictionary: dict[’filename’][’edge_id’][’inputs’][’pos1’/’pos2’/’rad_dist’], dict[’filename’][’edge_id’][’output’][’edge_message’]
        4. Load the dictionary
        5. Initialize pySR and configure its search space:
            1. Operators
            2. Metric
        6. Run pySR over inputs and outputs from the loaded dictionary
        7. Print predicted equations
4. How to relate predicted equation with LJ interparticle force equation?
    1. **TBD later**
'''

import numpy as np
import matplotlib.pyplot as  plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monolithic_gamd_impl import MDDataset, custom_collate, evaluate
import dgl
import dgl.function as fn
from typing import List, Set, Dict, Tuple, Optional
import random
import time
from sklearn.preprocessing import StandardScaler
import matplotlib.tri as mtri
import pandas as pd
from pysr import PySRRegressor
import pickle

# check for CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is: ", device)




class MPBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(MPBlock, self).__init__()
        # Set the seed
        seed = 10

        # PyTorch seed
        torch.manual_seed(seed)

        self.phi = nn.Sequential(nn.SiLU(),                               
            nn.Linear(embed_dim,  hidden_dim),                      
            nn.SiLU(),                               
            nn.Linear(hidden_dim, embed_dim)) # [e, embed_dim]
        
        # Set the seed
        seed = 10

        # PyTorch seed
        torch.manual_seed(seed)
        self.theta = nn.Sequential(nn.SiLU(),                               
            nn.Linear(embed_dim,  embed_dim)) 
        # Set the seed
        seed = 10

        # PyTorch seed
        torch.manual_seed(seed)                                 
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim) # normalize each node embedding across its embed_dim values.
        #self.initialize_weights()   

    def initialize_weights(self):
        # Initialize weights for phi
        print("INFO: CALLED")
        for layer in self.phi:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Apply Xavier initialization
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero

        # Initialize weights for theta
        for layer in self.theta:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Apply Xavier initialization
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero
    


    def forward(self, node_embeddings, edge_embeddings, edge_index_list): 
        # [b * num_nodes, embed_dim], [e, embed_dim], [e, embed_dim], [e, 2]         
        #mpblock_start_time = time.time()
        node_embeddings = self.layer_norm(node_embeddings) # [b * num_nodes, embed_dim]     
        center_node_index = edge_index_list[0, :] # [1, e] 
        neighbor_node_index = edge_index_list[1, :] # [1, e]
        center_node_embeddings = node_embeddings[center_node_index]
        neighbor_node_embeddings = node_embeddings[neighbor_node_index]
        sum_tensor = edge_embeddings + center_node_embeddings + neighbor_node_embeddings
        theta_edge = self.phi(sum_tensor)        

        aggregated_edge_messages_torch_vectorized = torch.zeros((node_embeddings.shape[0], theta_edge.shape[1]), dtype=torch.float32).cuda()

        
        edge_message_neigh_center = (neighbor_node_embeddings * theta_edge)#.to(torch.float16)  # Convert to float16        
        aggregated_edge_messages_torch_vectorized.index_add_(0, center_node_index, edge_message_neigh_center)  # Accumulate messages into AGG
        

        node_embeddings = self.theta(node_embeddings + aggregated_edge_messages_torch_vectorized)          
        
        self.edge_message_neigh_center = edge_message_neigh_center # for symbolic regression
        
        return node_embeddings


class MPNN(nn.Module):
    def __init__(self, num_mpnn_layers, embed_dim, hidden_dim):
        super(MPNN, self).__init__()
        self.num_mpnn_layers = num_mpnn_layers
        self.mp_blocks = nn.Sequential(*[MPBlock(embed_dim, hidden_dim) for _ in range(num_mpnn_layers)]) # list of length num_mpnn_layers of MPBlock objects.
        

    def forward(self, node_embeddings, edge_embeddings, edge_index_list): # [b * num_nodes, embed_dim], [e, embed_dim], [e, 2]                
        for layer in self.mp_blocks:
          node_embeddings_initial = node_embeddings.clone()
          node_embeddings = layer(node_embeddings, edge_embeddings, edge_index_list) + node_embeddings_initial  # residual connection
        return node_embeddings # [#num nodes, embed_dim], [#num edges, embed_dim]     


class GAMDNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_mpnn_layers, num_mlp_layers, num_atom_type_classes, num_edge_types, num_rbfs):
        super(GAMDNet, self).__init__()
        self.num_mpnn_layers = num_mpnn_layers
        self.num_mlp_layers = num_mlp_layers
        self.embed_dim = embed_dim
        # Set the seed
        seed = 10

        # PyTorch seed
        torch.manual_seed(seed)           
        self.edge_embedding_mlp = nn.Sequential(nn.Linear(3, hidden_dim), # [#e, num_rbfs + 3 + 1] 
            nn.GELU(),                               
            nn.Linear(hidden_dim, hidden_dim),                      
            nn.GELU(),                               
            nn.Linear(hidden_dim, embed_dim)  # [#e, embed_dim]               
        )
        self.edge_layer_norm = nn.LayerNorm(self.embed_dim)
        self.force_pred_mlp = nn.Sequential(nn.Linear(embed_dim, hidden_dim), 
            nn.GELU(),                                                                       
            nn.Linear(hidden_dim, 3)                 
        )        
      
        self.mpnn_mlps = MPNN(num_mpnn_layers, embed_dim, hidden_dim) 

        # Set the seed
        seed = 10

        # PyTorch seed
        torch.manual_seed(seed)           
        self.node_embeddings = nn.Parameter(torch.randn((1, self.embed_dim)), requires_grad=True)
        #self.initialize_weights()
        

    def initialize_weights(self):
        # Initialize weights for phi
        print("INFO: CALLED FROM GAMDNET")
        for layer in self.edge_embedding_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Apply Xavier initialization
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero

        # Initialize weights for theta
        for layer in self.force_pred_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Apply Xavier initialization
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero    

    def _get_interatomic_distance_vector(self, pos, edge_index_list): # [b * 258, 3], [e, 2]
        '''
        Compute edge directional features for all input edges.
        '''        
        node_i_coord = pos[(edge_index_list[0, :]).to(device)] # [#e, 3], 
        node_j_coord = pos[(edge_index_list[1, :]).to(device)] # [#e, 3]
        relative_pos = node_j_coord - node_i_coord
        return relative_pos

    def _get_edge_features(self, pos, edge_index_list): # [b * 258, 1], [e, 2]
        '''
        Returns features for all input edges as a list
        '''                
        interatomic_distance_vectors = self._get_interatomic_distance_vector(pos, edge_index_list) 
        edge_features = interatomic_distance_vectors
        return edge_features 
        
    def forward(self, pos, edge_index_list): # [b * 258, 3], [e, 2], [b * 258, 1]        
        edge_features = self._get_edge_features(pos, edge_index_list) # no weights, [#e, num_rbfs + num_edge_types]   

        edge_embeddings = self.edge_embedding_mlp(edge_features.float()) # weights, mlp, [#e, num_rbfs + num_edge_types] -> [#e, embed_dim]

        edge_embeddings = self.edge_layer_norm(edge_embeddings)

        num_nodes = pos.shape[0]                
        node_embeddings = self.node_embeddings.repeat((num_nodes, 1)).to(device)

        node_embeddings = self.mpnn_mlps(node_embeddings, edge_embeddings, edge_index_list) # L blocks of weights (L sequential MLPs),         
        node_forces = self.force_pred_mlp(node_embeddings) # weights, mlp, [b * num_nodes, embed_dim] -> [b * num_nodes, 3]
        
        return node_forces





## OFFICIAL gamd implementation
class MLP(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim=128,
                 hidden_layer=3,
                 activation_first=False,
                 activation='relu',
                 init_param=False):
        super(MLP, self).__init__()
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        else:
            raise Exception('Only support: relu, leaky_relu, sigmoid, tanh, elu, as non-linear activation')

        mlp_layer = []
        for l in range(hidden_layer):
            if l != hidden_layer-1 and l != 0:
                mlp_layer += [nn.Linear(hidden_dim, hidden_dim), act_fn]
            elif l == 0:
                if hidden_layer == 1:
                    if activation_first:
                        mlp_layer += [act_fn, nn.Linear(in_feats, out_feats)]
                    else:
                        print('Using MLP with no hidden layer and activations! Fall back to nn.Linear()')
                        mlp_layer += [nn.Linear(in_feats, out_feats)]
                elif not activation_first:
                    mlp_layer += [nn.Linear(in_feats, hidden_dim), act_fn]
                else:
                    mlp_layer += [act_fn, nn.Linear(in_feats, hidden_dim), act_fn]
            else:   # l == hidden_layer-1
                mlp_layer += [nn.Linear(hidden_dim, out_feats)]
        self.mlp_layer = nn.Sequential(*mlp_layer)
        if init_param:
            self._init_parameters()

    def _init_parameters(self):
        for layer in self.mlp_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, feat):
        return self.mlp_layer(feat.to('cuda:0'))


class SmoothConvLayerNew(nn.Module):
    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 hidden_dim=128,
                 activation='relu',
                 drop_edge=True,
                 update_edge_emb=False):

        super(SmoothConvLayerNew, self).__init__()
        self.drop_edge = drop_edge
        self.update_edge_emb = update_edge_emb
        if self.update_edge_emb:
            self.edge_layer_norm = nn.LayerNorm(in_edge_feats)

        # self.theta_src = nn.Linear(in_node_feats, hidden_dim)
        self.edge_affine = MLP(in_edge_feats, hidden_dim, activation=activation, hidden_layer=2)
        self.src_affine = nn.Linear(in_node_feats, hidden_dim)
        self.dst_affine = nn.Linear(in_node_feats, hidden_dim)        
        self.theta_edge = MLP(hidden_dim, in_node_feats,
                              hidden_dim=hidden_dim, activation=activation, activation_first=True,
                              hidden_layer=2)
        # self.theta = MLP(hidden_dim, hidden_dim, activation_first=True, hidden_layer=2)

        self.phi_dst = nn.Linear(in_node_feats, hidden_dim)
        self.phi_edge = nn.Linear(in_node_feats, hidden_dim)
        self.phi = MLP(hidden_dim, out_node_feats,
                       activation_first=True, hidden_layer=1, hidden_dim=hidden_dim, activation=activation)

    def forward(self, g: dgl.DGLGraph, node_feat: torch.Tensor) -> torch.Tensor:
        h = node_feat.clone()
        with g.local_scope():
            if self.drop_edge and self.training:
                src_idx, dst_idx = g.edges()
                e_feat = g.edata['e'].clone()
                dropout_ratio = 0.2
                idx = np.arange(dst_idx.shape[0])
                np.random.shuffle(idx)
                keep_idx = idx[:-int(idx.shape[0] * dropout_ratio)]
                src_idx = src_idx[keep_idx]
                dst_idx = dst_idx[keep_idx]
                e_feat = e_feat[keep_idx]
                g = dgl.graph((src_idx, dst_idx))
                g.edata['e'] = e_feat
            # for multi batch training
            if g.is_block:
                h_src = h
                h_dst = h[:g.number_of_dst_nodes()]
            else:
                h_src = h_dst = h

            g.srcdata['h'] = h_src
            g.dstdata['h'] = h_dst
            edge_idx = g.edges()
            src_idx = edge_idx[0]
            dst_idx = edge_idx[1]
            edge_code = self.edge_affine(g.edata['e'])            
            src_code = self.src_affine(h_src[src_idx])
            dst_code = self.dst_affine(h_dst[dst_idx])
            g.edata['e_emb'] = self.theta_edge(edge_code+src_code+dst_code)
            
            self.edge_message_neigh_center = src_code * g.edata['e_emb'] # for storing edge messages, SR           
            
            '''
            edge_msg_stds = torch.std(self.edge_message_neigh_center, dim=0)            
            print("Edge message components with highest stds: ", torch.argsort(edge_msg_stds)[-3:])
            e1 = self.edge_message_neigh_center[:50, torch.argsort(edge_msg_stds)[-3]]
            e2 = self.edge_message_neigh_center[:50, torch.argsort(edge_msg_stds)[-2]]
            e3 = self.edge_message_neigh_center[:50, torch.argsort(edge_msg_stds)[-1]]
            print("GNN: Edge message 250: ", e1, e2, e3)
            '''
            if self.update_edge_emb:
                normalized_e_emb = self.edge_layer_norm(g.edata['e_emb'])
            g.update_all(fn.u_mul_e('h', 'e_emb', 'm'), fn.sum('m', 'h'))
            edge_emb = g.ndata['h']
            self.aggregate_edge_messages = edge_emb # Add aggregate edge message recording, SR. 
            self.input_node_embeddings = h # For SR
        if self.update_edge_emb:
            g.edata['e'] = normalized_e_emb
        node_feat = self.phi(self.phi_dst(h) + self.phi_edge(edge_emb)) # So, we need to predict edge_emb (aggrgated edge messages), h (input node embeddings) inputs for the last message passing layer using SR to effecitively couple SR with the remaining GNN.  
        self.node_embeddings = node_feat # For SR
        return node_feat


class SmoothConvBlockNew(nn.Module):
    def __init__(self,
                 in_node_feats,
                 out_node_feats,
                 hidden_dim=128,
                 conv_layer=3,
                 edge_emb_dim=64,
                 use_layer_norm=False,
                 use_batch_norm=True,
                 drop_edge=False,
                 activation='relu',
                 update_egde_emb=False,
                 ):
        super(SmoothConvBlockNew, self).__init__()
        self.conv = nn.ModuleList()
        self.edge_emb_dim = edge_emb_dim
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm

        self.drop_edge = drop_edge
        if use_batch_norm == use_layer_norm and use_batch_norm:
            raise Exception('Only one type of normalization at a time')
        if use_layer_norm or use_batch_norm:
            self.norm_layers = nn.ModuleList()

        for layer in range(conv_layer):
            if layer == 0:
                self.conv.append(SmoothConvLayerNew(in_node_feats=in_node_feats,
                                                 in_edge_feats=self.edge_emb_dim,
                                                 out_node_feats=out_node_feats,
                                                 hidden_dim=hidden_dim,
                                                 activation=activation,
                                                 drop_edge=drop_edge,
                                                 update_edge_emb=update_egde_emb))
            else:
                self.conv.append(SmoothConvLayerNew(in_node_feats=out_node_feats,
                                                 in_edge_feats=self.edge_emb_dim,
                                                 out_node_feats=out_node_feats,
                                                 hidden_dim=hidden_dim,
                                                 activation=activation,
                                                 drop_edge=drop_edge,
                                                 update_edge_emb=update_egde_emb))
            if use_layer_norm:
                self.norm_layers.append(nn.LayerNorm(out_node_feats))
            elif use_batch_norm:
                self.norm_layers.append(nn.BatchNorm1d(out_node_feats))

    def forward(self, h: torch.Tensor, graph: dgl.DGLGraph) -> torch.Tensor:

        for l, conv_layer in enumerate(self.conv):
            if self.use_layer_norm or self.use_batch_norm:
                h = conv_layer.forward(graph, self.norm_layers[l](h)) + h
            else:
                h = conv_layer.forward(graph, h) + h

        return h


# code from DGL documents
class RBFExpansion(nn.Module):
    r"""Expand distances between nodes by radial basis functions.

    .. math::
        \exp(- \gamma * ||d - \mu||^2)

    where :math:`d` is the distance between two nodes and :math:`\mu` helps centralizes
    the distances. We use multiple centers evenly distributed in the range of
    :math:`[\text{low}, \text{high}]` with the difference between two adjacent centers
    being :math:`gap`.

    The number of centers is decided by :math:`(\text{high} - \text{low}) / \text{gap}`.
    Choosing fewer centers corresponds to reducing the resolution of the filter.

    Parameters
    ----------
    low : float
        Smallest center. Default to 0.
    high : float
        Largest center. Default to 30.
    gap : float
        Difference between two adjacent centers. :math:`\gamma` will be computed as the
        reciprocal of gap. Default to 0.1.
    """
    def __init__(self, low=0., high=30., gap=0.1):
        super(RBFExpansion, self).__init__()

        num_centers = int(np.ceil((high - low) / gap))
        self.centers = np.linspace(low, high, num_centers)
        self.centers = nn.Parameter(torch.tensor(self.centers).float(), requires_grad=False)
        self.gamma = 1 / gap

    def reset_parameters(self):
        """Reinitialize model parameters."""
        device = self.centers.device
        self.centers = nn.Parameter(
            self.centers.clone().detach().float(), requires_grad=False).to(device)

    def forward(self, edge_dists):
        """Expand distances.

        Parameters
        ----------
        edge_dists : float32 tensor of shape (E, 1)
            Distances between end nodes of edges, E for the number of edges.

        Returns
        -------
        float32 tensor of shape (E, len(self.centers))
            Expanded distances.
        """
        radial = edge_dists - self.centers
        coef = - self.gamma
        return torch.exp(coef * (radial ** 2))



class SimpleMDNetNew(nn.Module):  # no bond, no learnable node encoder
    def __init__(self,
                 encoding_size,
                 out_feats,
                 box_size,   # can also be array
                 hidden_dim=128,
                 conv_layer=4,
                 edge_embedding_dim=128,
                 dropout=0.1,
                 drop_edge=True,
                 use_layer_norm=False):
        super(SimpleMDNetNew, self).__init__()
        self.graph_conv = SmoothConvBlockNew(in_node_feats=encoding_size,
                                             out_node_feats=encoding_size,
                                             hidden_dim=hidden_dim,
                                             conv_layer=conv_layer,
                                             edge_emb_dim=edge_embedding_dim,
                                             use_layer_norm=use_layer_norm,
                                             use_batch_norm=not use_layer_norm,
                                             drop_edge=drop_edge,
                                             activation='silu')

        self.edge_emb_dim = edge_embedding_dim
        self.edge_expand = RBFExpansion(high=1, gap=0.025)
        self.edge_drop_out = nn.Dropout(dropout)

        self.length_mean = nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.length_std = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.length_scaler = StandardScaler()

        if isinstance(box_size, np.ndarray):
            self.box_size = torch.from_numpy(box_size).float()
        else:
            self.box_size = box_size
        self.box_size = self.box_size

        self.node_emb = nn.Parameter(torch.randn((1, encoding_size)), requires_grad=True)
        self.edge_encoder = MLP(3 + 1 + len(self.edge_expand.centers), self.edge_emb_dim, hidden_dim=hidden_dim,
                                activation='gelu')
        self.edge_layer_norm = nn.LayerNorm(self.edge_emb_dim)
        self.graph_decoder = MLP(encoding_size, out_feats, hidden_layer=2, hidden_dim=hidden_dim, activation='gelu')

    def calc_edge_feat(self,
                       src_idx: torch.Tensor,
                       dst_idx: torch.Tensor,
                       pos_src: torch.Tensor,
                       pos_dst=None) -> torch.Tensor:
        # this is the raw input feature

        # to enhance computation performance, dont track their calculation on graph
        if pos_dst is None:
            pos_dst = pos_src

        with torch.no_grad():
            rel_pos = pos_dst[dst_idx.long()] - pos_src[src_idx.long()]
            if isinstance(self.box_size, torch.Tensor):
                rel_pos_periodic = torch.remainder(rel_pos + 0.5 * self.box_size.to(rel_pos.device),
                                                   self.box_size.to(rel_pos.device)) - 0.5 * self.box_size.to(rel_pos.device)
            else:
                rel_pos_periodic = torch.remainder(rel_pos + 0.5 * self.box_size,
                                                   self.box_size) - 0.5 * self.box_size

            rel_pos_norm = rel_pos_periodic.norm(dim=1).view(-1, 1)  # [edge_num, 1]
            rel_pos_periodic /= rel_pos_norm + 1e-8   # normalized

        if self.training:
            self.fit_length(rel_pos_norm)
            self._update_length_stat(self.length_scaler.mean_, np.sqrt(self.length_scaler.var_))

        rel_pos_norm = (rel_pos_norm - self.length_mean) / self.length_std
        edge_feat = torch.cat((rel_pos_periodic,
                               rel_pos_norm,
                               self.edge_expand(rel_pos_norm)), dim=1)
        return edge_feat

    def build_graph(self,
                    fluid_edge_idx: torch.Tensor,
                    fluid_pos: torch.Tensor,
                    self_loop=True) -> dgl.DGLGraph:

        center_idx = fluid_edge_idx[0, :]  # [edge_num, 1]
        neigh_idx = fluid_edge_idx[1, :]
        fluid_graph = dgl.graph((neigh_idx, center_idx))
        fluid_edge_feat = self.calc_edge_feat(center_idx, neigh_idx, fluid_pos)

        fluid_edge_emb = self.edge_layer_norm(self.edge_encoder(fluid_edge_feat))  # [edge_num, 64]
        fluid_edge_emb = self.edge_drop_out(fluid_edge_emb)
        fluid_graph.edata['e'] = fluid_edge_emb

        # add self loop for fluid particles
        if self_loop:
            fluid_graph.add_self_loop()
        return fluid_graph

    def build_graph_batches(self, pos_lst, edge_idx_lst):
        graph_lst = []
        for pos, edge_idx in zip(pos_lst, edge_idx_lst):
            graph = self.build_graph(edge_idx, pos)
            graph_lst += [graph]
        batched_graph = dgl.batch(graph_lst)
        return batched_graph

    def _update_length_stat(self, new_mean, new_std):
        self.length_mean[0] = new_mean[0]
        self.length_std[0] = new_std[0]

    def fit_length(self, length):
        if not isinstance(length, np.ndarray):
            length = length.detach().cpu().numpy().reshape(-1, 1)
        self.length_scaler.partial_fit(length)

    def forward(self,
                fluid_pos_lst: List[torch.Tensor],  # list of [N, 3]
                fluid_edge_lst: List[torch.Tensor]
                ) -> torch.Tensor:
        if len(fluid_pos_lst) > 1:
            fluid_graph = self.build_graph_batches(fluid_pos_lst, fluid_edge_lst)
        else:
            fluid_graph = self.build_graph(fluid_edge_lst[0], fluid_pos_lst[0])
        num = np.sum([pos.shape[0] for pos in fluid_pos_lst])
        x = self.node_emb.repeat((num, 1))
        x = self.graph_conv(x, fluid_graph)

        x = self.graph_decoder(x)
        return x





############################################ prepare inputs for testing linearity hypothesis ###############################################
def load_model_and_dataset(gamdnet_model_filename, gamdnet_official_model_checkpoint_filename, md_filedir):
    '''
    Load model and MD dataset for SR from input filename and dataset directory.
    '''

    '''
    embed_dim = 128 
    hidden_dim = 128
    num_mpnn_layers = 4 # as per paper, for LJ system
    num_mlp_layers = 3
    num_atom_type_classes = 1 # Ar atoms only
    num_edge_types = 1 # non-bonded edges only
    num_rbfs = 10 # RBF expansion of interatomic distance vector of each edge to num_rbfs dimensions
    gamdnet = GAMDNet(embed_dim, hidden_dim, num_mpnn_layers, num_mlp_layers, num_atom_type_classes, num_edge_types, num_rbfs).to(device)  
    # Load the weights from 'model.pt'
    # Load the checkpoint from 'model.pt'
    checkpoint = torch.load(gamdnet_model_filename)
    gamdnet.load_state_dict(checkpoint['model_state_dict'])
    # Set the model to evaluation mode
    gamdnet.eval()
    
    print("GAMD model weights loaded successfully.")
    '''
    train_data_fraction = 1.0 # select 9k for training
    avg_num_neighbors = 20 # criteria for connectivity of atoms for any frame
    rotation_aug = False # online rotation augmentation for a frame    
    # create train data-loader
    return_train_data = True
    num_input_files = 1# 40#len(os.listdir(md_filedir))
    batch_size = num_input_files # number of graphs in a batch
    print("Loading input files: ", num_input_files)
    #print("Files are: ", os.listdir(md_filedir))
    dataset = MDDataset(md_filedir, rotation_aug, avg_num_neighbors, train_data_fraction, return_train_data) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    print("Dataloader initialized.")   

    
    print("Loading official GAMDNET...") 
    param_dict = {
                'encoding_size': 128,
                'out_feats': 3,
                'hidden_dim': 128,
                'edge_embedding_dim': 128,
                'conv_layer': 4,
                'drop_edge': False,
                'use_layer_norm': True,
                'box_size': 27.27,
                }
    gamdnet_official = SimpleMDNetNew(**param_dict).to(device)   
    checkpoint = torch.load(gamdnet_official_model_checkpoint_filename, map_location='cuda:0')    
    
    state_dict_original = checkpoint['state_dict']

    # Define the prefix to remove
    prefix_to_remove = 'pnet_model.'

    # Create a new dictionary with updated keys
    state_dict_without_prefix = {
        key[len(prefix_to_remove):]: value 
        for key, value in state_dict_original.items() 
        if key.startswith(prefix_to_remove)
    }
    
    gamdnet_official.load_state_dict(state_dict_without_prefix)
    
    print("GAMD official model weights loaded successfully.")
    
    gamdnet_official.eval()
    
    #return gamdnet, None, dataloader
    return None, gamdnet_official, dataloader


def compute_lj_force_and_potential(pos, edge_index_list):
    
 
    # Find columns where both rows have the same index
    same_index_columns = (edge_index_list[0] == edge_index_list[1])        

    #print("Number of edges before: ", edge_index_list.shape)    
    # Remove these columns from the original index tensor
    #edge_index_list = edge_index_list[:, ~same_index_columns]
    
    #print("Number of edges after: ", edge_index_list.shape)    
    
    #exit(0)
 
    center_node_idx = edge_index_list[0, :]
    neigh_node_idx = edge_index_list[1, :]

    neigh_node_pos = pos[neigh_node_idx]
    center_node_pos = pos[center_node_idx]

    # Calculate the distance vector
    r_vec = neigh_node_pos - center_node_pos  # Shape: [n, 3]
    
    # Calculate the distance (magnitude)
    r = torch.norm(r_vec, dim=1).unsqueeze(1)  # Shape: [n, 1]
    
    epsilon = 0.238
    sigma = 3.4 
    force_magnitude = 48 * epsilon * (
        ((sigma ** 12) / (r ** 13)) - 
        ((sigma ** 6) / (r ** 7))
    )  # Shape: [n, 1]

    # Calculate the force vector (directed)
    force_vector = force_magnitude * (r_vec / r)  # Shape: [n, 3]
    

    potential_magnitude = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    potential_vector = potential_magnitude * (r_vec / r)

    nan_mask = torch.isnan(force_vector).any(dim=1)
    valid_indices = ~nan_mask    

    dx = r_vec[:, 0]
    dy = r_vec[:, 1]
    dz = r_vec[:, 2]
    
    return force_vector, potential_vector, r, valid_indices, dx, dy, dz



def get_msg_force_dict(gamdnet, gamdnet_official, dataloader):
    msg_force_dict = {}
    # run inference over the input batched graph from dataloader
    # record aggregate edge messages and force ground truths for each node in output dictionary
    for pos, edge_index_list, force_gt in dataloader:
        with torch.no_grad():  # Disable gradient calculation for inference            
            # our implementation
            '''
            force_pred = gamdnet(pos, edge_index_list)  # Forward pass through the model
            msg_force_dict['edge_messages'] = gamdnet.mpnn_mlps.mp_blocks[3].edge_message_neigh_center                        
            evaluate(force_gt, force_pred)            
            '''
            # official implementation
            force_pred_official = gamdnet_official([pos],
                               [edge_index_list])
            msg_force_dict['edge_messages'] = gamdnet_official.graph_conv.conv[-1].edge_message_neigh_center 
            msg_force_dict['aggregate_edge_messages'] = gamdnet_official.graph_conv.conv[-1].aggregate_edge_messages
            
            print("Results from official model:")
            evaluate(force_gt, force_pred_official)
            
            # record messages for SR
            lj_force, lj_potential, radial_distance, valid_indices, dx, dy, dz = compute_lj_force_and_potential(pos, edge_index_list)                                 

            
            # remove nans
            lj_force = lj_force[valid_indices]
            msg_force_dict['edge_messages'] = msg_force_dict['edge_messages'][valid_indices]
            msg_force_dict['radial_distance'] = radial_distance[valid_indices]
            
            # Define a threshold for closeness to zero
            threshold = 1e-5

            # Identify rows where all elements are close to zero
            close_to_zero_rows = (torch.abs(lj_force) < threshold).all(dim=1)

            # Create a mask for rows that are NOT close to zero
            non_zero_rows_mask = ~close_to_zero_rows

            
            
            print("Before zero row removal, ", lj_force.shape, msg_force_dict['edge_messages'].shape)
            # Filter the tensor to keep only the non-zero rows
            lj_force = lj_force[non_zero_rows_mask]                     
            msg_force_dict['edge_messages'] = msg_force_dict['edge_messages'][non_zero_rows_mask]                                  
            msg_force_dict['force_gt'] = lj_force # [num_particle * batch_size, 3]
            msg_force_dict['radial_distance'] = msg_force_dict['radial_distance'][non_zero_rows_mask]
            msg_force_dict['dx'] = dx[valid_indices][non_zero_rows_mask]  
            msg_force_dict['dy'] = dy[valid_indices][non_zero_rows_mask]
            msg_force_dict['dz'] = dz[valid_indices][non_zero_rows_mask]
            msg_force_dict['potential_gt'] = lj_potential[valid_indices][non_zero_rows_mask]
            msg_force_dict['net_force_gt'] = force_gt
            msg_force_dict['pos'] = pos
            msg_force_dict['node_embeddings'] = gamdnet_official.graph_conv.conv[-1].node_embeddings
            

        break # run dataloader only once
    return msg_force_dict



def plot_param_sparsity(msg_force_dict, param_name):
    msg_array = msg_force_dict[param_name].cpu()
    msg_importance = msg_array.std(axis=0)      
    
    top_std_indices = torch.argsort(msg_importance)[-15:]
    top_importance_values = msg_importance[top_std_indices].numpy()
    print(f"{param_name} sparsity: ", top_importance_values)
    fig, ax = plt.subplots(1, 1)
    ax.pcolormesh(top_importance_values[None, ], cmap='gray_r', edgecolors='k')
    plt.axis('off')
    plt.grid(True)
    ax.set_aspect('equal')
    plt.text(15.5, 0.5, '...', fontsize=30)
    plt.tight_layout()    
    plt.show()


################################################################ test linearity hypothesis #################################

from scipy.optimize import minimize

msg_most_imp = None
expected_forces = None
expected_potentials = None

def percentile_sum(x):
    x = x.ravel()
    bot = x.min()
    top = np.percentile(x, 90)
    msk = (x>=bot) & (x<=top)
    frac_good = (msk).sum()/len(x)
    return x[msk].sum()/frac_good


def linear_transformation_3d_force(alpha):

    global msg_most_imp
    global expected_forces
    
    lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1] + alpha[2] * expected_forces[:, 2]) + alpha[3]
    lincomb2 = (alpha[0+4] * expected_forces[:, 0] + alpha[1+4] * expected_forces[:, 1] + alpha[2+4] * expected_forces[:, 2]) + alpha[3+4]
    lincomb3 = (alpha[0+8] * expected_forces[:, 0] + alpha[1+8] * expected_forces[:, 1] + alpha[2+8] * expected_forces[:, 2]) + alpha[3+8]
    '''
    score = (
        percentile_sum(np.square(msg_most_imp[:, 0] - lincomb1)) +
        percentile_sum(np.square(msg_most_imp[:, 1] - lincomb2)) +
        percentile_sum(np.square(msg_most_imp[:, 2] - lincomb3))
    )/3.0
    '''
    score = np.mean([np.abs(msg_most_imp[:, 0] - lincomb1) +
        np.abs(msg_most_imp[:, 1] - lincomb2) +
        np.abs(msg_most_imp[:, 2] - lincomb3)]) / 3.0
    
    print("Alpha now is: ", alpha)
    print("Score now is: ", score)
    return score


def out_linear_transformation_3d_force(alpha):

    global msg_most_imp
    global expected_forces

    lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1] + alpha[2] * expected_forces[:, 2]) + alpha[3]
    lincomb2 = (alpha[0+4] * expected_forces[:, 0] + alpha[1+4] * expected_forces[:, 1] + alpha[2+4] * expected_forces[:, 2]) + alpha[3+4]
    lincomb3 = (alpha[0+8] * expected_forces[:, 0] + alpha[1+8] * expected_forces[:, 1] + alpha[2+8] * expected_forces[:, 2]) + alpha[3+8]    
    return lincomb1, lincomb2, lincomb3


def linear_transformation_3d_potential(alpha):

    global msg_most_imp
    global expected_potentials
    
    lincomb1 = (alpha[0] * expected_potentials[:, 0] + alpha[1] * expected_potentials[:, 1] + alpha[2] * expected_potentials[:, 2]) + alpha[3]
    lincomb2 = (alpha[0+4] * expected_potentials[:, 0] + alpha[1+4] * expected_potentials[:, 1] + alpha[2+4] * expected_potentials[:, 2]) + alpha[3+4]
    lincomb3 = (alpha[0+8] * expected_potentials[:, 0] + alpha[1+8] * expected_potentials[:, 1] + alpha[2+8] * expected_potentials[:, 2]) + alpha[3+8]

    '''
    score = (
        percentile_sum(np.square(msg_most_imp[:, 0] - lincomb1)) +
        percentile_sum(np.square(msg_most_imp[:, 1] - lincomb2)) +
        percentile_sum(np.square(msg_most_imp[:, 2] - lincomb3))
    )/3.0
    '''
    score = np.mean([np.abs(msg_most_imp[:, 0] - lincomb1) + np.abs(msg_most_imp[:, 1] - lincomb2) + np.abs(msg_most_imp[:, 2] - lincomb3)]) / 3.0
    
    print("Alpha now is: ", alpha)
    print("Score now is: ", score)
    return score


def out_linear_transformation_3d_potential(alpha):

    global msg_most_imp
    global expected_potentials

    lincomb1 = (alpha[0] * expected_potentials[:, 0] + alpha[1] * expected_potentials[:, 1] + alpha[2] * expected_potentials[:, 2]) + alpha[3]
    lincomb2 = (alpha[0+4] * expected_potentials[:, 0] + alpha[1+4] * expected_potentials[:, 1] + alpha[2+4] * expected_potentials[:, 2]) + alpha[3+4]
    lincomb3 = (alpha[0+8] * expected_potentials[:, 0] + alpha[1+8] * expected_potentials[:, 1] + alpha[2+8] * expected_potentials[:, 2]) + alpha[3+8]
    
    print("alphas: ", alpha)
    return lincomb1, lincomb2,  lincomb3



def are_edge_msgs_gt_force_correlated(msg_force_dict):
    '''
    msg_force_dict: {'edge_messages': [total_edges, emb_dim], 'gt_force': [total_edges, 3]}
    '''
    global msg_most_imp
    global expected_forces

    print("edge message shape: ", msg_force_dict['edge_messages'].shape)
    print("force shape: ", msg_force_dict['force_gt'].shape)

    # Calculate variance for each component of agg_msg across all samples
    msg_comp_std = torch.std(msg_force_dict['edge_messages'], axis=0)  # Variance for each component

    # Step 3: Get top-3 indices based on variance
    top_std_indices = torch.argsort(msg_comp_std)[-3:]  # Get indices of top-3 components with maximum variance  

    # Prepare data for linear regression using top-3 components as output variables
    msg_most_imp = msg_force_dict['edge_messages'][:, top_std_indices].cpu()  # Select only the top-3 components

    
    # normalize the messages
    #msg_most_imp = ((msg_most_imp - torch.mean(msg_most_imp, axis=0)) / torch.std(msg_most_imp, axis=0)).cpu()
    
    expected_forces = msg_force_dict['force_gt'].cpu()

    
    dim = 3
    min_result = minimize(linear_transformation_3d_force, np.ones(dim**2 + dim), method='Powell')

    print("Fit score: ", min_result.fun/msg_force_dict['edge_messages'].shape[0])
    

    # Visualize the fit
    for i in range(dim):
        px = out_linear_transformation_3d_force(min_result.x)[i]
        py = msg_most_imp[:, i] 
        plt.scatter(px, py)
        plt.show()
    
    are_correlated = False
    return are_correlated, msg_most_imp



def are_edge_msgs_gt_potential_correlated(msg_force_dict):
    '''
    msg_force_dict: {'edge_messages': [total_edges, emb_dim], 'gt_force': [total_edges, 3]}
    '''
    global msg_most_imp
    global expected_potentials

    print("edge message shape: ", msg_force_dict['edge_messages'].shape)
    print("force shape: ", msg_force_dict['potential_gt'].shape)

    # Calculate variance for each component of agg_msg across all samples
    msg_comp_std = torch.std(msg_force_dict['edge_messages'], axis=0)  # Variance for each component

    # Step 3: Get top-3 indices based on variance
    top_std_indices = torch.argsort(msg_comp_std)[-3:]  # Get indices of top-3 components with maximum variance  

    # Prepare data for linear regression using top-3 components as output variables
    msg_most_imp = msg_force_dict['edge_messages'][:, top_std_indices].cpu()  # Select only the top-3 components

    
    # normalize the messages
    #msg_most_imp = ((msg_most_imp - torch.mean(msg_most_imp, axis=0)) / torch.std(msg_most_imp, axis=0)).cpu()
    
    expected_potentials = msg_force_dict['potential_gt'].cpu()

    
    dim = 3
    min_result = minimize(linear_transformation_3d_potential, np.ones(dim**2 + dim), method='Powell')

    print("Fit score: ", min_result.fun/msg_force_dict['edge_messages'].shape[0])

    # Visualize the fit
    for i in range(dim):
        px = out_linear_transformation_3d_potential(min_result.x)[i]
        py = msg_most_imp[:, i]
        plt.scatter(px, py)
        plt.show()
    
    are_correlated = False
    return are_correlated, msg_most_imp




def plot_lj_force_vs_rad_dist_with_messages(msg_force_dict):
    edge_messages = msg_force_dict['edge_messages'].cpu()
    lj_force = msg_force_dict['force_gt'].cpu()
    r = msg_force_dict['radial_distance'].cpu()

    lj_force_std = torch.std(lj_force, dim=0)    
    lj_force_comp1_index = torch.argsort(lj_force_std)[-1] # most imp. last
    lj_force_comp_1 = lj_force[:, lj_force_comp1_index]
    print("LJ force stds: ", lj_force_std[torch.argsort(lj_force_std)[-3:]])
    print("LJ force means: ", torch.mean(lj_force, axis = 0)[torch.argsort(lj_force_std)[-3:]])
    
    edge_msg_comp_std = torch.std(edge_messages, dim=0)        
    edge_msg_comp1_index = torch.argsort(edge_msg_comp_std)[-1] # most imp. last    
    msg_comp_1 = edge_messages[:, edge_msg_comp1_index]
    print("Edge message stds: ", edge_msg_comp_std[torch.argsort(edge_msg_comp_std)[-3:]])
    print("Edge message means: ", torch.mean(edge_messages, axis = 0)[torch.argsort(edge_msg_comp_std)[-3:]])

    # Create first subplot
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.scatter(r, lj_force_comp_1, color='red', alpha=0.6) 
    plt.title('LJ force comp-1 vs radial distance')
    plt.xlabel('radial distance')
    plt.ylabel('LJ force comp-1')

    # Create second subplot
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.scatter(r, msg_comp_1, color='blue', alpha=0.6)
    plt.title('Edge msg comp-1 vs radial distance')
    plt.xlabel('radial distance')
    plt.ylabel('edge msg. comp-1')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.show()



def plot_lj_potential_vs_rad_dist_with_messages(msg_force_dict):
    edge_messages = msg_force_dict['edge_messages'].cpu()
    lj_potential = msg_force_dict['potential_gt'].cpu()
    r = msg_force_dict['radial_distance'].cpu()

    lj_potential_std = torch.std(lj_potential, dim=0)    
    lj_potential_comp1_index = torch.argsort(lj_potential_std)[-1] # most imp. last
    lj_potential_comp_1 = lj_potential[:, lj_potential_comp1_index]
    print("LJ potential stds: ", lj_potential_std[torch.argsort(lj_potential_std)[-3:]])
    print("LJ potential means: ", torch.mean(lj_potential, axis = 0)[torch.argsort(lj_potential_std)[-3:]])

    edge_msg_comp_std = torch.std(edge_messages, dim=0)    
    edge_msg_comp1_index = torch.argsort(edge_msg_comp_std)[-1] # most imp. last
    msg_comp_1 = edge_messages[:, edge_msg_comp1_index]
    print("Edge message stds: ", edge_msg_comp_std[torch.argsort(edge_msg_comp_std)[-3:]])
    print("Edge message means: ", torch.mean(edge_messages, axis = 0)[torch.argsort(edge_msg_comp_std)[-3:]])

    # Create first subplot
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.scatter(r, lj_potential_comp_1, color='red', alpha=0.6)
    plt.title('LJ potential comp-1 vs radial distance')
    plt.xlabel('radial distance')
    plt.ylabel('LJ potential comp-1')

    # Create second subplot
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.scatter(r, msg_comp_1, color='blue', alpha=0.6)
    plt.title('Edge msg comp-1 vs radial distance')
    plt.xlabel('radial distance')
    plt.ylabel('edge msg. comp-1')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.show()



################################################## Do, symbolic regression if linearity exists #########################################
def drop_outliers(data):
    # Calculate IQR for each column and identify outliers
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    # Determine outliers based on IQR
    outlier_condition = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))

    # Remove outliers
    data = data[~outlier_condition.any(axis=1)]        
    return data



def regress_edge_message_equation(msg_force_dict):     
    
    dx = msg_force_dict['dx'].cpu()
    dy = msg_force_dict['dy'].cpu()
    dz = msg_force_dict['dz'].cpu()
    r = msg_force_dict['radial_distance'].cpu()
    
    msg_comp_std = torch.std(msg_force_dict['edge_messages'], axis = 0)
    msg_comp_most_imp_indices = torch.argsort(msg_comp_std)[-3:] # in ascending order
    msg_most_imp = msg_force_dict['edge_messages'][:, msg_comp_most_imp_indices]
    z1 = msg_most_imp[:, -1].cpu() # the one with highest std
    z2 = msg_most_imp[:, -2].cpu()
    z3 = msg_most_imp[:, -3].cpu()

    # Create a DataFrame for easier handling of data
    data = pd.DataFrame({
        'dx': dx.squeeze(),
        'dy': dy.squeeze(),
        'dz': dz.squeeze(),
        'r': r.squeeze(),
        'z1': z1.squeeze(),
        'z2': z2.squeeze(),
        'z3': z3.squeeze(),
    })


    #print("Data shape before outlier removal: ", data.shape)
    #data = drop_outliers(data)    
    #print("Data shape after outlier removal: ", data.shape)
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Set up the matplotlib figure
    plt.figure(figsize=(15, 10))
    # Calculate IQR for each column and identify outliers
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    # Determine outliers based on IQR
    outlier_condition = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))

    # Display outliers
    outliers = data[outlier_condition.any(axis=1)]
    print("Outliers based on IQR:")
    print(outliers)

    # Optional: Visualize the IQR ranges
    plt.figure(figsize=(10, 6))
    for col in data.columns:
        plt.subplot(3, 3, list(data.columns).index(col) + 1)
        sns.boxplot(data[col])
        plt.title(f'IQR Visualization: {col}')

    plt.tight_layout()
    plt.show()

    exit(0)
    '''


    
    # Define the features and target variable
    X = data[['dx', 'dy', 'dz', 'r']].values
    #X = data[['r']].values
    
    for comp_id in range(3):
        y = data[f"z{comp_id + 1}"].values
        print(f"Fitting edge message comp-{comp_id + 1} now....")
        '''
        model = PySRRegressor(
        niterations=1000,
        model_selection="accuracy",
        binary_operators=["-", "/", "pow"],        
        unary_operators=[
            "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],      
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        # ^ Custom loss function (julia syntax)  
        complexity_of_variables=2,
        constraints={"pow": (-1, 1)}, 
        batching=True, 
        )
        '''
        model = PySRRegressor(
        niterations=1000,  # < Increase me for better results
        model_selection="accuracy",
        binary_operators=["+", "*"],
        unary_operators=[
            "cos",
            "exp",
            "sin",
            "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        # ^ Custom loss function (julia syntax)
        batching=True, 
        )
        
        # Fit the model to the data
        model.fit(X, y, variable_names = ['dx', 'dy', 'dz', 'r'])
        #model.fit(X, y, variable_names = ['r'])
        model_filename = f"pysr_model_msg_{comp_id}_pred.pkl"

        # Save the model to a file
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)

        # Get the best equation found by PySR
        best_equation = model.get_best()
        
        print(f"Best equation for edge message comp-{comp_id + 1} as a function of dx, dy, dz, and r:")
        print(best_equation)        
        from matplotlib import pyplot as plt
        plt.scatter(y, model.predict(X))
        plt.xlabel('Truth')
        plt.ylabel('Prediction')
        plt.show()        


def regress_net_force_equation(msg_force_dict):
    x = msg_force_dict['pos'][:, 0].cpu()
    y = msg_force_dict['pos'][:, 1].cpu()
    z = msg_force_dict['pos'][:, 2].cpu()    
    
    aggregate_edge_messages = msg_force_dict['aggregate_edge_messages']    
    agg_msg_comp_std = torch.std(aggregate_edge_messages, axis = 0)
    agg_msg_comp_most_imp_indices = torch.argsort(agg_msg_comp_std)[-3:] # in ascending order
    agg_msg_most_imp = aggregate_edge_messages[:, agg_msg_comp_most_imp_indices]
    
    agg_comp_1 = agg_msg_most_imp[:, -1].cpu() # the one with highest std
    agg_comp_2 = agg_msg_most_imp[:, -2].cpu()
    agg_comp_3 = agg_msg_most_imp[:, -3].cpu()

    force_gt = msg_force_dict['net_force_gt'].cpu()    

    print("Shapes: ", x.squeeze().shape, agg_comp_1.squeeze().shape, force_gt[:, 0].squeeze().shape)
    # Create a DataFrame for easier handling of data
    data = pd.DataFrame({
        'x': x.squeeze(),
        'y': y.squeeze(),
        'z': z.squeeze(),        
        'agg_comp_1': agg_comp_1.squeeze(),
        'agg_comp_2': agg_comp_2.squeeze(),
        'agg_comp_3': agg_comp_3.squeeze(),
        'force_gt_comp_1': force_gt[:, 0].squeeze(),
        'force_gt_comp_2': force_gt[:, 1].squeeze(),
        'force_gt_comp_3': force_gt[:, 2].squeeze(),
    })
    
    #print("Data shape before outlier removal: ", data.shape)
    #data = drop_outliers(data)    
    #print("Data shape after outlier removal: ", data.shape)

    # Define the features and target variable
    X = data[['x', 'y', 'z', 'agg_comp_1', 'agg_comp_2', 'agg_comp_3']].values
    
    for comp_id in range(3):
        y = data[f"force_gt_comp_{comp_id + 1}"].values
        print(f"Fitting force_gt comp-{comp_id + 1} now....")
        '''
        model = PySRRegressor(
            niterations=10000,  # < Increase me for better results
            model_selection="accuracy",
            binary_operators=["-", "/", "^"],
            unary_operators=[
                "inv(x) = 1/x",
                # ^ Custom operator (julia syntax)
            ],
            extra_sympy_mappings={"inv": lambda x: 1 / x},
            # ^ Define operator for SymPy as well
            elementwise_loss="loss(prediction, target) = abs(prediction - target)",
            # ^ Custom loss function (julia syntax)
          complexity_of_variables=2,
        constraints={"pow": (-1, 1)}, 
        batching=True, 
        )  
        '''
        model = PySRRegressor(
        niterations=10000,  # < Increase me for better results
        model_selection="accuracy",
        binary_operators=["+", "*"],
        unary_operators=[
            "cos",
            "exp",
            "sin",
            "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        # ^ Custom loss function (julia syntax)
        batching=True, 
        )

        # Fit the model to the data
        model.fit(X, y)
        
        model_filename = f"pysr_model_netforce_{comp_id}_pred.pkl"

        # Save the model to a file
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
      
        # Get the best equation found by PySR
        best_equation = model.get_best()
        
        print(f"Best equation for force gt comp-{comp_id + 1} as a function of x, y, z, and aggreggate edge messages is: ")
        print(best_equation)


    return



def regress_net_force_equation_node_embeddings(msg_force_dict):
    
    
    node_embeddings_std = torch.std(msg_force_dict['node_embeddings'], axis = 0)
    node_emb_most_imp_idx = torch.argsort(node_embeddings_std)[-6:] # top-3
    node_embeddings_most_imp = msg_force_dict['node_embeddings'][:, node_emb_most_imp_idx]
    
    x = node_embeddings_most_imp[:, -1].cpu()
    y = node_embeddings_most_imp[:, -2].cpu()
    z = node_embeddings_most_imp[:, -3].cpu()    
    p = node_embeddings_most_imp[:, -4].cpu()
    q = node_embeddings_most_imp[:, -5].cpu()
    r = node_embeddings_most_imp[:, -6].cpu()
    '''
    aggregate_edge_messages = msg_force_dict['aggregate_edge_messages']    
    agg_msg_comp_std = torch.std(aggregate_edge_messages, axis = 0)
    agg_msg_comp_most_imp_indices = torch.argsort(agg_msg_comp_std)[-3:] # in ascending order
    agg_msg_most_imp = aggregate_edge_messages[:, agg_msg_comp_most_imp_indices]
    
    agg_comp_1 = agg_msg_most_imp[:, -1].cpu() # the one with highest std
    agg_comp_2 = agg_msg_most_imp[:, -2].cpu()
    agg_comp_3 = agg_msg_most_imp[:, -3].cpu()
    '''
    force_gt = msg_force_dict['net_force_gt'].cpu()    

    
    # Create a DataFrame for easier handling of data
    data = pd.DataFrame({
        'x': x.squeeze(),
        'y': y.squeeze(),
        'z': z.squeeze(),        
        'p': p.squeeze(),
        'q': q.squeeze(),
        'r': r.squeeze(),
        'force_gt_comp_1': force_gt[:, 0].squeeze(),
        'force_gt_comp_2': force_gt[:, 1].squeeze(),
        'force_gt_comp_3': force_gt[:, 2].squeeze(),
    })
    
    #print("Data shape before outlier removal: ", data.shape)
    #data = drop_outliers(data)    
    #print("Data shape after outlier removal: ", data.shape)

    # Define the features and target variable
    #X = data[['x', 'y', 'z', 'agg_comp_1', 'agg_comp_2', 'agg_comp_3']].values
    X = data[['x', 'y', 'z', 'p', 'q', 'r']].values
    for comp_id in range(3):
        y = data[f"force_gt_comp_{comp_id + 1}"].values
        print(f"Fitting force_gt comp-{comp_id + 1} now....")
        '''
        model = PySRRegressor(
            niterations=10000,  # < Increase me for better results
            model_selection="accuracy",
            binary_operators=["-", "/", "^"],
            unary_operators=[
                "inv(x) = 1/x",
                # ^ Custom operator (julia syntax)
            ],
            extra_sympy_mappings={"inv": lambda x: 1 / x},
            # ^ Define operator for SymPy as well
            elementwise_loss="loss(prediction, target) = abs(prediction - target)",
            # ^ Custom loss function (julia syntax)
          complexity_of_variables=2,
        constraints={"pow": (-1, 1)}, 
        batching=True, 
        )  
        '''
        model = PySRRegressor(
        niterations=10000,  # < Increase me for better results
        model_selection="accuracy",
        binary_operators=["+", "*"],
        unary_operators=[
            "cos",
            "exp",
            "sin",
            "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        # ^ Custom loss function (julia syntax)
        batching=True, 
        )

        # Fit the model to the data
        model.fit(X, y)
        
        model_filename = f"pysr_model_netforce_node_embeddings_{comp_id}_pred.pkl"

        # Save the model to a file
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
      
        # Get the best equation found by PySR
        best_equation = model.get_best()
        
        print(f"Best equation for force gt comp-{comp_id + 1} as a function of node embeddings, and aggreggate edge messages is: ")
        print(best_equation)


    return


def fit_decision_tree_edge_msg_prediction(msg_force_dict):
    print("Fitting decision tree for edge messsage prediction...")
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor, plot_tree
    from sklearn.metrics import mean_absolute_error

    dx = msg_force_dict['dx'].cpu()
    dy = msg_force_dict['dy'].cpu()
    dz = msg_force_dict['dz'].cpu()
    r = msg_force_dict['radial_distance'].cpu()
    
    msg_comp_std = torch.std(msg_force_dict['edge_messages'], axis = 0)
    msg_comp_most_imp_indices = torch.argsort(msg_comp_std)[-3:] # in ascending order
    msg_most_imp = msg_force_dict['edge_messages'][:, msg_comp_most_imp_indices]
    z1 = msg_most_imp[:, -1].cpu() # the one with highest std
    z2 = msg_most_imp[:, -2].cpu()
    z3 = msg_most_imp[:, -3].cpu()

    # Create a DataFrame for easier handling of data
    data = pd.DataFrame({
        'dx': dx.squeeze(),
        'dy': dy.squeeze(),
        'dz': dz.squeeze(),
        'r': r.squeeze(),
        'z1': z1.squeeze(),
        'z2': z2.squeeze(),
        'z3': z3.squeeze(),
    })
    
    X = data[['dx', 'dy', 'dz', 'r']].values    
    Y = data[['z1', 'z2', 'z3']].values

    # Creating the Decision Tree Regressor
    regressor = DecisionTreeRegressor(random_state=42, max_depth=4, min_samples_split=10, min_samples_leaf=10, 
    max_leaf_nodes=20, ccp_alpha=0.01)

    # Fitting the model
    regressor.fit(X , Y)


    Y_pred = regressor.predict(X)
    # Evaluating the model
    mae = mean_absolute_error(Y , Y_pred)
    print(f'Mean absolute Error: {mae:.2f}')

    # Plotting Predictions vs Ground Truth
    plt.figure(figsize=(8,6))
    plt.scatter(Y, Y_pred, color='blue', label='Predictions', alpha=0.7)
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', linestyle='--', label='Perfect Prediction')
    plt.title('Predictions vs Ground Truth (Regression)')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.legend()
    plt.grid()
    plt.show()


    # Visualize the decision tree
    plt.figure(figsize=(20, 10))  # Set figure size for better readability
    plot_tree(regressor, feature_names=['dx', 'dy', 'dz', 'r'], filled=True, rounded=True)
    plt.title("Decision Tree Regressor Visualization")
    plt.show()    




def fit_decision_tree_net_force_prediction(msg_force_dict):
    print("Fitting decision tree for netforce prediction...")
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor, plot_tree
    from sklearn.metrics import mean_absolute_error

    node_embeddings_std = torch.std(msg_force_dict['node_embeddings'], axis = 0)
    node_emb_most_imp_idx = torch.argsort(node_embeddings_std)[-6:] # top-3
    node_embeddings_most_imp = msg_force_dict['node_embeddings'][:, node_emb_most_imp_idx]
    
    x = node_embeddings_most_imp[:, -1].cpu()
    y = node_embeddings_most_imp[:, -2].cpu()
    z = node_embeddings_most_imp[:, -3].cpu()    
    p = node_embeddings_most_imp[:, -4].cpu()
    q = node_embeddings_most_imp[:, -5].cpu()
    r = node_embeddings_most_imp[:, -6].cpu()
    force_gt = msg_force_dict['net_force_gt'].cpu()    

    
    # Create a DataFrame for easier handling of data
    data = pd.DataFrame({
        'x': x.squeeze(),
        'y': y.squeeze(),
        'z': z.squeeze(),        
        'p': p.squeeze(),
        'q': q.squeeze(),
        'r': r.squeeze(),
        'force_gt_comp_1': force_gt[:, 0].squeeze(),
        'force_gt_comp_2': force_gt[:, 1].squeeze(),
        'force_gt_comp_3': force_gt[:, 2].squeeze(),
    })
    
    X = data[['x', 'y', 'z', 'p', 'q', 'r']].values
    #X = data[['x', 'y', 'z']].values
    Y = data[['force_gt_comp_1', 'force_gt_comp_2', 'force_gt_comp_3']].values

    # Creating the Decision Tree Regressor
    regressor = DecisionTreeRegressor(random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=10, 
    max_leaf_nodes=20, ccp_alpha=0.01)

    # Fitting the model
    regressor.fit(X , Y)


    Y_pred = regressor.predict(X)
    # Evaluating the model
    mae = mean_absolute_error(Y , Y_pred)
    print(f'Mean absolute Error: {mae:.2f}')

    # Plotting Predictions vs Ground Truth
    plt.figure(figsize=(8,6))
    plt.scatter(Y, Y_pred, color='blue', label='Predictions', alpha=0.7)
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', linestyle='--', label='Perfect Prediction')
    plt.title('Predictions vs Ground Truth (Regression)')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.legend()
    plt.grid()
    plt.show()


    # Visualize the decision tree
    plt.figure(figsize=(20, 10))  # Set figure size for better readability
    plot_tree(regressor, feature_names=['x', 'y', 'z','p','q', 'r'], filled=True, rounded=True)
    plt.title("Decision Tree Regressor Visualization")
    plt.show()    




if __name__ == '__main__':
    gamd_model_weights_filename = 'best_model_vectorized_message_passing.pt'

    #gamdnet_official_model_checkpoint_filename = 'checkpoint.ckpt'
    #gamdnet_official_model_checkpoint_filename = 'epoch=29-step=270000_standard.ckpt'
    #gamdnet_official_model_checkpoint_filename = 'epoch=29-step=270000_l1_message.ckpt'
    #gamdnet_official_model_checkpoint_filename = 'epoch=29-step=270000_l1_message_node_embed.ckpt'
    #gamdnet_official_model_checkpoint_filename = 'epoch=29-step=270000_l1_0.1_reg_message_node_embed.ckpt'
    #gamdnet_official_model_checkpoint_filename = 'epoch=39-step=360000_bottleneck.ckpt'
    gamdnet_official_model_checkpoint_filename = 'epoch=39-step=360000_edge_msg_constrained_std.ckpt'
    md_filedir = '../top/'
    gamdnet, gamdnet_official, dataloader = load_model_and_dataset(gamd_model_weights_filename, gamdnet_official_model_checkpoint_filename, md_filedir)
    msg_force_dict = get_msg_force_dict(gamdnet, gamdnet_official, dataloader)

    '''
    print("Visualizing sparsity of message components...")
    plot_param_sparsity(msg_force_dict, "edge_messages")
    
    print("Checking the fit between pair force and edge messages now....")
    are_edge_msgs_gt_force_correlated(msg_force_dict)
    
    print("Checking the fit between pair potentials and edge messages now....")
    are_edge_msgs_gt_potential_correlated(msg_force_dict)
    
    # REGRESS EQ-1
    print("Finding equations for top-3 message components...")
    regress_edge_message_equation(msg_force_dict)
    
    #plot_lj_force_vs_rad_dist_with_messages(msg_force_dict)
    #plot_lj_potential_vs_rad_dist_with_messages(msg_force_dict)
    
    # REGRESS EQ-2
    print("Finding equations for net force components...")
    regress_net_force_equation(msg_force_dict)
    
    # REGRESS EQ-3
    print("Finding equations for net force components as function of agg. msg, node embeddings...")
    regress_net_force_equation_node_embeddings(msg_force_dict)
    
    '''
    # Fit decision tree for net force prediction    
    fit_decision_tree_net_force_prediction(msg_force_dict)
    

    # Fit decision tree for edge message prediction
    fit_decision_tree_edge_msg_prediction(msg_force_dict)
