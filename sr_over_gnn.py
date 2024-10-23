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
            
            if self.update_edge_emb:
                normalized_e_emb = self.edge_layer_norm(g.edata['e_emb'])
            g.update_all(fn.u_mul_e('h', 'e_emb', 'm'), fn.sum('m', 'h'))
            edge_emb = g.ndata['h']

        if self.update_edge_emb:
            g.edata['e'] = normalized_e_emb
        node_feat = self.phi(self.phi_dst(h) + self.phi_edge(edge_emb))
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
    num_input_files = 100#len(os.listdir(md_filedir))
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
    checkpoint = torch.load(gamdnet_official_model_checkpoint_filename)    
    
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
    return None, gamdnet_official, dataloader


def compute_lj_force(pos, edge_index_list):
    
    center_node_idx = edge_index_list[0, :]
    neigh_node_idx = edge_index_list[1, :]

    neigh_node_pos = pos[neigh_node_idx]
    center_node_pos = pos[center_node_idx]

    # Calculate the distance vector
    r_vec = neigh_node_pos - center_node_pos  # Shape: [n, 3]
    
    # Calculate the distance (magnitude)
    r = torch.norm(r_vec, dim=1).unsqueeze(1)  # Shape: [n, 1]

    # Avoid division by zero
    r = torch.where(r == 0, torch.tensor(1e-10, dtype=r.dtype), r)
    
    epsilon = 0.238
    sigma = 3.4 
    force_magnitude = 24 * epsilon * (
        2 * (sigma ** 12) / (r ** 13) - 
        (sigma ** 6) / (r ** 7)
    )  # Shape: [n, 1]

    # Calculate the force vector (directed)
    force_vector = force_magnitude * (r_vec / r)  # Shape: [n, 3]    
    force_vector = torch.nan_to_num(force_vector, nan=0.0)
    return force_vector



def get_msg_force_dict(gamdnet, gamdnet_official, dataloader):
    msg_force_dict = {}
    # run inference over the input batched graph from dataloader
    # record aggregate edge messages and force ground truths for each node in output dictionary
    for pos, edge_index_list, force_gt in dataloader:
        with torch.no_grad():  # Disable gradient calculation for inference            
            # our implementation
            #force_pred = gamdnet(pos, edge_index_list)  # Forward pass through the model
            #msg_force_dict['edge_messages'] = gamdnet.mpnn_mlps.mp_blocks[3].edge_message_neigh_center 
            # Now, unload model 1 from GPU before inference on model 2
            #del gamdnet  # Delete the reference to model 1

            #evaluate(force_gt, force_pred)            
            # official implementation
            force_pred_official = gamdnet_official([pos],
                               [edge_index_list])
            print("Results from official model:")
            evaluate(force_gt, force_pred_official)
            
            # record messages for SR
            lj_force = compute_lj_force(pos, edge_index_list)
                       
            msg_force_dict['edge_messages'] = gamdnet_official.graph_conv.conv[-1].edge_message_neigh_center
            msg_force_dict['force_gt'] = lj_force # [num_particle * batch_size, 3]
                        
        break # run dataloader only once
    return msg_force_dict




################################################################ test linearity hypothesis #################################

def are_edge_msgs_gt_force_correlated(msg_force_dict):
    '''
    msg_force_dict: {'edge_messages': [total_edges, emb_dim], 'gt_force': [total_edges, 3]}
    '''

    print("edge message shape: ", msg_force_dict['edge_messages'].shape)
    print("force shape: ", msg_force_dict['force_gt'].shape)

    # Calculate variance for each component of agg_msg across all samples
    msg_comp_std = torch.std(msg_force_dict['edge_messages'], axis=0)  # Variance for each component

    # Step 3: Get top-3 indices based on variance
    top_std_indices = torch.argsort(msg_comp_std)[-3:]  # Get indices of top-3 components with maximum variance

    # Output the results of top-3 components
    print("Top-3 components with maximum STD are: ", top_std_indices)
    print("std values: ", msg_comp_std[top_std_indices])
    

    # Prepare data for linear regression using top-3 components as output variables
    agg_msg_selected_values = msg_force_dict['edge_messages'][:, top_std_indices]  # Select only the top-3 components

    # Performing linear regression for each selected agg_msg component against gt_force
    for component_index in range(3):
        model = LinearRegression()
        
        # Fit model to predict selected agg_msg component based on gt_force components
        model.fit(msg_force_dict['force_gt'].cpu(), agg_msg_selected_values[:, component_index].cpu())
        
        slope = model.coef_
        intercept = model.intercept_
        
        # Calculate R^2 score
        predictions = model.predict(msg_force_dict['force_gt'].cpu())
        r2 = r2_score(msg_force_dict['edge_messages'][:, component_index].cpu(), predictions)
        
        print(f"\nLinear fit results for Component {component_index + 1}:")
        print(f"Slope: {slope}, Intercept: {intercept}, R^2 Score: {r2}")
        x = predictions
        y = msg_force_dict['edge_messages'][:, component_index].cpu() 
        plt.plot(x, y)
        plt.show()
    are_correlated = False
    return are_correlated








gamd_model_weights_filename = 'best_model_vectorized_message_passing.pt'
gamdnet_official_model_checkpoint_filename = 'checkpoint.ckpt'
md_filedir = '../top/'
gamdnet, gamdnet_official, dataloader = load_model_and_dataset(gamd_model_weights_filename, gamdnet_official_model_checkpoint_filename, md_filedir)
msg_force_dict = get_msg_force_dict(gamdnet, gamdnet_official, dataloader)

print("Result: ", are_edge_msgs_gt_force_correlated(msg_force_dict))
