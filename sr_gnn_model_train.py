'''
1. Load dataloader, force decoder model weights for GAMDNet
2. In the training loop:
    1. for each batch of pos, edge index list and force gt:
        1. compute e1, e2, e3
        2. compute n1, n2…nk
        3. compute aggregate messages using e1, e2, e3
        4. compute node embeddings tensor (128-D) using n1, n2 … nk
        5. pass above inputs to https://github.com/BaratiLab/GAMD/blob/main/code/nn_module.py#L147
        6. pass resulting node embeddings to force MLP to get force prediction
        7. compute loss wrt. force gt
'''
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monolithic_gamd_impl import MDDataset, custom_collate, evaluate, GAMDLoss

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
        return self.mlp_layer(feat.to('cuda:2'))

def init_dataloader_and_models(train_data_fraction, avg_num_neighbors, rotation_aug, return_train_data,
    num_input_files, batch_size, lambda_reg, md_filedir, in_node_feats, hidden_dim, out_node_feats, activation, 
    encoding_size, out_feats):
    print("Loading input files: ", num_input_files)
    dataset = MDDataset(md_filedir, rotation_aug, avg_num_neighbors, train_data_fraction, return_train_data) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    print("Dataloader initialized.")   

    phi_dst = nn.Linear(3, hidden_dim) # input size matches the dimensionality of problem.
    phi_edge = nn.Linear(3, hidden_dim)
    phi = MLP(hidden_dim, out_node_feats,
                    activation_first=True, hidden_layer=1, hidden_dim=hidden_dim, activation=activation)

    force_decoder = MLP(encoding_size, out_feats, hidden_layer=2, hidden_dim=hidden_dim, activation='gelu')
    srgnn_loss = GAMDLoss(lambda_reg)
    return dataloader, srgnn_loss, phi, phi_dst, phi_edge, force_decoder

def inv(x):
    try:
        return torch.reciprocal(x + 1e-2)
    except:
        print("Divide by 0, please fix")
        exit(0) 

def sin(x):
    return torch.sin(torch.tensor(x))

def exp(x):
    return torch.exp(torch.tensor(x))

def cos(x):
    return torch.cos(torch.tensor(x))

# SR-MLP model inference
def forward(pos, edge_index_list): 
    center_node_index = edge_index_list[0, :]
    neighbor_node_index = edge_index_list[1, :]
    
    dx = pos[neighbor_node_index, 0] - pos[center_node_index, 0]
    dy = pos[neighbor_node_index, 1] - pos[center_node_index, 1]
    dz = pos[neighbor_node_index, 2] - pos[center_node_index, 2]

    # Calculate the distance vector
    center_node_pos = pos[center_node_index]
    neighbor_node_pos = pos[neighbor_node_index]
    r_vec = neighbor_node_pos - center_node_pos
    
    # Calculate the distance (magnitude)
    r = torch.norm(r_vec, dim=1).unsqueeze(1)  # Shape: [n, 1]
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]
    pos_z = pos[:, 2]   

    # create edge messages tensor using eq. from SR
    e1 = (0.2722 * (((inv(dx - 1.6381) + (0.35378 / dx)) * -1.9265e-05) / 0.55934)) / r
    e2 = 0.77933 * (((-9.2687e-06 / r) / ((dx + -1.2794) - dz)) / (-0.57706 + dz)) 
    e3 = inv(r) * ((inv(inv(inv(-1.2083))) - dy) * (8.9686e-07 / (dz - 0.57655)))
    edge_messages = torch.cat((e1, e2), axis = 1)
    edge_messages = torch.cat((edge_messages, e3), axis = 1)


    aggregate_edge_messages = torch.zeros((pos.shape[0], edge_messages.shape[1]), dtype=torch.float32)
    aggregate_edge_messages.index_add_(0, center_node_index, edge_messages)  # Accumulate messages into AGG    
    
    
    agg_msg_comp_std = torch.std(aggregate_edge_messages, axis = 0)
    agg_msg_comp_most_imp_indices = torch.argsort(agg_msg_comp_std)[-3:] # in ascending order
    agg_msg_most_imp = aggregate_edge_messages[:, agg_msg_comp_most_imp_indices]
    
    agg_comp_1 = agg_msg_most_imp[:, -1]
    agg_comp_2 = agg_msg_most_imp[:, -2]
    agg_comp_3 = agg_msg_most_imp[:, -3]

    # create node embeddings tensor using eq. from SR
    n1 = (1.0566 / ((3.435 / agg_comp_2) - ((inv(-0.78501 + agg_comp_3) + sin(sin(-0.78501) * pos_y)) - 0.47231))) - -0.0053185
    n2 = (((-0.067859 * sin(0.28926 * pos_z)) - agg_comp_1) / (sin(inv(sin(agg_comp_1)) - -0.76162) + exp(1.4285))) / 1.1111
    n3 = sin(sin(agg_comp_3 / ((-3.9187 + (cos(sin(inv(agg_comp_3))) / pos_y)) - ((agg_comp_2 + agg_comp_3) * (1.38 + agg_comp_2))))) 
    
    print('HSAPES: ', n1.shape, n2.shape, n3.shape, e1.shape, e2.shape, e3.shape)

    node_embeddings = torch.cat((n1, n2), axis = 1)
    node_embeddings = torch.cat((node_embeddings, n3), axis = 1)

    node_feat = phi(phi_dst(node_embeddings) + phi_edge(aggregate_edge_messages)) # So, we need to predict edge_emb (aggrgated edge messages), h (input node embeddings) inputs for the last message passing layer using SR to effecitively couple SR with the remaining GNN.  
    force_pred = force_decoder(node_feat)    
    
    return force_pred





######################################## Control script #####################################################
train_data_fraction = 1.0 # select 9k for training
avg_num_neighbors = 20 # criteria for connectivity of atoms for any frame
rotation_aug = False # online rotation augmentation for a frame    
# create train data-loader
return_train_data = True
md_filedir = '../top/'
num_input_files = 10#len(os.listdir(md_filedir))
batch_size = 1 
encoding_size = 128
out_feats = 3
hidden_dim = 128
lambda_reg = 1e-2
activation = 'silu'
in_node_feats = 3
out_node_feats = 128

dataloader, srgnn_loss, phi, phi_dst, phi_edge, force_decoder = init_dataloader_and_models(train_data_fraction, 
avg_num_neighbors, rotation_aug, return_train_data, num_input_files, batch_size, lambda_reg, md_filedir, in_node_feats,
hidden_dim, out_node_feats, activation, encoding_size, out_feats)

sr_loss = srgnn_loss.to('cpu')
phi = phi.to('cpu')
phi_dst = phi_dst.to('cpu')
phi_edge = phi_edge.to('cpu')
force_decoder = force_decoder.to('cpu')

n_epochs = 100
# train model
for epoch_id in range(n_epochs):
    for iter_id, (pos, edge_index_list, force_gt) in enumerate(dataloader):
        pos = pos.to('cpu')
        edge_index_list = edge_index_list.to('cpu')
        force_pred = forward(pos, edge_index_list)
        loss = sr_loss(force_pred, force_gt)
        print("Loss value: ", loss.item())
        loss.backward()
        if iter_id % 100 == 0:
            evaluate(force_gt, force_pred)

