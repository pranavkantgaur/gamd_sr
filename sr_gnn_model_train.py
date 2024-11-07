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

import torch.nn as nn
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
        return self.mlp_layer(feat.to('cuda:0'))

def init_dataloader_and_models(train_data_fraction, avg_num_neighbors, rotation_aug, return_train_data,
    num_input_files, batch_size, lambda_reg, md_filedir, in_node_feats, hidden_dim, out_node_feats, activation, 
    encoding_size, out_feats):
    print("Loading input files: ", num_input_files)
    dataset = MDDataset(md_filedir, rotation_aug, avg_num_neighbors, train_data_fraction, return_train_data) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    print("Dataloader initialized.")   

    phi_dst = nn.Linear(in_node_feats, hidden_dim)
    phi_edge = nn.Linear(in_node_feats, hidden_dim)
    phi = MLP(hidden_dim, out_node_feats,
                    activation_first=True, hidden_layer=1, hidden_dim=hidden_dim, activation=activation)

    force_decoder = MLP(encoding_size, out_feats, hidden_layer=2, hidden_dim=hidden_dim, activation='gelu')
    srgnn_loss = SRGNNLoss(lambda_reg)
    return dataloader, srgnn_loss, phi, phi_dst, phi_edge, force_decoder

def forward() 
    e1 = 
    e2 = 
    e3 = 
    n1 = 
    n2 = 
    n3 = 
    aggregate_edge_messages = 
    node_embeddings = 

    node_feat = phi(phi_dst(h) + phi_edge(edge_emb)) # So, we need to predict edge_emb (aggrgated edge messages), h (input node embeddings) inputs for the last message passing layer using SR to effecitively couple SR with the remaining GNN.  
    force_pred = force_decoder(x)    
    return force_pred








######################################## Control script #####################################################
train_data_fraction, avg_num_neighbors, rotation_aug, return_train_data,
num_input_files, batch_size, lambda_reg, md_filedir, in_node_feats, hidden_dim, out_node_feats, activation, 
encoding_size, out_feats    # TODO
dataloader, srgnn_loss, phi, phi_dst, phi_edge, force_decoder = init_dataloader_and_models()
for epoch_id in range(n_epochs):
    for iter_id, pos, edge_index_list, force_gt in enumerate(dataloader):
        force_pred = forward(pos, edge_index_list)
        sr_loss.backward()
        if iter_id % 100 == 0:
            evaluate(force_gt, force_pred)

