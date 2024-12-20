'''
    1. Load ID, OD datasets using separate dataloaders
    2. Load SR and GNN models
    3. Call eval on predictions from SR and GNN
    4. Report results
'''

from monolithic_gamd_impl import evaluate, custom_collate, MDDataset
from sr_over_gnn_messages import SimpleMDNetNew
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

# check for CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is: ", device)

def load_id_dataset():
    data_dir = '../top/'#"./id_dataset/" # [10000, [[258, 3], [258, 3], [258, 3]]]
    train_data_fraction = 0.9 # select 9k for training
    avg_num_neighbors = 20 # criteria for connectivity of atoms for any frame
    rotation_aug = False # online rotation augmentation for a frame
    batch_size = 1 # number of graphs in a batch
    # create train data-loader
    return_train_data = True  
    dataset = MDDataset(data_dir, rotation_aug, avg_num_neighbors, train_data_fraction, return_train_data) 
    id_dataloader = DataLoader(dataset, num_workers = 0, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)#, num_workers = os.cpu_count()) # create a batched graph and return    
    
    return id_dataloader

def load_ood_dataset():
    data_dir = '../top/' #"./ood_dataset/" # [10000, [[258, 3], [258, 3], [258, 3]]]
    train_data_fraction = 0.9 # select 9k for training
    avg_num_neighbors = 20 # criteria for connectivity of atoms for any frame
    rotation_aug = False # online rotation augmentation for a frame
    batch_size = 1 # number of graphs in a batch
    # create train data-loader
    return_train_data = True  
    dataset = MDDataset(data_dir, rotation_aug, avg_num_neighbors, train_data_fraction, return_train_data) 
    ood_dataloader = DataLoader(dataset, num_workers = 0, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)#, num_workers = os.cpu_count()) # create a batched graph and return    

    return ood_dataloader

def load_gnn():
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
    gamdnet_official_model_checkpoint_filename = 'epoch=39-step=360000_edge_msg_constrained_std.ckpt'
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

    return gamdnet_official



class SRModel(object):
    def __init__(self, msg_models, netforce_models):
        self.msg_models = msg_models
        self.netforce_models = netforce_models
    
    def predict(self, pos, edge_index_list):

        center_node_idx = edge_index_list[0, :]
        neigh_node_idx = edge_index_list[1, :]
        neigh_node_pos = pos[neigh_node_idx]
        center_node_pos = pos[center_node_idx]

        # Identify edges where center and neighbor nodes are the same
        same_index_columns = (center_node_idx == neigh_node_idx)
        # Get the self-loop edge indices
        self_loop_indices = edge_index_list[:, same_index_columns]

        # Print the self-loop edge indices
        print("Self-loop Edge Indices:\n", self_loop_indices)

        #exit(0)

        # Remove edges where center and neighbor nodes are the same
        center_node_pos = center_node_pos[~same_index_columns]
        neigh_node_pos = neigh_node_pos[~same_index_columns]

        center_node_idx = center_node_idx[~same_index_columns]
        neigh_node_idx = neigh_node_idx[~same_index_columns]

        # Calculate the distance vector
        r_vec = neigh_node_pos - center_node_pos  # Shape: [n, 3]        
        # Calculate the distance (magnitude)
        r = torch.norm(r_vec, dim=1).unsqueeze(1)  # Shape: [n, 1]
        dx = r_vec[:, 0]
        dy = r_vec[:, 1]
        dz = r_vec[:, 2]
        # Create a DataFrame for easier handling of data
        data = pd.DataFrame({
            'dx': dx.squeeze().cpu(),
            'dy': dy.squeeze().cpu(),
            'dz': dz.squeeze().cpu(),
            'r': r.squeeze().cpu(),
        })        
        # Define the features
        X = data[['dx', 'dy', 'dz', 'r']].values
        e1 = torch.Tensor(self.msg_models[0].predict(X)).squeeze()
        e2 = torch.Tensor(self.msg_models[1].predict(X)).squeeze()
        e3 = torch.Tensor(self.msg_models[2].predict(X)).squeeze()
  
        edge_messages = torch.stack((e1, e2, e3), dim=1)


        aggregate_edge_messages = torch.zeros((pos.shape[0], edge_messages.shape[1]), dtype=torch.float32)
        aggregate_edge_messages.index_add_(0, center_node_idx.cpu(), edge_messages)  # Accumulate messages into AGG    
        
        
        agg_msg_comp_std = torch.std(aggregate_edge_messages, axis = 0)
        agg_msg_comp_most_imp_indices = torch.argsort(agg_msg_comp_std)[-3:] # in ascending order
        agg_msg_most_imp = aggregate_edge_messages[:, agg_msg_comp_most_imp_indices]
        
        agg_comp_1 = agg_msg_most_imp[:, -1]
        agg_comp_2 = agg_msg_most_imp[:, -2]
        agg_comp_3 = agg_msg_most_imp[:, -3]

        x = pos[:, 0].cpu()
        y = pos[:, 1].cpu()
        z = pos[:, 2].cpu()    

        data = pd.DataFrame({
            'x': x.squeeze(),
            'y': y.squeeze(),
            'z': z.squeeze(),        
            'agg_comp_1': agg_comp_1.squeeze(),
            'agg_comp_2': agg_comp_2.squeeze(),
            'agg_comp_3': agg_comp_3.squeeze(),
        })
        
        # Define the features
        X = data[['x', 'y', 'z', 'agg_comp_1', 'agg_comp_2', 'agg_comp_3']].values       

        f1 = torch.Tensor(self.netforce_models[0].predict(X)).squeeze()
        f2 = torch.Tensor(self.netforce_models[1].predict(X)).squeeze()
        f3 = torch.Tensor(self.netforce_models[2].predict(X)).squeeze()
        
        force_pred_sr = torch.stack((f1, f2, f3), dim=1).cuda()
        return force_pred_sr



def load_sr():
    import pickle
    # Load the model from the file
    sr_msg_models = []
    sr_netforce_models = []
    for comp_id in range(3): # 1 model file per component
        model_filename = f"pysr_model_msg_{comp_id}_pred.pkl"
        with open(model_filename, 'rb') as file:
            sr_msg_models.append(pickle.load(file))    
        model_filename = f"pysr_model_netforce_{comp_id}_pred.pkl"
        with open(model_filename, 'rb') as file:
            sr_netforce_models.append(pickle.load(file))                
    
    sr_model = SRModel(sr_msg_models, sr_netforce_models)
    return sr_model



id_dataloader = load_id_dataset()
ood_dataloader = load_ood_dataset()

gnn_model = load_gnn()
sr_model = load_sr()


print("Testing in-domain generalization performance...")
for pos, edge_index_list, force_gt in id_dataloader:
    print("GNN performance: ")
    force_pred_gnn = gnn_model([pos],
                               [edge_index_list])
    evaluate(force_pred_gnn, force_gt)
    print("SR perfomance: ")
    force_pred_sr = sr_model.predict(pos, edge_index_list)    
    evaluate(force_pred_sr, force_gt)
    break


print("Testing out of domain generalization performance...")
for pos, edge_index_list, force_gt in ood_dataloader:
    print("GNN performance: ")    
    force_pred_gnn = gnn_model([pos],
                               [edge_index_list])
    evaluate(force_pred_gnn, force_gt)                               
    print("SR perfomance: ")
    force_pred_sr = sr_model.predict(pos, edge_index_list)    
    evaluate(force_pred_sr, force_gt)
    break    


