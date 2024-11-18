'''
    1. Load ID, OD datasets using separate dataloaders
    2. Load SR and GNN models
    3. Call eval on predictions from SR and GNN
    4. Report results
'''

from monolithic_gamd_impl import evaluate, custom_collate, MDDataset
from sr_over_gnn_messages import SimpleMDNetNew

def load_id_dataset():
    data_dir = "./id_dataset/" # [10000, [[258, 3], [258, 3], [258, 3]]]
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
    data_dir = "./ood_dataset/" # [10000, [[258, 3], [258, 3], [258, 3]]]
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
    gamd_model_weights_filename = 'epoch=39-step=360000_edge_msg_constrained_std.ckpt'
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
    def __init__(self, msg_model, force_model):
        self.msg_model = msg_model
        self.force_model = force_model
    
    def predict(pos, edge_index_list):
        dx = 
        dy = 
        dz = 
        r = 
        X = pd.DataFrame('dx': dx, 'dy': dy, 'dz': dz, 'r': r)
        e1 = self.msg_comp_1_model.predict(X)
        e2 = self.msg_comp_2_model.predict(X)
        e3 = self.msg_comp_3_model.predict(X)

        
        agg_comp_1, agg_comp_2, agg_comp_3 = compute_aggregate_msgs(e1, e2, e3)
        pos_X = 
        pos_Y = 
        pos_Z = 
        
        X = pd.DataFrame('pos_X': , 'pos_Y': , 'pos_Z': , 'agg_comp_1: ', 'agg_comp_2': , 'agg_comp_3': )
        
        f1 = self.force_comp_1_model.predict(X)
        f2 = self.force_comp_2_model.predict(X)
        f3 = self.force_comp_3_model.predict(X)
        
        force_pred_sr = torch.cat((f1, f2), dim = 1)
        force_pred_sr = torch.cat((force_pred_sr, f3), dim=1)
        return force_pred_sr

def load_sr():
    import pickle
    # Load the model from the file
    with open('pysr_model_msg_pred.pkl', 'rb') as file:
        sr_model_msg_pred = pickle.load(file)    
    with open('pysr_model_msg_pred.pkl', 'rb') as file:
        sr_model_msg_pred = pickle.load(file)        
    
     
    
    sr_model = SRModel(sr_model_msg_pred, sr_model_force_pred)
    return sr_model


def sr_forward(pos, sr_model_msg_pred, sr_model_force_pred):


id_dataloader = load_id_dataset()
ood_dataloader = load_ood_dataset()

gnn_model = load_gnn()
sr_model = load_sr()

for pos, edge_index_list, force_gt in id_dataloader:
    force_pred_gnn = gnn_model(pos, edge_index_list)
    force_pred_sr = sr_model(pos, edge_index_list)
    evaluate(force_pred_gnn, force_gt)
    evaluate(force_pred_sr, force_gt)
    break


