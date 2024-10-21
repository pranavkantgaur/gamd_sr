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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monolithic_gamd_impl import MDDataset, custom_collate, evaluate

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
    are_correlated = False
    return are_correlated


def load_model_and_dataset(model_filename, md_filedir):
    '''
    Load model and MD dataset for SR from input filename and dataset directory.
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
    checkpoint = torch.load(model_filename)
    gamdnet.load_state_dict(checkpoint['model_state_dict'])
    # Set the model to evaluation mode
    gamdnet.eval()
    
    print("Model weights loaded successfully.")
    
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
    
    return gamdnet, dataloader

#print("Result: ", are_aggregate_edge_msgs_gt_force_correlated())


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
    
    epsilon = 1.0
    sigma = 1.0

    force_magnitude = 24 * epsilon * (
        2 * (sigma ** 12) / (r ** 13) - 
        (sigma ** 6) / (r ** 7)
    )  # Shape: [n, 1]

    # Calculate the force vector (directed)
    force_vector = force_magnitude * (r_vec / r)  # Shape: [n, 3]

    return force_vector







def get_msg_force_dict(gamdnet, dataloader):
    msg_force_dict = {}
    # run inference over the input batched graph from dataloader
    # record aggregate edge messages and force ground truths for each node in output dictionary
    for pos, edge_index_list, force_gt in dataloader:
        with torch.no_grad():  # Disable gradient calculation for inference            
            force_pred = gamdnet(pos, edge_index_list)  # Forward pass through the model
            evaluate(force_gt, force_pred)
            
            # record messages for SR
            lj_force = compute_lj_force(pos, edge_index_list)
            msg_force_dict['edge_messages'] = gamdnet.mpnn_mlps.mp_blocks[3].edge_message_neigh_center
            msg_force_dict['force_gt'] = lj_force # [num_particle * batch_size, 3]
                        
        break # run dataloader only once
    return msg_force_dict




model_weights_filename = 'best_model_vectorized_message_passing.pt'
md_filedir = '../top/'
gamdnet, dataloader = load_model_and_dataset(model_weights_filename, md_filedir)
msg_force_dict = get_msg_force_dict(gamdnet, dataloader)

print("Result: ", are_edge_msgs_gt_force_correlated(msg_force_dict))
