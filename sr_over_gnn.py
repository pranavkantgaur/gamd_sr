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
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from monolithic_gamd_impl import *



def are_aggregate_edge_msgs_gt_force_correlated(msg_force_dict):
    '''
    msg_force_dict: {'aggregate_edge_messages': [total_edges, emb_dim], 'gt_force': [num_particles * batch_size, 3]}
    '''

    # Calculate variance for each component of agg_msg across all samples
    msg_comp_variances = torch.var(msg_force_dict['aggregate_edge_messages'], axis=0)  # Variance for each component

    # Step 3: Get top-3 indices based on variance
    top_var_indices = torch.argsort(msg_comp_variances)[-3:]  # Get indices of top-3 components with maximum variance

    # Output the results of top-3 components
    print("Top-3 components with maximum variance:")
    for index in top_var_indices:
        print(f"Component Index: {index}, Variance: {top_var_indices[index]}")

    # Step 4: Prepare data for linear regression using top-3 components as output variables
    agg_msg_selected_values = msg_force_dict['aggregate_edge_messages'][:, top_indices]  # Select only the top-3 components

    # Step 5: Performing linear regression for each selected agg_msg component against gt_force
    r2_threshold = 0.7  # Set your threshold here

    for component_index in range(3):
        model = LinearRegression()
        
        # Fit model to predict selected agg_msg component based on gt_force components
        model.fit(msg_force_dict['force_gt'], agg_msg_selected_values[:, component_index])
        
        slope = model.coef_
        intercept = model.intercept_
        
        # Calculate R^2 score
        predictions = model.predict(msg_force_dict['force_gt'])
        r2 = r2_score(msg_force_dict['aggregate_edge_messages'][:, component_index], predictions)
        
        print(f"\nLinear fit results for Component {component_index + 1}:")
        print(f"Slope: {slope}, Intercept: {intercept}, R^2 Score: {r2}")

        # Check goodness of fit against threshold
        if r2 > r2_threshold:
            are_correlated = True 
        else:
            are_correlated = False
            break
    return are_correlated

# add updated forward for MPBlock class
def new_forward(self, node_embeddings, edge_embeddings, edge_index_list): 
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
    self.aggregated_edge_messages = aggregated_edge_messages_torch_vectorized        
    node_embeddings = self.theta(node_embeddings + aggregated_edge_messages_torch_vectorized)  
    return node_embeddings


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
    num_input_files = len(os.listdir(md_filedir))
    batch_size = num_input_files # number of graphs in a batch
    print("Loading input files: ", num_input_files)
    dataset = MDDataset(md_filedir, rotation_aug, avg_num_neighbors, train_data_fraction, return_train_data) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    print("Dataloader initialized.")    
    
    return gamdnet, dataloader

#print("Result: ", are_aggregate_edge_msgs_gt_force_correlated())


def get_msg_force_dict(gamdnet, dataloader):
    msg_force_dict = None
    # run inference over the input batched graph from dataloader
    # record aggregate edge messages and force ground truths for each node in output dictionary
    for pos, edge_index_list, force_gt in dataloader:
        with torch.no_grad():  # Disable gradient calculation for inference
            gamdnet.mpnn.mpblock[3].forward = new_forward # monkey-patch for saving aggregate edge messages
            model_output = gamdnet(pos, edge_index_list)  # Forward pass through the model
            msg_force_dict['aggregagte_edge_messages'] = gamdnet.mpnn.mpblock[3].aggregate_edge_messages # [total_edges, emb_dim]
            msg_force_dict['force_gt'] = force_gt # [num_particle * batch_size, 3]
    return msg_force_dict





model_weights_filename = 'best_model_vectorized_message_passing.pt'
md_filedir = 'sr_inputs'
gamdnet, dataloader = load_model_and_dataset(model_weights_filename, md_filedir)
#msg_force_dict = get_msg_force_dict(gamdnetm, dataloader)

#print("Result: ", are_aggregate_edge_msgs_gt_force_correlated(msg_force_dict))
