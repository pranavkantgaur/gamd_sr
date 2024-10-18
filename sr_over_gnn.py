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



def are_aggregate_edge_msgs_gt_force_correlated(gamdnet = None, dataloader = None, msg_force_dict = None):
    '''
    gamdnet: pytorch model
    dataloader: pytorch dataloader
    msg_force_dict: [filename, node_id, aggregate_message, gt_force]
    '''   

    # result variable
    are_correlated = False
    

    # Example dictionary structure with nested access
    msg_force_dict = {
        'file1': {
            '220': {
                'agg_msg': torch.randn(1, 128),  # Example tensor
                'gt_force': torch.randn(1, 3)     # Example tensor
            }
        },
        'file2': {
            '221': {
                'agg_msg': torch.randn(1, 128),
                'gt_force': torch.randn(1, 3)
            }
        },
        # Add more entries as needed
    }

    # Step 1: Prepare lists to collect aggregate_message and gt_force values
    all_agg_msg_values = []
    all_gt_force_values = []

    # Collecting data for variance calculation and regression
    for filename, nodes in msg_force_dict.items():
        for node_id, data in nodes.items():
            aggregate_message = data['agg_msg']
            gt_force = data['gt_force']

            # Append the aggregate message tensor to the list
            all_agg_msg_values.append(aggregate_message.flatten().numpy())  # Convert to numpy array
            all_gt_force_values.append(gt_force.flatten().numpy())          # Convert to numpy array

    # Convert lists to numpy arrays for further processing
    all_agg_msg_values = np.array(all_agg_msg_values)  # Shape: (num_samples, 128)
    all_gt_force_values = np.array(all_gt_force_values)  # Shape: (num_samples, 3)

    # Step 2: Calculate variance for each component of agg_msg across all samples
    variances = np.var(all_agg_msg_values, axis=0)  # Variance for each component

    # Step 3: Get top-3 indices based on variance
    top_indices = np.argsort(variances)[-3:]  # Get indices of top-3 components with maximum variance

    # Output the results of top-3 components
    print("Top-3 components with maximum variance:")
    for index in top_indices:
        print(f"Component Index: {index}, Variance: {variances[index]}")

    # Step 4: Prepare data for linear regression using top-3 components as output variables
    agg_msg_selected_values = all_agg_msg_values[:, top_indices]  # Select only the top-3 components

    # Step 5: Performing linear regression for each selected agg_msg component against gt_force
    r2_threshold = 0.7  # Set your threshold here

    for component_index in range(3):
        model = LinearRegression()
        
        # Fit model to predict selected agg_msg component based on gt_force components
        model.fit(all_gt_force_values, agg_msg_selected_values[:, component_index])
        
        slope = model.coef_
        intercept = model.intercept_
        
        # Calculate R^2 score
        predictions = model.predict(all_gt_force_values)
        r2 = r2_score(agg_msg_selected_values[:, component_index], predictions)
        
        print(f"\nLinear fit results for Component {component_index + 1}:")
        print(f"Slope: {slope}, Intercept: {intercept}, R^2 Score: {r2}")

    # Check goodness of fit against threshold
    if r2 > r2_threshold:
        are_correlated = True 

    return are_correlated


from monolithic_gamd_impl import *

def load_model_and_dataset(model_filename, md_filedir):
    '''
    Load model and MD dataset for SR from input filename and dataset directory.
    '''
    gamdnet = 
    num_input_files = len(os.listdir(md_filedir))
    dataset = MDDataset(data_dir, rotation_aug, avg_num_neighbors, train_data_fraction, return_train_data, val_idx) 
    dataloader = DataLoader(dataset, batch_size=num_input_files, shuffle=False, collate_fn=custom_collate)
    
    
    dataloader = 
    return gamdnet, dataloader

#print("Result: ", are_aggregate_edge_msgs_gt_force_correlated())


model_weights_filename = 'best_model_vectorized_message_passing.pt'
md_filedir = 'sr_inputs'
gamdnet, dataloader = load_model_and_dataset(model_weights_filename, md_filedir)
print("Result: ", are_aggregate_edge_msgs_gt_force_correlated(gamdnet, dataloader))
