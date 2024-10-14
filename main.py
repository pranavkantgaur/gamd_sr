'''
As per paper:
forward: feature encoding -> message passing -> embedding decoding = force prediction

GAMD uses force prediction to update the particle positions, velocities for next time-step.

MD using GNN = force_prediction using GNN forward -> Updation of particle position and velocity using force prediction -> Repeat 

===

Compute node and edge embeddings for each node using equation 1
Compute edge messages for every atom using equation 2 (it is edge message coming out of lth message passing block)
'''

import os
default_n_threads = 8 # per https://stackoverflow.com/a/77609017/985166
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import jax
from jax import vmap
import jax.numpy as jnp 
import jax_md
from jax_md import partition, space
import random
from sklearn.cluster import KMeans   
import cupy  
from sklearn.preprocessing import StandardScaler  
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import triton
import triton.language as tl

# check for CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is: ", device)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# Set the seed
seed = 10

# PyTorch seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Python random seed
random.seed(seed)

class MDDataset(Dataset):
  def __init__(self, data_dir, rotation_aug=False, avg_num_neighbors = 20, train_data_fraction = 0.9, return_train_data = False, val_idx = None):
    self.data_dir = data_dir

    # randomly sample 90% files from the dir
    original_data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
   
    idxs = np.arange(len(original_data_files))
    np.random.seed(0)   # fix same random seed
    np.random.shuffle(idxs)
    ratio = train_data_fraction
    if return_train_data:
      self.idx = idxs[:int(len(idxs)*ratio)]
      self.val_idx = idxs[int(len(idxs)*ratio):]
    else:
      if len(val_idx):
        self.idx = val_idx
      else:
        print("Please provide validation file indices as input")
        exit(0)
    self.rotation_aug = rotation_aug
    self.avg_num_neighbors = avg_num_neighbors
    
    self.force_scaler = StandardScaler()
    self.num_particles = 258  
    self.cutoff_distance = 7.5 # as per https://github.com/BaratiLab/GAMD/blob/main/code/LJ/train_network_lj.py#L26
    BOX_SIZE = 27.27
    self.simulation_box = jnp.array([BOX_SIZE, BOX_SIZE, BOX_SIZE])
    print("Simulation box size based on MD simulation data is: ", self.simulation_box)
    self.displacement_function, _ = jax_md.space.periodic(self.simulation_box)
    self.neighbor_list_fn = jax_md.partition.neighbor_list(displacement_or_metric = self.displacement_function, \
    box = self.simulation_box, r_cutoff = self.cutoff_distance, dr_threshold= self.cutoff_distance / 6.,
                                                       mask_self=False)  
    self.neighbor_list_fn_jit = jax.jit(self.neighbor_list_fn) # to be used for neighborlist updates.                                                           
    self.neighborlist_has_been_init = False
    self.last_neighbor_list_obj = None                                                       
    
  def get_val_data_idx(self):
    return self.val_idx # for validation 

  def __len__(self):    
    return len(self.idx)
  
  def masking_fn(self, pos: jnp.ndarray, neigh_idx: jnp.ndarray):
    # notice here, pos must be jax numpy array, otherwise fancy indexing will fail
    d = partial(self.displacement_function)
    d = space.map_neighbor(d)
    pos_neigh = pos[neigh_idx]
    dR = d(pos, pos_neigh)
    dr_2 = space.square_distance(dR)  
    mask = jnp.logical_and(neigh_idx != self.num_particles, dr_2 < self.cutoff_distance ** 2)
    return mask
  
  def get_edge_idx(self, nbrs, pos_jax, mask):
      dummy_center_idx = nbrs.idx.copy()
      #Instead of ops.index_update(x, idx, vals) you should use x.at[idx].set(vals).
      dummy_center_idx = dummy_center_idx.at[:].set(jnp.arange(pos_jax.shape[0]).reshape(-1, 1))
      #jax.ops.index_update(dummy_center_idx, None, jnp.arange(pos_jax.shape[0]).reshape(-1, 1))
      center_idx = dummy_center_idx.reshape(-1)
      center_idx_ = cupy.asarray(center_idx)
      center_idx_tsr = torch.as_tensor(center_idx_, device='cuda')

      neigh_idx = nbrs.idx.reshape(-1)

      # cast jax device array to cupy array so that it can be transferred to torch
      neigh_idx = cupy.asarray(neigh_idx)
      mask = cupy.asarray(mask)
      mask = torch.as_tensor(mask, device='cuda')
      flat_mask = mask.view(-1)
      neigh_idx_tsr = torch.as_tensor(neigh_idx, device='cuda')

      edge_idx_tsr = torch.cat((center_idx_tsr[flat_mask].view(1, -1), neigh_idx_tsr[flat_mask].view(1, -1)),
                                dim=0)
      return edge_idx_tsr


  def _get_edge_index_using_jax_md(self, pos):   
    # get neighbors list for each node 
    # get rad cutoff   
    pos = jnp.array(pos) # TODO: use torch2jax for converting torch tensor to jax tensor 
    
    if self.cutoff_distance > self.simulation_box[0]:
      print("BUG: SELECTED CUTOFF DISTANCE IS BEYOND THE BOX.")
      exit(0)
    else:
      pass
    
    # create edge index list , [num_edges, 2]
    if self.neighborlist_has_been_init == False:
      self.last_neighbor_list_obj = self.neighbor_list_fn.allocate(pos) # [num nodes, num of neighbors for that node]    
      self.neighborlist_has_been_init = True
      #print("INFO: INIT CALLED")
    else:
      self.last_neighbor_list_obj = self.neighbor_list_fn_jit.update(pos, self.last_neighbor_list_obj)
      #print("INFO: Update called")
      if self.last_neighbor_list_obj.did_buffer_overflow:
        self.last_neighbor_list_obj = self.neighbor_list_fn.allocate(pos)      
    neighbor_indices = self.last_neighbor_list_obj.idx
    edge_mask_all = self.masking_fn(pos, neighbor_indices)
    edge_index = self.get_edge_idx(self.last_neighbor_list_obj, pos, edge_mask_all).long()

    return edge_index
  

  def center_positions(self, pos):
    offset = torch.mean(pos, axis=0)
    return pos - offset, offset

  def __getitem__(self, idx):
    get_item_start_time = time.time()
    idx = self.idx[idx]
    sample_num = 1000
    sample_to_read = idx % sample_num
    seed = idx // sample_num
    # TODO: DEBUG: SET FOR TRACING EXECUTION FOR THIS SAMPLE. TO BE REMOVED.
    #seed = 5
    #sample_to_read = 611
    
    data_file = f'ljdata_{seed}_{sample_to_read}.npz'
    print(f"Reading {data_file}...")
    data = np.load(os.path.join(self.data_dir, data_file))

    # Extract and convert data

    pos = torch.from_numpy(np.mod(data['pos'].astype(np.float32), self.simulation_box[0])).float() # apply boundary condition.    

    b_pnum, dims = data['forces'].shape
    force_flat = data['forces'].reshape((-1, 1))
    self.force_scaler.partial_fit(force_flat)
    force = torch.from_numpy(self.force_scaler.transform(force_flat)).float().view(b_pnum, dims)    

    # Apply random rotation augmentation (if enabled)    
    if self.rotation_aug:
      # Generate random rotation matrix
      rotation_matrix = self.generate_random_rotation().float()      
      # center positions: https://github.com/BaratiLab/GAMD/blob/main/code/LJ/train_network_lj.py#L210
      pos, off = self.center_positions(pos)
      pos = torch.einsum("bi,ij->bj", pos, rotation_matrix)      
      pos += off
      force = torch.einsum("bi,ij->bj", force, rotation_matrix)  # Apply rotation to force as well      

    # apply random jittering to positions: https://github.com/BaratiLab/GAMD/blob/main/code/LJ/train_network_lj.py#L228       
    #pos = pos + torch.randn_like(pos) * 0.005
    edge_start_time = time.time()
    edge_index = self._get_edge_index_using_jax_md(pos)    
    edge_end_time = time.time()
    get_item_end_time = time.time()
    #print(f"Get item time: {get_item_end_time - get_item_start_time}")
    #print(f"Edge index time: {edge_end_time - edge_start_time}")
    return pos, edge_index, force


  def generate_random_rotation(self):
    # Implement your logic here to generate a random rotation matrix
    # This example uses a random axis and a small random angle
    random_axis = torch.randn(3)  # Random unit tensor (1D) of shape [] with 3 elements
    random_angle = torch.rand(1) * 0.1  # Small random angle (adjust as needed)

    # Use Rodrigues' rotation formula to calculate rotation matrix
    norm = random_axis.norm() # L2 norm, single scalar, 0-D tensor
    if norm < 1e-6:
        return torch.eye(3)  # Handle case where random_axis is too small, 3x3 identity matrix

    random_axis = random_axis / norm # 1D tensor of 3 elements / scalar = scaled 1D tensor of 3 elements
    K = torch.tensor([[0, -random_axis[2], random_axis[1]],
                      [random_axis[2], 0, -random_axis[0]],
                      [-random_axis[1], random_axis[0], 0]]) # rotation matrix using random axis and angle

    rotation_matrix = torch.eye(3) + torch.sin(random_angle) * K + (1 - torch.cos(random_angle)) * torch.matmul(K, K) # [3,3] shaped tensor
    return rotation_matrix # [3, 3] shaped tensor

@triton.jit
def _propogate_messages(src_node_embeddings_ptr, edge_embeddings_ptr,
    dest_node_messages_ptr,
    adjacency_list_ptr, 
    num_edges,
    BLOCK_SIZE: tl.constexpr      
  ):
  # Get the unique thread ID
  tid = tl.program_id(0)

  if tid < num_edges:
      # Load destination node index from adjacency list (0th row)
      dest_node_idx = tl.load(adjacency_list_ptr + tid)  # Use load instead of direct indexing

      # Load edge embedding
      edge_embedding = tl.load(edge_embeddings_ptr + tid)  
      
      # Load source node embedding for this edge
      src_embedding = tl.load(src_node_embeddings_ptr + tid) #+ src_node_idx)  # Use src_node_idx for correct loading
      
      # Compute the message as the product of source node embedding and edge embedding
      message = src_embedding * edge_embedding
      
      # Aggregate the message into destination node messages using atomic addition
      tl.atomic_add(dest_node_messages_ptr + dest_node_idx, message)

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
        node_embeddings = self.layer_norm(node_embeddings) # [b * num_nodes, embed_dim]     
        center_node_index = edge_index_list[0, :] # [1, e] 
        neighbor_node_index = edge_index_list[1, :] # [1, e]
        center_node_embeddings = node_embeddings[center_node_index]
        neighbor_node_embeddings = node_embeddings[neighbor_node_index]
        sum_tensor = edge_embeddings + center_node_embeddings + neighbor_node_embeddings
        theta_edge = self.phi(sum_tensor)        
        
        
        # Launching the kernel
        block_size = 256  # Define block size based on your GPU architecture
        num_edges = theta_edge.shape[0]
        grid_size = (num_edges + block_size - 1) // block_size        
        aggregated_edge_messages = torch.zeros((node_embeddings.shape[0], theta_edge.shape[1]), dtype=torch.float32).cuda()
        _propogate_messages[(grid_size,)](neighbor_node_embeddings, theta_edge, 
        aggregated_edge_messages, edge_index_list, num_edges, BLOCK_SIZE=block_size) # calls the message passing kernel
        
        '''
        edge_message_neigh_center = (neighbor_node_embeddings * theta_edge)#.to(torch.float16)  # Convert to float16        

        # Collect edge messages received by each node
        node_edge_messages = [[torch.zeros(node_embeddings.shape[1]).to(device)] for _ in range(node_embeddings.shape[0])]  # Use float16
        
        
        for edge_id, edge_message in enumerate(edge_message_neigh_center):  # edge_message is already float16
            center_node_idx = edge_index_list[0, edge_id]
            
            if center_node_idx >= len(node_edge_messages):  # Fixed the condition to use >= instead of >
                print(f"BUG: {center_node_idx} >= {len(node_edge_messages)}")   
                exit(0)           
            
            node_edge_messages[center_node_idx].append(edge_message)  # edge_message is float16

        # Aggregate edge messages for each node
        aggregated_edge_messages = torch.cat(
            [torch.sum(torch.stack(row), dim=0).unsqueeze(0) for row in node_edge_messages],
            dim=0
        )#.to(torch.float16)  # Ensure the result is float16
        '''
        #print(f"INFO: theta edge before PHI: {theta_edge[150]}")
        #print(f"INFO: edge embeddings before PHI: {edge_embeddings[150]}")
        #print(f"INFO before PHI: node embeddings: {node_embeddings[150]}, aggregated edge mesage: {aggregated_edge_messages[150]}")
        #print("Edge index list before PHI: ", edge_index_list.shape, edge_index_list[0, 350], edge_index_list[1, 350])
        node_embeddings = self.theta(node_embeddings + aggregated_edge_messages)  
        return node_embeddings


class MPNN(nn.Module):
    def __init__(self, num_mpnn_layers, embed_dim, hidden_dim):
        super(MPNN, self).__init__()
        self.num_mpnn_layers = num_mpnn_layers
        self.mp_blocks = nn.Sequential(*[MPBlock(embed_dim, hidden_dim) for _ in range(num_mpnn_layers)]) # list of length num_mpnn_layers of MPBlock objects.
        

    def forward(self, node_embeddings, edge_embeddings, edge_index_list): # [b * num_nodes, embed_dim], [e, embed_dim], [e, 2]                
        #print(f"INFO: node_embeddings received: {node_embeddings[100]}, edge embeddings: {edge_embeddings[100]}")
        #print(f"INFO: edge list received: {edge_index_list[0][100]}, {edge_index_list[1][100]}")          
        for layer in self.mp_blocks:
          node_embeddings_initial = node_embeddings.clone()
          node_embeddings = layer(node_embeddings, edge_embeddings, edge_index_list) + node_embeddings_initial  # residual connection
          #print(f"INFO: node_embeddings after 1st MP: {node_embeddings[100]}, edge embeddings: {edge_embeddings[100]}")
          #print(f"INFO: edge list after 1st MP: {edge_index_list[0][100]}, {edge_index_list[1][100]}")  
          #exit(0)            
        #print("MPNN forward completed.")
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
        #print(f"INFO: edge features: {edge_features[150]}")        
        edge_embeddings = self.edge_embedding_mlp(edge_features.float()) # weights, mlp, [#e, num_rbfs + num_edge_types] -> [#e, embed_dim]
        #print(f"INFO: edge embeddings: {edge_embeddings[150]}")        
        edge_embeddings = self.edge_layer_norm(edge_embeddings)
        #print(f"INFO: edge layer norm: {edge_embeddings[150]}")        
        num_nodes = pos.shape[0]                
        node_embeddings = self.node_embeddings.repeat((num_nodes, 1)).to(device)
        #print(f"INFO: node_embeddings: {node_embeddings[100]}, edge embeddings: {edge_embeddings[100]}, edge list: {edge_index_list[:, 100]}")
        
        node_embeddings = self.mpnn_mlps(node_embeddings, edge_embeddings, edge_index_list) # L blocks of weights (L sequential MLPs),         
        node_forces = self.force_pred_mlp(node_embeddings) # weights, mlp, [b * num_nodes, embed_dim] -> [b * num_nodes, 3]
        
        return node_forces
        
def configure_optimizer_schedular(model, num_epochs):
    # Set the initial and final learning rates
    initial_lr = 3e-4
    # Create the Adam optimizer with the learning rate function
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)       
    # Create the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.001**(5/num_epochs))
    return optimizer, scheduler

class GAMDLoss(nn.Module):
    def __init__(self, lambda_reg):
        super(GAMDLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, predictions, ground_truth):
        mae = nn.L1Loss()(predictions, ground_truth)
        #magnitude = torch.abs(torch.mean(predictions))
        #loss = mae + self.lambda_reg * magnitude
        loss = mae
        #print("Pred is: ", predictions)
        #print("GT is: ", ground_truth[0])
        #print("prediction: ", predictions[150])
        #print("GT: ", ground_truth[150])
        print("Loss is: ", loss.item())
        #exit(0)
        return loss


def custom_collate(batch):
  # creates a batched graph over a batch of graphs
  #print("Dataloader called custom collate.")
  pos_batch = []
  edge_index_list_batch = []
  force_batch = []
  for sample_id, (pos, edge_index_list, force) in enumerate(batch):
    pos_batch.append(pos)    
    # increment the edge indices to point to correct atom ids in the batched graph    
    edge_index_list_batch.append(edge_index_list + sample_id * pos.shape[0])         
    force_batch.append(force)  
  pos_batch = torch.cat(pos_batch, dim=0).to(device)
  edge_index_list_batch = torch.cat(edge_index_list_batch, dim=1).to(device)
  force_batch = torch.cat(force_batch, dim=0).to(device)  
  
  return pos_batch, edge_index_list_batch, force_batch

      
def print_model_summary(gamdnet, summary_writer):
  from torch_geometric.nn import summary
  # Create dummy input tensors based on your input shapes
  node_features = torch.randn(258, 3).to(device)  # Shape: [b * 258, 3]
  edge_indices = torch.randint(100, size=(2, 20)).to(device)  # Shape: [e, 2]

  # Print the model summary
  print(summary(gamdnet, node_features, edge_indices, max_depth = 5))  

  summary_writer.add_graph(gamdnet, (node_features, edge_indices))

# Calculate MAPE
def evaluate(y_true, y_pred):
    
  # Avg. cosine similarity
  # Step 1: Compute the dot product
  dot_product = torch.sum(y_true * y_pred, dim=1)

  # Step 2: Compute the magnitudes
  magnitude_F = torch.norm(y_true, dim=1)
  magnitude_F_prime = torch.norm(y_pred, dim=1)

  # Step 3: Compute cosine similarity
  cosine_similarity = dot_product / (magnitude_F * magnitude_F_prime)

  # Step 4: Compute the average value of cosine similarity
  average_cosine_similarity = torch.mean(cosine_similarity)

  print("Average Cosine Similarity:", average_cosine_similarity.item())

  # RMSE  
  # Step 1: Calculate the squared differences
  squared_diff = (y_pred - y_true) ** 2
    
  # Step 2: Calculate the mean of squared differences along the first dimension (across n)
  mean_squared_diff = torch.mean(squared_diff, dim=0)
    
  # Step 3: Take the square root of the mean squared difference for each dimension
  rmse = torch.sqrt(mean_squared_diff).mean()

  print("RMSE is:", rmse.item())    

  ## Relative error
  # Step 1: Calculate Mean Absolute Error (MAE)
  mae = torch.mean(torch.abs(y_pred - y_true))

  print("MAE is: ", mae.item())
  
  # Step 2: Calculate Mean L2 Norm of the ground truth
  mean_l2_norm = torch.mean(torch.norm(y_true, dim=1))  # mean across n samples
  
  # Step 3: Calculate relative error
  if mean_l2_norm == 0:
      raise ValueError("Mean L2 norm of targets is zero; cannot compute relative error.")
  
  relative_error = mae / mean_l2_norm
  
  print("Relative error is: ", relative_error.item())

  return mae.item()
  
def load_val_dataset(val_dataset_filename):
  return val_pos, val_force

if __name__ == '__main__':
  # read GAMD 
  data_dir = "../top/" # [10000, [[258, 3], [258, 3], [258, 3]]]
  train_data_fraction = 0.9 # select 9k for training
  avg_num_neighbors = 20 # criteria for connectivity of atoms for any frame
  rotation_aug = False # online rotation augmentation for a frame
  batch_size = 1 # number of graphs in a batch
  # create train data-loader
  return_train_data = True  
  dataset = MDDataset(data_dir, rotation_aug, avg_num_neighbors, train_data_fraction, return_train_data) 
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)#, num_workers = os.cpu_count()) # create a batched graph and return
  
  # create val data loader
  return_train_data = False
  rotation_aug = False
  val_idx = dataset.get_val_data_idx()
  dataset_val = MDDataset(data_dir, rotation_aug, avg_num_neighbors, train_data_fraction, return_train_data, val_idx) 
  dataloader_val = DataLoader(dataset_val, batch_size=10, shuffle=False, collate_fn=custom_collate)#, num_workers = os.cpu_count()) # create a batched graph and return
  
  # train MPNN and input and output MLPs
  num_epochs = 30 # for batch size 1, 300k gradient updates are required. (as per paper)
  embed_dim = 128 
  hidden_dim = 128
  num_mpnn_layers = 4 # as per paper, for LJ system
  num_mlp_layers = 3
  num_atom_type_classes = 1 # Ar atoms only
  num_edge_types = 1 # non-bonded edges only
  num_rbfs = 10 # RBF expansion of interatomic distance vector of each edge to num_rbfs dimensions
  print("Creating GAMDNet module...")
  gamdnet = GAMDNet(embed_dim, hidden_dim, num_mpnn_layers, num_mlp_layers, num_atom_type_classes, num_edge_types, num_rbfs).to(device)  
  #torch.set_float32_matmul_precision('high')  # or 'medium'
  #gamdnet = torch.compile(gamdnet)
  #if torch.cuda.device_count() > 1:
  #  print(f"No. of detected CUDA devices: {torch.cuda.device_count()}, enabling data-parallel mode")
  #  gamdnet = nn.DataParallel(gamdnet)  
  print("GAMDNet inialized.")
  optimizer, scheduler = configure_optimizer_schedular(gamdnet, num_epochs)
  lambda_reg = 1e-3  
  criterion = GAMDLoss(lambda_reg=lambda_reg) # custom loss to include MAE and L1 reg. over predicted forces
  average_loss = 0.0
  total_items = 0
  train_data_size = dataset.__len__()
  num_iterations = train_data_size // batch_size

  selected_layers = ['edge_embedding_mlp', 'force_pred_mlp', 'edge_layer_norm', 'mpnn_mlps', '']  # Replace with actual layer names you want to monitor
  # Create a TensorBoard SummaryWriter
  writer = SummaryWriter()
  #print_model_summary(gamdnet, writer)
  #exit(0)
  best_mae = float('inf')
  save_path = 'best_model.pt'
  for epoch in range(num_epochs): 
    total_loss = 0.0           
    iteration = 0
    for pos, edge_index_list, force in dataloader: # [batch_size*258, 3], [#edges in a batch, 2], [batch_size * 258, 1] -> [258* batch size, 3]
      iter_start_time = time.time()
      print(f"Epoch: {epoch}, Iteration: {iteration}")
      pos = pos.to(device)
      edge_index_list = edge_index_list.to(device)
      force = force.to(device)      
      forward_start_time = time.time()           
      node_forces = gamdnet.forward(pos, edge_index_list) # [b * num_nodes, 3], [e, 2], [b * num_nodes] -> [b * num_nodes, 3]
      forward_end_time = time.time()      
      force_loss = criterion(node_forces, force) # [b * num_nodes, 3], [b * num_nodes, 3] -> [1]
      optimizer.zero_grad()
      force_loss.backward()
      with torch.no_grad():  # Disable gradient calculation
        #print("TRAIN PERFORMANCE: ")
        #evaluate(force, node_forces)
        if (iteration + 1) % 100 == 0: 
          # validate
          print("Reading val data...")
          pos_val, edge_index_list_val, force_val = next(iter(dataloader_val))
          node_forces_val = gamdnet.forward(pos_val, edge_index_list_val)                            
          print("VAL PEFORMANCE: ")
          mae = evaluate(force_val, node_forces_val)
          if mae < best_mae:
            best_mae = mae           
            # Save the new best model checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': gamdnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mae': best_mae,
            }, save_path)
            print(f'Saved new best model with MAE: {best_mae:.4f}')                              
      '''
      # Print gradients for each parameter in the model
      for name, param in gamdnet.named_parameters():
          if 'force_pred_mlp' in name:
              print(f'Gradient for {name}: {param.grad}')
      exit(0)
      
      # Collect gradients for selected layers
      for name, param in gamdnet.named_parameters():
        if param.requires_grad and any(layer in name for layer in selected_layers):
          # Log the gradient histogram to TensorBoard
          writer.add_histogram(f'gradients/{name}', param.grad, iteration + epoch * num_iterations)
      # Log GELU outputs
      for j, gelu_output in enumerate(gelu_outputs):
          writer.add_histogram(f'GELU_Output/layer_{j}/epoch_{epoch}', gelu_output.cpu().numpy(), iteration + epoch * num_iterations)          
      '''
      optimizer.step()       
      total_loss += force_loss.item()
      iteration += 1
      iter_end_time = time.time()

    scheduler.step()
    total_items += batch_size  
    average_loss = (average_loss * (total_items - batch_size) + total_loss) / total_items
    print(f"Average loss at epoch: {epoch} is: {average_loss}")    
    '''
    # After each epoch, log the average gradient norm for each layer
    for name, param in gamdnet.named_parameters():
      if param.requires_grad and any(layer in name for layer in selected_layers):
        writer.add_scalar(f'avg_gradient_norm/{name}', param.grad.norm().item(), epoch)   
    '''
  # Close the TensorBoard SummaryWriter
  writer.close()

  
  # infer on test data TODO
