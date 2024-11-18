# gamd_sr
For the datasets studied in the [reference paper](https://arxiv.org/abs/2112.03383):
1. Fit GNN over data to predict net force on particles given input positions : https://github.com/pranavkantgaur/gamd_sr/blob/main/reference_gamd_train.py
2. Recover edge messages to see if it correlates with pair potential or pair force? : https://github.com/pranavkantgaur/gamd_sr/blob/main/sr_over_gnn.py (refer `are_edge_msgs_gt_potential_correlated`)
3. Regress analytic equation to predict edge messages as a function of particle-pair position and radial distance (A) : https://github.com/pranavkantgaur/gamd_sr/blob/main/sr_over_gnn.py (refer `regress_edge_message_equation`)
4. Regress analytic equation to predict net force as a function of aggregate edge messages (B) : https://github.com/pranavkantgaur/gamd_sr/blob/main/sr_over_gnn.py (refer `regress_net_force_equation`)
5. Compare generalization performance of resulting analytic equation (Input -> A -> B -> net force) vs the GNN over a test dataset.: https://github.com/pranavkantgaur/gamd_sr/blob/main/test_sr_generalization.py (**in progress**)
