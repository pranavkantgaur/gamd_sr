# gamd_sr
The objective is test the validity of [Miles hypothesis](https://arxiv.org/abs/2006.11287) over the intramolecular weak-force regime. Partial demonstration of the hypothesis by the author is [here](https://colab.research.google.com/github/MilesCranmer/symbolic_deep_learning/blob/master/GN_Demo_Colab.ipynb).

## Motivation
If the hypothesis works, then we can perhaps discover potential functions for systems with only [neural network potentials](https://github.com/torchmd/torchmd-net) or [DFT based force-field](https://www.pnas.org/content/113/30/8368.short) calculations.
## Methodology
For the [datasets](https://github.com/BaratiLab/GAMD?tab=readme-ov-file#data-generation) studied in the [reference paper](https://arxiv.org/abs/2112.03383) with [official implementation](https://github.com/BaratiLab/GAMD), apply Miles hypothesis for symbolic distillation of GNN:
1. Fit GNN over data to predict net force on particles given input positions : https://github.com/pranavkantgaur/gamd_sr/blob/main/reference_gamd_train.py
2. Recover edge messages to see if it correlates with pair potential or pair force? : https://github.com/pranavkantgaur/gamd_sr/blob/main/sr_over_gnn.py (refer `are_edge_msgs_gt_potential_correlated`)
3. Regress analytic equation to predict edge messages as a function of particle-pair position and radial distance (A) : https://github.com/pranavkantgaur/gamd_sr/blob/main/sr_over_gnn.py (refer `regress_edge_message_equation`)
4. Regress analytic equation to predict net force as a function of aggregate edge messages (B) : https://github.com/pranavkantgaur/gamd_sr/blob/main/sr_over_gnn.py (refer `regress_net_force_equation`)
5. Compare generalization performance of resulting analytic equation (Input -> A -> B -> net force) vs the GNN over a test dataset.: https://github.com/pranavkantgaur/gamd_sr/blob/main/test_sr_generalization.py (**in progress**)
