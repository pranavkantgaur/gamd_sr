# gamd_sr
The objective is test the validity of [Miles hypothesis](https://arxiv.org/abs/2006.11287) over the intermolecular weak-force regime. Partial demonstration of the hypothesis by the author is [here](https://colab.research.google.com/github/MilesCranmer/symbolic_deep_learning/blob/master/GN_Demo_Colab.ipynb).
### What is Miles hypothesis?
If we add physics-informed inductive bias in the neural network based surrogate model of physical laws, then the intermidiate computations in NNs gets aligned to the underlying physical principles over which SR can be used to extract symbolic representation. The extracted or distilled symbolic representation _may_ generalize better than the NN itself (from which it was extracted).

## Motivation
If the hypothesis works, then we can perhaps discover potential functions for systems with only [neural network potentials](https://github.com/torchmd/torchmd-net) or [DFT based force-field](https://www.pnas.org/content/113/30/8368.short) calculations. If it does not work, we aim for a data-driven explanation for the same.


## Methodology
For the [datasets](https://github.com/BaratiLab/GAMD?tab=readme-ov-file#data-generation) studied in the [reference paper](https://arxiv.org/abs/2112.03383) with [official implementation](https://github.com/BaratiLab/GAMD), apply Miles hypothesis for symbolic distillation of GNN:
1. Fit GNN over data to predict net force on particles given input positions : https://github.com/pranavkantgaur/gamd_sr/blob/main/reference_gamd_train.py
2. Recover edge messages to see if it correlates with pair potential or pair force? : https://github.com/pranavkantgaur/gamd_sr/blob/main/sr_over_gnn.py (refer `are_edge_msgs_gt_potential_correlated`)
3. Regress analytic equation to predict edge messages as a function of particle-pair position and radial distance (A) : https://github.com/pranavkantgaur/gamd_sr/blob/main/sr_over_gnn.py (refer `regress_edge_message_equation`)
4. Regress analytic equation to predict net force as a function of aggregate edge messages (B) : https://github.com/pranavkantgaur/gamd_sr/blob/main/sr_over_gnn.py (refer `regress_net_force_equation`)
5. Compare generalization performance of resulting analytic equation (Input -> A -> B -> net force) vs the GNN over a test dataset.: https://github.com/pranavkantgaur/gamd_sr/blob/main/test_sr_generalization.py (**in progress**)

### Notes on the methodology
1. For systems (or test-cases) where pair-potential functions are well known, we can aim for best fit in SRs (A and B) alone, but still going for extracting the _known_ laws via SR makes sense becuase known forms are well-tested over the years. In that process, we learn to steer SR to the priors (if exist) on the functional forms, refer [this](https://github.com/MilesCranmer/PySR/issues/285).
2. Reported methods (in Miles paper) to introduce inductive bias in GNN-based surrogate models for physical laws are:
   1. l1 regularization over edge messages
   2. Constrain edge message dimensions to match that of the problem.
3. Additional ones experimented with here:
   1. Penalize standard deviation of all components but the first-3 (for 3D space) edge-message components.
   2. For known laws like `LJ potential` based systems, add factors like `r^-6`, `r^-12` in the edge-features (**planned**)
      
## Related works
### Generalization performance of NNs
1. https://arxiv.org/abs/2009.11848
2. https://arxiv.org/pdf/1905.13211

### SR approaches:
There are many approaches for SR nowdays, above is more of a hand-engineering approach, below are the recent end-to-end ones:
1. https://github.com/facebookresearch/symbolicregression
2. https://github.com/deep-symbolic-mathematics/LLM-SR
3. https://github.com/facebookresearch/recur
4. https://github.com/vastlik/symformer
5. https://github.com/omron-sinicx/transformer4sr
6. https://thegradient.pub/neural-algorithmic-reasoning/ (NAR looks eerily similar to SR)
