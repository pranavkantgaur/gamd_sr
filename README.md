# gamd_sr
The objective is test the validity of [GNN symbolic distillation hypothesis](https://arxiv.org/abs/2006.11287) over the intermolecular weak-force regime. Partial demonstration of the hypothesis by the author is [here](https://colab.research.google.com/github/MilesCranmer/symbolic_deep_learning/blob/master/GN_Demo_Colab.ipynb), and an attempt to complete the demo is [here](https://github.com/MilesCranmer/PySR/issues/36).

### What is GNN symbolic distillation hypothesis?
If we add physics-informed inductive bias in the neural network based surrogate model of physical laws, then the intermidiate computations in NNs gets aligned to the underlying physical principles over which SR can be used to extract symbolic representation. The extracted or distilled symbolic representation _may_ generalize better than the NN itself (from which it was extracted).

## Motivation
If the hypothesis works in our case, then we can perhaps discover potential functions for systems with only [neural network potentials](https://github.com/torchmd/torchmd-net) or [DFT based force-field](https://www.pnas.org/content/113/30/8368.short) calculations. If it does not work, we aim for a data-driven explanation for the same.

### What an end-to-end SR looks like in our case?
1. End to End SR is tasked to map a N * 3 dimensional input to N * 3 dimensional output, where N is number of atoms in a system -> Intractable.
2. The approach of symbolic distillation of GNN as proposed by Miles, breaks down SR task to two cascaded SRs: (Tractable)
   1. SR-1 generates 3 equations each of which maps `dx`, `dy`, `dz`, `r` to `e1`, `e2`, `e3` (4D input to 1D output)
   2. SR-2 generates 3 equations each of which maps `posX`, `posY`, `posZ`, `agg-1`, `agg-2`, `agg-3` to `f1`, `f2`, `f3` (6D input to 1D output)
3. We can also fit MLPs or interpretable models like decision trees instead of SRs here (aka, _knowledge distillation_ from teacher model (GNN) to student (SRs, MLPs, Decision tree and so on)). 


## Methodology
For the [datasets](https://github.com/BaratiLab/GAMD?tab=readme-ov-file#data-generation) studied in the [reference paper](https://arxiv.org/abs/2112.03383) with [official implementation](https://github.com/BaratiLab/GAMD), apply Miles hypothesis for symbolic distillation of GNN:
1. Fit GNN over data to predict net force on particles given input positions : https://github.com/pranavkantgaur/gamd_sr/blob/main/reference_gamd_train.py
2. Recover edge messages to see if it correlates with pair potential or pair force? : https://github.com/pranavkantgaur/gamd_sr/blob/main/sr_over_gnn.py (refer `are_edge_msgs_gt_potential_correlated`)
3. Regress analytic equation to predict edge messages as a function of particle-pair position and radial distance (A) : https://github.com/pranavkantgaur/gamd_sr/blob/main/sr_over_gnn.py (refer `regress_edge_message_equation`)
4. Regress analytic equation to predict net force as a function of aggregate edge messages (B) : https://github.com/pranavkantgaur/gamd_sr/blob/main/sr_over_gnn.py (refer `regress_net_force_equation`)
5. Compare generalization performance of resulting analytic equation (Input -> A -> B -> net force) vs the GNN over a test dataset.: https://github.com/pranavkantgaur/gamd_sr/blob/main/test_sr_generalization.py 

### Notes on the methodology
1. For systems (or test-cases) where pair-potential functions are well known, we can aim for best fit in SRs (A and B) alone, but still going for extracting the _known_ laws via SR makes sense becuase known forms are well-tested over the years. In that process, we learn to steer SR to the priors (if exist) on the functional forms, refer [this](https://github.com/MilesCranmer/PySR/issues/285).
2. Reported methods (in Miles paper) to introduce inductive bias in GNN-based surrogate models for physical laws are:
   1. l1 regularization over edge messages
   2. Constrain edge message dimensions to match that of the problem.
   3. Minimise KL divergence between edge messages and Gaussian distribution.
3. Additional ones experimented with here:
   1. Penalize standard deviation of all components but the first-3 (for 3D space) edge-message components.
   2. For known laws like `LJ potential` based systems, add factors like `r^-6`, `r^-12` in the edge-features (**planned**)
   3. Penalize top-3 aggregate edge message components to align with net LJ potential evaluation (**planned**)
   4. Using PCA map 128-dim edge messages to top-3 message components (**planned**)
4. Feasibility of doing symbolic regression over intermediate representations of GNN:
   1. SR for message components: If LJ pair-potential is a function of `dx`, `dy`, `dz`, `r`, and edge message components `e1`, `e2`, `e3` (with highest standard deviation) can be expressed as linear combinations of the components of LJ potential, then `e1`, `e2`, `e3` are functions of `dx`, `dy`, `dz`, `r` as well -> SR could be done.
   2. SR for net force components: If net LJ force is a function aggregate edge message components 1, aggregate edge message components 2, aggregate edge messages component 3, position x, position y, position z.
   
## Results (in progress)
1. Dataset: Lennard Jones system with non bonded argon atoms 
   1. GNN MAE over validation data: 
   2. Linearity fit score between edge messages and LJ pair potential (e1,e2, e3) as a function of dx, dy, dz and r: 
   3. Linearity for score between edge message components (e1, e2, e3) and LJ pair force a function of dx, dy, dz and r: 
   4. SR MAE for predicting edge message components as a function of dx, dy, dz, r: 
   5. Linearity fit score between aggregate edge message components 1, aggregate edge message components 2, aggregate edge messages component 3, position x, position y, position z and net LJ force equation:
   6. SR MAE for predicting net force components f1, f2, f3 as functions of aggregate edge message components 1, aggregate edge message components 2, aggregate edge messages component 3, position x, position y, position z: 
   7. End to end SR MAE vs GNN over interpolation test data: 
   8. End to end SR MAE vs GNN over extrapolation test data:
2. Continue with other datasets if the hypothesis shows promise over LJ system test-case.
      
## Related works
### Generalization of NNs: 
1. https://arxiv.org/pdf/1905.13211 ([official](https://github.com/NNReasoning/What-Can-Neural-Networks-Reason-About) and [community](https://github.com/KushajveerSingh/deep_learning/tree/main/graph_machine_learning/what_can_neural_networks_reason_about))
2. https://arxiv.org/abs/1907.03950 ([official](https://github.com/ceyzaguirre4/NSM))
### Extrapolation by NNs:
1. https://arxiv.org/abs/2009.11848 ([official](https://github.com/jinglingli/nn-extrapolate))


### SR approaches:
There are many approaches for SR nowdays, above is more of a hand-engineering approach, below are the recent end-to-end ones:
1. https://github.com/facebookresearch/symbolicregression
2. https://github.com/deep-symbolic-mathematics/LLM-SR (Uses LLM for regression over program template and fits parameters using minimization procedures, very similar to what pySR does using GA + optimizers like BGFS, refer [here](https://github.com/MilesCranmer/PySR/issues/36#issuecomment-791890120)).
3. https://github.com/facebookresearch/recur
4. https://github.com/vastlik/symformer
5. https://github.com/omron-sinicx/transformer4sr
6. https://thegradient.pub/neural-algorithmic-reasoning/ (NAR looks eerily similar to SR)

### Knowledge distillation from GNNs:
1. https://github.com/lexpk/LogicalDistillationOfGNNs
2. https://github.com/wutaiqiang/awesome-GNN2MLP-distillation (a curated list)
3. https://github.com/YangLing0818/VQGraph
4. https://github.com/Rufaim/routing-by-memory
