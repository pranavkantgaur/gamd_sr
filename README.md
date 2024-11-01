# gamd_sr
For the datasets studied in the [reference paper](https://arxiv.org/abs/2112.03383):
1. Fit GNN over data to predict net force on particles given input positions
2. Recover edge messages to see if it correlates with pair potential or pair force?
3. Regress analytic equation to predict edge messages as a function of particle-pair position and radial distance (A)
4. Regress analytic equation to predict net force as a function of aggregate edge messages (B)
5. Compare generalization performance of resulting analytic equation (Input -> A -> B -> net force) vs the GNN over a test dataset.
