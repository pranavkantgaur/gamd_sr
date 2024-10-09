## Debugging summary
1. What we mean by the debugging activities here?
    1. All development activities post reading the GAMD paper till the model started achieving convergence.
2. What is the objective of debugging? What the success looks like?
    1. To match the loss value pattern in our implementation with those for the official impl. for single MD frame and all 9k MD frames.
3. Steps / Chronology of debugging:
    1. Started working on this from 8/8/22 with the hypothesis of GNN → SR for weak forces between atoms. Before 6/8/24 the underlying GNN architecture, the objective (force vs potential) and the time-commitment kept changing.
    2. Studied the [GAMD paper](https://arxiv.org/abs/2112.03383) on 7/8/24 for implementing it from scratch as-is with the available libraries, filling out a lot of implementation details ourselves. Perhaps resulting in different implementation hypothesis than the one claimed by authors to have worked.
    3. Trained the resulting implemented model but it did not converge.
    4. Read the [official codebase](https://github.com/BaratiLab/GAMD) line by line to mark out the following differences wrt. what we filled out in our implementation solely based on the paper:
        1. Dataset loading:
            1. None, 9000 training files, 1000 randomly selected validation dataset files
        2. Dataset preprocessing:
            1. periodic boundary condition enforced on the particle positions via numpy.mod
            2. jitter to positions
            3. normalization of forces based on a running mean and variance of training dataset force values
        3. Graph representation and processing:
            1. Node embedding calculations:
                1. Using nn.param to randomly init a tunable node embedding and copying it to all nodes as the initial node embeddings before message passing
                2. Did not use one-hot over atom type followed by passing it through a MLP for LJ dataset atleast.
            2. Edge embedding calculations:
                1. Edge feature calculations:
                    1. RBF implementation:
                        1. Static center calculation with a fixed gap value
                2. Layer norm over edges
                3. Edge embedding MLP:
                    1. Number of linear layers: 3 
                    2. Activation function: GeLU
            3. Edge index calculation:
                1. Fixed box size, and radius cutoff (not based on average number of neighbors).
            4. Message passing implementation:
                1. No updates to the initial edge embeddings across message passing layers
                2. Fused kernel for SGEMM used for message passing rather than the explicit message calculation
                3. Theta:
                    1. Names switched?: Theta ↔ Phi
                    2. Number of linear layers : 2
                    3. Activation function : SiLU
                4. Phi:
                    1. Number of linear layers: 1
                    2. Activation function: SiLU
        4. Force prediction:
            1. Number of Linear layer: 2
            2. Activation function : GeLU
        5. Optimizer/schedular differences:
            1. StepLR used for LR scheduling
    5. Implemented changes in our codebase to align our implementation hypothesis with official one and retrained but still no convergence. Decided to debug more deeply, ideally unit-testing our implementation wrt. intermidiate tensor outputs from the official impl. hypothesis.
    6. Simplified the official impl. hypothesis and targetting convergence within 1.2k iterations of the 1st epoch to ease the complexity of subsequent debugging (effectively switching off the non-critical components in the official impl. for LJ dataset):
        1. List of elements which are not critical for convergence:
            1. dataset preprocessing:
                1. random augmentation
                2. position jittering
                3. np.mod for enforcing boundary conditions
            2. graph representation:
                1. self loops 
                2. edge embeddings:
                    1. relative position norm for edge feature
                    2. RBF based edge feature expansion
                3. message passing:
                    1. edge dropout
                    2. affine mapping/embedding of edge embeddings, source and destination node embeddings
                    3. Linear embedding over  the aggregate edge messages and node embeddings just before updating the node embeddings after message passing using phi MLP
        2. List of elements which are very critical:
            1. layer norm and residual connections during message passing
            2. batch size = 1, for higher numbers convergence slows down drastically
    7. Eventually, when the model implementation matched exactly with official impl., model started converging. Reference: [our implementation](9be5c36271acdc54c3500dbb8928feacdf296f74), [official implementation](5827363cdd4f16e2d5d95ce7f3d8d2fb4e068b88).
    8. Still the loss is not super-low, so maybe we still need to continue debugging to acheive lower loss value. (**Current status 9/10/24**)
4. Results:
    1. Model converges with identical loss values as the simplified official implementation on single MD frame (or simulation).
    2. Model converges to a loss value of 0.1 starting from around 0.7 matching the performance of the simplified official implementation on 9k MD frames.
5. Retrospective:
    1. What these results mean?
        1. No serious bug in this portion, perhaps will require adding more components in next iteration of GNN-SR cycle. Lets move over to setting up SR on this.
    2. What is next?
        1. Complete GNN-SR iteraton 1:
            1. How to express and track mae in percentage? https://www.perplexity.ai/search/how-to-express-mean-absolute-e-PHS6mVllTOWDSkriPaI6Sg#0
            2. Add validation loss , model checkpoint 
            3. Dump edge embeddings of best performing model for subsequent SR
            4. Add SR code for force equation prediction
            5. Run the resulting GNN -> SR pipeline and plan workitems for the next iteration based on results.
            6. Tentative workitems for the next iteration:
               1. Whether the model overfits on a single file?
               2. At what loss value to say that it has overfitted?
               3. Why the current model is not able to go below 0.1 on training data?
