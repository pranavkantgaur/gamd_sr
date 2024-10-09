# gamd_sr
GNN over MD simulation data -> Symbolic regression over edge embeddings -> Force equation recovered?

# TODOs
1. Summarize debugging so far
2. Update PhD in nos note
3. How does the official model perform on this data without any code changes?
4. Model does not go below 0.2-0.1 loss? WHY? 
5. Next: Create separate repo for GAMD reimplementation and share with Biswajit for review
6. Write Sgmm based message passing using triton?
7. Also, not that GAMD train in mixed precision, why? Also, do we require to do loss scaling here? Whether it is happening in GAMD? https://lightning.ai/pages/community/tutorial/accelerating-large-language-mod.els-with-mixed-precision-techniques/
8. Schedule https://docs.dgl.ai/en/2.2.x/tutorials/models/4_old_wines/7_transformer.html#sphx-glr-tutorials-models-4-old-wines-7-transformer-py
9. create a notebook for LJ data exploration and the following:
10. E2E training eval skeleton:
    1. Create dataloader
    2. Create a model class which can then latter be experimented with
    3. Create validation data loader
    4. Write [model.fit](http://model.fit) where the model class could be experimented with starting from a baseline model
11. Get and document dumb baselines in a notebook:
    1. Lower bound: dumbest model for prediction on validation data
    2. Upper bound: Human performance on validation data
12. Overfit on single MD file
13. Test overfitting on any random MD file
14. Regularize the model to improve it for the validation dataset
15. Tune by hyperparam search
16. Further improve by ensembling and longer training runs
