## Intro
This repo contains code for the [paper](https://openreview.net/forum?id=U49N5V51rU) *A Formal Framework for Understanding Length Generalization in Transformers*

See [QUICKSTART.md](QUICKSTART.md) for setup instructions, usage, and documentation of new features (including the MQAR Word Problem task).

## Current setup and workflow

The ```example.sub``` file gives a blue-print of how experiments can be run. All dependencies are handled within the docker image. Run scripts for the different architectures can be found in ```algorithmic/run_scripts/```. 

#### Arguments

For help on the available arguments see ```algorithmic/run_scripts/language_modeling_train_shared.py```.


#### Model Hyperparameters

For each run, the model's hyperparameters need to be manually set within the respective run scripts in ```algorithmic/run_scripts/```. They form list over which the script iterates, performing training on all combinations of those model parameters.

#### Tasks
They can be simply specified within the .sub file.
