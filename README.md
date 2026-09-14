## Intro
This repo contains code for the [paper](https://openreview.net/forum?id=U49N5V51rU) *A Formal Framework for Understanding Length Generalization in Transformers*

See [QUICKSTART.md](QUICKSTART.md) for setup instructions, usage, and documentation of new features (including the MQAR Word Problem task).

## Current setup and workflow

The ```example.sub``` file gives a blue-print of how experiments can be run. All dependencies are handled within the docker image. All architectures share one entrypoint, ```algorithmic/language_modeling_train.py```, which is told which model to train with ```--model <spec>```.

#### Arguments

For help on the available arguments run ```python algorithmic/language_modeling_train.py --help```.


#### Model Hyperparameters

The model — family, positional encoding, layer norm, SSM kernel, hybrid layer pattern, and the architecture sweep — is described by a YAML spec in ```algorithmic/model_specs/```; list them with ```python algorithmic/language_modeling_train.py --list-models```. Fields of an ```architectures``` entry may be lists, in which case training iterates over all combinations. Every other hyperparameter is a command-line flag and can therefore be set in the .sub file.

#### Tasks
They can be simply specified within the .sub file.
