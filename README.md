# PyTorch-Lightning for Natural Language Processing

A template for Natural Language Processing using:
* [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to simplify training management including distributed computing, logging, checkpointing, early stopping, half precision training, ...
* [Hydra](https://hydra.cc/docs/intro/) for a clean management of experiments (setting hyper-parameters, ...)
* [Weights and Biases](https://wandb.ai) for clean logging and experiment tracking
* [Poetry](https://python-poetry.org/) for stricter dependency management and easy packaging

## Running the code

`src.cli.main` can be called directly using the command `nlp-lightning` (which can be edited in the `.toml` file):
```shell
poetry run nlp-lightning <args>
```
Arguments are parse using Hydra, configurations are organized into modules (nested dictionary structure). Each attribute can be modified through the arguments:
```shell
poetry run nlp-lightning trainger.gpus=0 trainer.max_epochs=100 logger=wandb datamodule.lr=0.007
```

Experiment configurations define a full experimental setup, overriding other configurations
```shell
poetry run nlp-lightning +experiment=quick_test
```

## Testing

```shell
poetry run python -m unittest discover
```


## credits

A large part of the repo is copied from [ashleve](https://github.com/ashleve/lightning-hydra-template).
