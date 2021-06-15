# Medical Open Domain Question Answering - FindZebra

## Environment Setup

Install poetry

```shell
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

Install dependencies (within the project)
```shell
poetry install
```

## Running the code

`src.cli.main` can be called directly using the command `fzqa` (which can be edited in the `.toml` file):
```shell
poetry run fzqa <args>
```
Arguments are parse using Hydra, configurations are organized into modules (nested dictionary structure). Each attribute can be modified through the arguments:
```shell
poetry run fzqa trainger.gpus=0 trainer.max_epochs=100 logger=wandb datamodule.lr=0.007
```

Experiment configurations define a full experimental setup, overriding other configurations
```shell
poetry run fzqa +experiment=quick_test
```

Running on the server:
```shell
 CUDA_VISIBLE_DEVICES=7 poetry run fzqa +experiment=reader_only work_dir=/scratch/valv/runs 
 ```

Multi-gpus training:
```shell
CUDA_VISIBLE_DEVICES=3,4,5,6 poetry run python run.py +experiment=retriever_only +trainer.accelerator=ddp trainer.gpus=4
```

## Testing

```shell
poetry run python -m unittest discover
```

## Main dependencies

The package relies on:
* [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to simplify training management including distributed computing, logging, checkpointing, early stopping, half precision training, ...
* [Hydra](https://hydra.cc/docs/intro/) for a clean management of experiments (setting hyper-parameters, ...)
* [Weights and Biases](https://wandb.ai) for clean logging and experiment tracking
* [Poetry](https://python-poetry.org/) for stricter dependency management and easy packaging


## credits

The original template comes from [ashleve](https://github.com/ashleve/lightning-hydra-template).
