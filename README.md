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

Arguments are parse using Hydra, configurations are organized into modules (nested dictionary structure). Each attribute
can be modified through the arguments:

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

## Controlling code quality

<details>
<summary>Using Github</summary>

### Opening issues

Each task, bug or idea should be registered as an issue. New issues are automatically added to `project/development/todo`.
Use `- [ ] <text>` to describe each item in a task.

### Using the project tab

Use the [project page](https://github.com/vlievin/fz-openqa/projects) to keep track of progress

### Branching

Do not implement features in the `master` branch. Create a new branch for each issue. Use a pull request to merge the branch with master and close the corresponding issue.
Closed issues are automatically moved to `project/development/done`.

</details>

<details>

<summary>Code formatting</summary>

### [Black](https://github.com/psf/black)

Black is a code formatter for python. You can run it indepedently using

```shell
black <directory>
```

### [flake8](https://flake8.pycqa.org/en/latest/)

Flake8 is a tool to ensure the code to be correctly formatted.

### Setting up git hooks using [pre-commit](http://python-poetry.org)

Git hooks allows to execute some piece of code before every commit/push/pull request/... Pre-commit hooks aim at
checking the format of the code before a commit. They can be installed using the following commands:

```shell
pip install pre-commit
pre-commit install
```

At every commit, both `black` and `flake8` will be run. If the code is not `flake8` compliant, the commit will be
rejected. Furthermore, you can run `flake8` and `black` using:

```shell
pre-commit run --all-files
```

</details>

<details>

<summary>Unit tests</summary>

Core functions should be properly tested. Unit tests can be implemented in `tests/` and executed using:

```shell
poetry run python -m unittest discover
```

## Main dependencies

The package relies on:

* [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to simplify training management including
  distributed computing, logging, checkpointing, early stopping, half precision training, ...
* [Hydra](https://hydra.cc/docs/intro/) for a clean management of experiments (setting hyper-parameters, ...)
* [Weights and Biases](https://wandb.ai) for clean logging and experiment tracking
* [Poetry](https://python-poetry.org/) for stricter dependency management and easy packaging

</details>

## credits

The original template comes from [ashleve](https://github.com/ashleve/lightning-hydra-template).
