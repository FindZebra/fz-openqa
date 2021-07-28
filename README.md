![FindZebra: rare disease search](fz-banner.png)

# Medical Open Domain Question Answering

<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7--3.9-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

[![hackmd-github-sync-badge](https://hackmd.io/HQFPXkocSMKuJvtWWVJNKg/badge)](https://hackmd.io/HQFPXkocSMKuJvtWWVJNKg)

## Setup

<details>
<summary>Evironment</summary>

1. Install poetry

```shell
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

2. Install dependencies (within the project)

```shell
poetry install
```

3. setting up git hooks

```shell
pip install pre-commit
pre-commit install
```

4. Run something using the environment

```shell
poetry run python <file.py>
poetry run which python # return the path to the virtual environment
```

</details>

<details>
<summary>Running the code</summary>

`src.cli.main` can be called directly using the command `fzqa` (which can be edited in the `.toml` file):

```shell
poetry run fzqa <args>
```

Or run the python script directly:

```shell
poetry run python run.py <args>
```

</details>


<details>
<summary>Using Github</summary>

### Opening issues

Each task, bug or idea should be registered as an issue. New issues are automatically added
to `project/development/todo`. Use `- [ ] <text>` to describe each item in a task.

### Using the project tab

Use the [project page](https://github.com/vlievin/fz-openqa/projects) to keep track of progress

### Branching

Do not implement features in the `master` branch. Create a new branch for each issue. Use a pull request to merge the
branch with master and close the corresponding issue. Closed issues are automatically moved
to `project/development/done`.

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

</details>

## Running the code

`src.cli.main` can be called directly using the command `fzqa` (which can be edited in the `.toml` file):

```shell
poetry run fzqa <args>
```

<details>
<summary>Passing Arguments</summary>
Arguments are parse using Hydra, configurations are organized into modules (nested dictionary structure). Each attribute
can be modified through the arguments:

```shell
poetry run fzqa trainger.gpus=0 trainer.max_epochs=100 logger=wandb datamodule.lr=0.007
```

</details>

<details>
<summary>Configuring experiments</summary>

Experiment configurations define a full experimental setup in `configs/experiment/`. Run the experiment config using:

```shell
poetry run fzqa experiment=quick_test
```

The ´environ´ configuration adjust the experiment to the environment (e.g. cache location).

</details>

<details>
<summary>GPU cluster</summary>

When running experiments on the GPU cluster, you need to pass the flag `CUDA_VISIBLE_DEVICES` to expose GPU devices to
your script. The `/scratch` directory should be used to store large files (cache).

```shell
 CUDA_VISIBLE_DEVICES=7 poetry run fzqa experiment=reader_only environ=valv trainer.gpus=1
 ```

Lightning enables multi-gpus training using `torch.nn.DataParallel`. Simply configure the Lightning trainer:

```shell
CUDA_VISIBLE_DEVICES=3,4,5,6 poetry run python run.py experiment=retriever_only +trainer.accelerator=dp trainer.gpus=4
```
 </details>

 <details>
<summary>Hyper parameter optimization</summary>

The `tune.py` script allow scheduling and running a set of experiments using `Ray[tune]`. Each experiment is described in `configs/hpo/`. Run an experiment using:

```shell
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 poetry run python tune.py +hpo=search_retriever
 ```

</details>

## Documentation

### Module design

All modules should inherit from `BaseModel`, which in turn inherits from `pl.LightningModule`.
Each module features one `Evaluator` which role is to compute the loss and the metrics.
metrics are computed using `torchmetrics` (see the section `Data flow for multiple GPUs` for more details).

<details>
<summary>Data flow within the `BaseModels` (multi-GPUs)</summary>

The main computation should be implemented in the `_step()` and `_step_end()` methods of the `BaseModel`.
The `_step()` method runs independently on each device whereas the `_step_end()` method runs on
a single device: this is where the final aggregated loss should be implemented (see the diagram below).
The metrics must be implemented in the `_step_end` method in order to avoid errors with mutli-GPU training.

![Lightning module data flow](.assets/lighning_steps.png)

</details>

<details>
<summary>`Evaluator`</summary>
The evaluator handles computing the loss and the metrics. Two methods must be implemented:

1. The `forward` method that calls the model and compute logits or potentially a pre-loss term.
This method is called in the `module._step()` method
2. The `post_forward` method that implements the final computation of the loss given the aggregated outputs of the
`Evaluator.foward()` method from each device.
</details>

### Pesudo-code for Supervised OpenQA

The basic End-to-end OpenQA model relies on a single pretrained BERT model. The model functions as follows:

<details>
<summary>Retriever model</summary>

```python
from copy import deepcopy
import torch
from torch import Tensor, nn, einsum, argmax
from transformers import AutoModel
from fz_openqa.modeling.layers.lambd import Lambda

hdim = 16
bert = AutoModel.from_pretrained('model_id')
head_q = nn.Sequential(nn.Linear(bert.config.hidden_size, hdim),
                       Lambda(lambda x: x[:, 0]))
head_d = deepcopy(head_q)


def h_q(q: Tensor) -> Tensor:
    """pseudo-code for the query model"""
    return head_q(bert(q).last_hidden_state)  # tensor of shape [n_q, h,]


def h_d(d: Tensor) -> Tensor:
    """pseudo-code for the query model"""
    return head_d(bert(d).last_hidden_state)  # tensor of shape [n_d, h,]


def sim(h_q: Tensor, h_d: Tensor) -> Tensor:
    """Compute the similarity matrix between the batch of queries and the documents"""
    return einsum(f'nh, mh -> nm', h_q, h_d)  # tensor  of shape [n_q, n_d]


def topk(similarities: Tensor, k: int) -> Tensor:
    """return the topk document for each query in the batch given the similarity matrix"""
    values, indices = torch.topk(similarities, k=k, dim=1)  # tensor of shape [m_q, min(k, n_d)]
    return indices


def retriever(q: Tensor, d: Tensor, k: int) -> Tensor:
    """Retrieve the top k document form the corpus `d`
    for each query in the batch `q`"""
    similarities = sim(h_q(q), h_d(d))
    return topk(similarities, k)
```

</details>

<details>
<summary>Reader model</summary>

```python
from copy import deepcopy
import torch
from torch import Tensor, nn, einsum, cat
from transformers import AutoModel
from fz_openqa.modeling.layers.lambd import Lambda

hdim = 16
bert = AutoModel.from_pretrained('model_id')
head_qd = nn.Sequential(nn.Linear(bert.config.hidden_size, hdim),
                        Lambda(lambda x: x[:, 0]))
head_a = deepcopy(head_qd)


def h_qd(q: Tensor, d: Tensor) -> Tensor:
    """pseudo-code for the query-document model"""
    qd = cat([q, d], dim=1)
    return head_qd(bert(qd).last_hidden_state)  # tensor of shape [n_qd, h,]


def h_a(a: Tensor) -> Tensor:
    """pseudo-code for the answer model"""
    return head_a(bert(a).last_hidden_state)  # tensor of shape [n_a, h,]


def sim(h_qd: Tensor, h_a: Tensor) -> Tensor:
    """Compute the similarity matrix between the batch of query-documents and the answers"""
    return einsum(f'nh, nah -> na', h_qd, h_a)  # tensor  of shape [n_qd, n_a]


def topk(similarities: Tensor, k: int) -> Tensor:
    """return the top k document-answers for each query in the batch given the similarity matrix"""
    values, indices = torch.topk(similarities, k=k, dim=1)  # tensor of shape [n_qd min(k, n_a)]
    return indices


def reader(q: Tensor, d: Tensor, a: Tensor, k: int) -> Tensor:
    """Retrieve the top k answers given a batch of triplets (query, document, answer)"""
    similarities = sim(h_qd(q, d), h_a(a))
    return topk(similarities, k)
```

</details>

<details>
<summary>Supervised Training</summary>

The `FZxMedQA` dataset provides triplets `(q, d, a)` that can be exploited for supervised learning. In this setup the
retriever only learns from the label (golden passage). The pseudo-code looks like:

```python
import torch

for batch in loader:
    # shapes: q: [bs, L_q, :], d: [bs, L_d, :], a: [bs, N_a, L_a, :], a_index: [bs,]
    q, d, a, a_index = batch
    bs, N_a, L_a, _ = a.shape
    # retriever loss
    ir_logits = sim(h_q(q), h_d(d))
    retriever_loss = torch.nn.functional.cross_entropy(ir_logits, torch.range(ir_logits.shape[0]))
    # reader loss
    _h_qd = h_qd(q, d)  # shape [bs, h]
    _h_a = h_a(a.view(bs * N_a, *a.shape[2:])).view(bs, N_a,
                                                    -1)  # collapse bs and N_a, and reshape back, shape [bs, N_a, h]
    qa_logits = torch.einsum(f'nh, nah -> na', _h_qd, _h_a)
    reader_loss = torch.nn.functional.cross_entropy(qa_logits, a)
    # total loss
    loss = retriever_loss + retriever_loss
    # backward, etc...
    ...




```

</details>

<details>
<summary>End-to-end evaluation</summary>

During supervised training, the retriever only learns from the golden passages, and the reader is only evaluated using
the golden passage. During end-to-end evaluation, we wish to use the documents actually retrieved using the trained
model.

```python
# step 1. index the corpus
for batch in corpus:
    batch['vectors'] = h_d(batch['input_ids'])
corpus.add_faiss_index('vectors')

# step 2. end to end evaluation
for batch in loader:
    q, a, a_index = batch
    # retriever the best document for each query
    d = corpus.get_nearest_examples_batch('vectors', k=1)  # potentially use k>1
    # feed the best document to the reader
    a_inferred = reader(q, d, a)
    accuracy = Accuracy(a_inferred, a_index)
    # log, etc...
    ...
```

</details>

## Future Improvements

<details>
<summary>Feed the answer choices to the retriever</summary>
At the moment the current model does not use the answer choices for retrieval. Concatenate the answer choices with the query.
</details>

<details>
<summary>Late-interaction reader model</summary>
At the moment, the reader model requires concatenating the query with the document,
which requires processing the query and document two times (1 time for IR, one time for reading comprehension).
A late interaction model for the reader component would allow processing each input one time with the BERT model.
</details>

<details>
<summary>End-to-end training</summary>
The current retriever only learns to identify the golden passage (which is noisily labelled).
Sample from the retriever lives and learn from the signal given by the reader component.
</details>

### Credits

The package relies on:

* [Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to simplify training management including
  distributed computing, logging, checkpointing, early stopping, half precision training, ...
* [Hydra](https://hydra.cc/docs/intro/) for a clean management of experiments (setting hyper-parameters, ...)
* [Weights and Biases](https://wandb.ai) for clean logging and experiment tracking
* [Poetry](https://python-poetry.org/) for stricter dependency management and easy packaging
* The original template was copied form [ashleve](https://github.com/ashleve/lightning-hydra-template)
