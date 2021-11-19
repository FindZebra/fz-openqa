# DataModule

> Auto-generated documentation for [fz_openqa.datamodules.datamodule](blob/master/fz_openqa/datamodules/datamodule.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Datamodules](index.md#datamodules) / DataModule
    - [DataModule](#datamodule)
        - [DataModule().collate_fn](#datamodulecollate_fn)
        - [DataModule().display_one_sample](#datamoduledisplay_one_sample)
        - [DataModule().display_samples](#datamoduledisplay_samples)
        - [DataModule().get_dataset](#datamoduleget_dataset)
        - [DataModule().prepare_data](#datamoduleprepare_data)
        - [DataModule().setup](#datamodulesetup)
        - [DataModule().test_dataloader](#datamoduletest_dataloader)
        - [DataModule().train_dataloader](#datamoduletrain_dataloader)
        - [DataModule().val_dataloader](#datamoduleval_dataloader)

## DataModule

[[find in source code]](blob/master/fz_openqa/datamodules/datamodule.py#L31)

```python
class DataModule(LightningDataModule):
    def __init__(
        builder: Union[DatasetBuilder, DictConfig],
        train_batch_size: int = 64,
        eval_batch_size: int = 128,
        num_workers: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        **kwargs,
    ):
```

A task agnostic base datamodule. This implements and showcase the
basic functionalities of a text DataModule.

Implementing a sub-class of a `BaseDataModule` mostly requires overriding the
`.preprocessing()` and `.collate_fn()` methods.

<original documentation>
A DataModule implements 5 key methods:
    - prepare_data (things to do on every noe)
    - setup (things to do on every accelerator in distributed mode)
    - train_dataloader (the training dataloader)
    - val_dataloader (the validation dataloader(s))
    - test_dataloader (the test dataloader(s))

This allows you to share a full dataset without explaining how to download,
split, transform and process the data
Read the docs:
    https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html

### DataModule().collate_fn

[[find in source code]](blob/master/fz_openqa/datamodules/datamodule.py#L152)

```python
def collate_fn(examples: List[Batch]) -> Batch:
```

The function that is used to merge examples into a batch.
Concatenating sequences with different length requires padding them.

#### See also

- [Batch](../utils/datastruct.md#batch)

### DataModule().display_one_sample

[[find in source code]](blob/master/fz_openqa/datamodules/datamodule.py#L175)

```python
def display_one_sample(example: Dict[str, torch.Tensor]):
```

Decode and print one example from the batch

### DataModule().display_samples

[[find in source code]](blob/master/fz_openqa/datamodules/datamodule.py#L160)

```python
@rank_zero_only
def display_samples(n_samples: int = 1):
```

Sample a batch and pretty print it.

### DataModule().get_dataset

[[find in source code]](blob/master/fz_openqa/datamodules/datamodule.py#L139)

```python
def get_dataset(
    split: Union[str, Split],
    dataset: Optional[HfDataset] = None,
) -> Union[TorchDataset, Dataset]:
```

Return the dataset corresponding to the split,
or the dataset iteself if there is no split.

### DataModule().prepare_data

[[find in source code]](blob/master/fz_openqa/datamodules/datamodule.py#L83)

```python
def prepare_data():
```

Download data if needed. This method is called only from a single GPU.
Do not use it to assign state (self.x = y).

### DataModule().setup

[[find in source code]](blob/master/fz_openqa/datamodules/datamodule.py#L89)

```python
def setup(stage: Optional[str] = None):
```

Load data and preprocess the data.
1. Store all data into the attribute `self.dataset` using `self.preprocess_dataset`
2. Build the operator to collate examples into a batch (`self.collate_pipe`).

### DataModule().test_dataloader

[[find in source code]](blob/master/fz_openqa/datamodules/datamodule.py#L136)

```python
def test_dataloader(**kwargs):
```

### DataModule().train_dataloader

[[find in source code]](blob/master/fz_openqa/datamodules/datamodule.py#L101)

```python
def train_dataloader(shuffle: bool = True):
```

### DataModule().val_dataloader

[[find in source code]](blob/master/fz_openqa/datamodules/datamodule.py#L133)

```python
def val_dataloader(**kwargs):
```
