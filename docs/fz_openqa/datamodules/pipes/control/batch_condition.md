# BatchCondition

> Auto-generated documentation for [fz_openqa.datamodules.pipes.control.batch_condition](blob/master/fz_openqa/datamodules/pipes/control/batch_condition.py) module.

- [Fz-openqa](../../../../README.md#fz-openqa-index) / [Modules](../../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../../index.md#fz-openqa) / [Datamodules](../../index.md#datamodules) / [Pipes](../index.md#pipes) / [Control](index.md#control) / BatchCondition
    - [AllValuesOfType](#allvaluesoftype)
    - [BatchCondition](#batchcondition)
    - [HasKeyWithPrefix](#haskeywithprefix)
    - [HasKeys](#haskeys)

## AllValuesOfType

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/control/batch_condition.py#L65)

```python
class AllValuesOfType(BatchCondition):
    def __init__(cls: type, **kwargs):
```

Check if all batch values are of the specified type

#### See also

- [BatchCondition](#batchcondition)

## BatchCondition

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/control/batch_condition.py#L9)

```python
class BatchCondition(Condition):
```

Condition operating on the batch level.

#### See also

- [Condition](condition.md#condition)

## HasKeyWithPrefix

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/control/batch_condition.py#L37)

```python
class HasKeyWithPrefix(BatchCondition):
    def __init__(prefix: str, **kwargs):
```

Test if the batch contains at least one key with the specified prefix

#### See also

- [BatchCondition](#batchcondition)

## HasKeys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/control/batch_condition.py#L51)

```python
class HasKeys(BatchCondition):
    def __init__(keys: List[str], **kwargs):
```

Test if the batch contains all the required keys

#### See also

- [BatchCondition](#batchcondition)
