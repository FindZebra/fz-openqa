# Condition

> Auto-generated documentation for [fz_openqa.datamodules.pipes.control.condition](blob/master/fz_openqa/datamodules/pipes/control/condition.py) module.

- [Fz-openqa](../../../../README.md#fz-openqa-index) / [Modules](../../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../../index.md#fz-openqa) / [Datamodules](../../index.md#datamodules) / [Pipes](../index.md#pipes) / [Control](index.md#control) / Condition
    - [Condition](#condition)
        - [Condition().\_\_call\_\_](#condition__call__)
    - [Contains](#contains)
    - [HasPrefix](#hasprefix)
    - [In](#in)
    - [Not](#not)
    - [Reduce](#reduce)
    - [Static](#static)

## Condition

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/control/condition.py#L10)

```python
class Condition(Component):
```

This class implements a condition for the control pipe.

#### See also

- [Component](../../component.md#component)

### Condition().\_\_call\_\_

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/control/condition.py#L17)

```python
@abc.abstractmethod
def __call__(x: Any, **kwargs) -> bool:
```

Returns True if the input matches the condition.

Parameters
----------
x
    object to be tested.

Returns
-------
bool
    True if the input matches the condition.

## Contains

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/control/condition.py#L35)

```python
class Contains(Condition):
    def __init__(pattern: str, **kwargs):
```

check if the key is in the set of `allowed_keys`

#### See also

- [Condition](#condition)

## HasPrefix

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/control/condition.py#L60)

```python
class HasPrefix(Condition):
    def __init__(prefix: str, **kwargs):
```

check if the key starts with a given prefix

#### See also

- [Condition](#condition)

## In

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/control/condition.py#L46)

```python
class In(Condition):
    def __init__(allowed_values: List[str], **kwargs):
```

check if the key is in the set of `allowed_keys`

#### See also

- [Condition](#condition)

## Not

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/control/condition.py#L91)

```python
class Not(Condition):
    def __init__(condition: Condition, **kwargs):
```

`not` Operator for a condition.

#### See also

- [Condition](#condition)

## Reduce

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/control/condition.py#L74)

```python
class Reduce(Condition):
    def __init__(reduce_op: Callable = all, *conditions: Condition, **kwargs):
```

Reduce multiple conditions into outcome.

#### See also

- [Condition](#condition)

## Static

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/control/condition.py#L105)

```python
class Static(Condition):
    def __init__(cond: bool, **kwargs):
```

Condition with a static boolean outcome.

#### See also

- [Condition](#condition)
