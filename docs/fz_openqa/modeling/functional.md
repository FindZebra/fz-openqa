# Functional

> Auto-generated documentation for [fz_openqa.modeling.functional](blob/master/fz_openqa/modeling/functional.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Modeling](index.md#modeling) / Functional
    - [count_right_padding](#count_right_padding)
    - [flatten](#flatten)
    - [is_valid_seq_attr](#is_valid_seq_attr)
    - [pad](#pad)
    - [padless_cat](#padless_cat)

## count_right_padding

[[find in source code]](blob/master/fz_openqa/modeling/functional.py#L16)

```python
def count_right_padding(
    x: Union[Sequence[Any], Tensor],
    pad_token: Any,
) -> int:
```

count the number of right padding tokens

## flatten

[[find in source code]](blob/master/fz_openqa/modeling/functional.py#L124)

```python
def flatten(x: Tensor) -> Tensor:
```

## is_valid_seq_attr

[[find in source code]](blob/master/fz_openqa/modeling/functional.py#L46)

```python
def is_valid_seq_attr(x: BatchValue, ref: BatchValue):
```

check that x is a sequence tensor of the same length as ref

#### See also

- [BatchValue](#batchvalue)

## pad

[[find in source code]](blob/master/fz_openqa/modeling/functional.py#L30)

```python
def pad(batch: BatchValue, pad_token: Any, length: Optional[int] = None):
```

pad a sequence of tensors to the same size, and remove the unnecessary right padding

#### See also

- [BatchValue](#batchvalue)

## padless_cat

[[find in source code]](blob/master/fz_openqa/modeling/functional.py#L104)

```python
def padless_cat(
    batches: List[TorchBatch],
    pad_token: Any,
    master_key: str = 'input_ids',
    aux_pad_tokens: Optional[Dict[str, Any]] = None,
) -> TorchBatch:
```

#### See also

- [TorchBatch](#torchbatch)
