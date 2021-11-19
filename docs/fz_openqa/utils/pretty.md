# Pretty

> Auto-generated documentation for [fz_openqa.utils.pretty](blob/master/fz_openqa/utils/pretty.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Utils](index.md#utils) / Pretty
    - [get_separator](#get_separator)
    - [pprint_batch](#pprint_batch)
    - [pretty_decode](#pretty_decode)
    - [repr_batch](#repr_batch)

## get_separator

[[find in source code]](blob/master/fz_openqa/utils/pretty.py#L43)

```python
def get_separator(char='─'):
```

## pprint_batch

[[find in source code]](blob/master/fz_openqa/utils/pretty.py#L48)

```python
def pprint_batch(batch: Batch, header=None):
```

#### See also

- [Batch](datastruct.md#batch)

## pretty_decode

[[find in source code]](blob/master/fz_openqa/utils/pretty.py#L20)

```python
def pretty_decode(
    tokens: Union[Tensor, List[int], np.ndarray],
    tokenizer: PreTrainedTokenizerFast,
    style: str = 'deep_sky_blue3',
    only_text: bool = False,
    **kwargs,
):
```

Pretty print an encoded chunk of text

## repr_batch

[[find in source code]](blob/master/fz_openqa/utils/pretty.py#L60)

```python
def repr_batch(batch: Batch, header=None, rich: bool = False) -> str:
```

#### See also

- [Batch](datastruct.md#batch)
