# Passage

> Auto-generated documentation for [fz_openqa.datamodules.pipes.passage](blob/master/fz_openqa/datamodules/pipes/passage.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Passage
    - [GeneratePassages](#generatepassages)
        - [GeneratePassages.extract_passage_text_from_doc](#generatepassagesextract_passage_text_from_doc)
        - [GeneratePassages.generate_passages_for_all_keys](#generatepassagesgenerate_passages_for_all_keys)
        - [GeneratePassages().output_keys](#generatepassagesoutput_keys)
    - [gen_passages](#gen_passages)

## GeneratePassages

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/passage.py#L14)

```python
class GeneratePassages(Pipe):
    def __init__(
        size: int,
        stride: int,
        start_tokens: List[int],
        end_tokens: List[int],
        pad_token_id: int,
        verbose: bool = True,
        **kwargs,
    ):
```

A pipe to Extract passages from a batch of documents.

### GeneratePassages.extract_passage_text_from_doc

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/passage.py#L134)

```python
@staticmethod
def extract_passage_text_from_doc(
    document: str,
    offset_mapping: List[Tuple[int, int]],
) -> str:
```

Extract the text passage from the original document
given the offset mapping of the passage

### GeneratePassages.generate_passages_for_all_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/passage.py#L81)

```python
@staticmethod
def generate_passages_for_all_keys(
    examples: Dict[str, List[Any]],
    keys: List[str],
    args: Dict[str, Dict[str, Any]],
) -> Tuple[List[int], Batch]:
```

This functions generate the passages for each attribute in `keys`,
 the `arg` dictionary must contain an entry for all `keys`.
 The first pass is used to store the document/example indexes
and compute the `passage_mask`.

The passage mask is used for segmentation, and is optional for this project.
In this context, all tokens are attributed to a single passage,
although they appear in multiple passages (strides).
The passage mask indicates if a token is attributed to this specific passage.

return:
    - indexes: index of the parent example for each passage
    - output: Batch of data for all keys + `idx` (document id) and `passage_mask`

### GeneratePassages().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/passage.py#L56)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```

## gen_passages

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/passage.py#L144)

```python
def gen_passages(
    sequence: List[Any],
    size: int,
    stride: int,
    start_tokens: Optional[List[Any]] = None,
    end_tokens: Optional[List[Any]] = None,
    pad_token: Optional[Any] = None,
    return_mask: bool = True,
) -> Iterable[Union[List[int], Tuple[List[int], List[Any]]]]:
```

Generate overlapping windows with the corresponding
masking such that each token appears only in one window.
