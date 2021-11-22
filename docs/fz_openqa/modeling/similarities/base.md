# Base

> Auto-generated documentation for [fz_openqa.modeling.similarities.base](blob/master/fz_openqa/modeling/similarities/base.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Modeling](../index.md#modeling) / [Similarities](index.md#similarities) / Base
    - [GolbalSimilarity](#golbalsimilarity)
        - [GolbalSimilarity().\_\_call\_\_](#golbalsimilarity__call__)
        - [GolbalSimilarity().vec_similarity](#golbalsimilarityvec_similarity)
    - [Similarity](#similarity)
        - [Similarity().\_\_call\_\_](#similarity__call__)

## GolbalSimilarity

[[find in source code]](blob/master/fz_openqa/modeling/similarities/base.py#L14)

```python
class GolbalSimilarity(Similarity):
```

#### See also

- [Similarity](#similarity)

### GolbalSimilarity().\_\_call\_\_

[[find in source code]](blob/master/fz_openqa/modeling/similarities/base.py#L20)

```python
def __call__(query: Tensor, document: Tensor) -> Tensor:
```

Compute the similarity between a batch of N queries and
M elements. Returns a similarity matrix S of shape N x M
where S_ij is the similarity between the query[i] and document[i]
`query` and `document` are global representation (shape [bs, h]).
If `query` or `document` is tensor of dimension 3, take the
first element (corresponds to the CLS token)

### GolbalSimilarity().vec_similarity

[[find in source code]](blob/master/fz_openqa/modeling/similarities/base.py#L15)

```python
def vec_similarity(x: Tensor, y: Tensor) -> Tensor:
```

## Similarity

[[find in source code]](blob/master/fz_openqa/modeling/similarities/base.py#L6)

```python
class Similarity():
```

### Similarity().\_\_call\_\_

[[find in source code]](blob/master/fz_openqa/modeling/similarities/base.py#L7)

```python
def __call__(query: Tensor, document: Tensor) -> Tensor:
```

Compute the similarity between a batch of N queries and
M elements. Returns a similarity matrix S of shape N x M
where S_ij is the similarity between the query[i] and document[i]
