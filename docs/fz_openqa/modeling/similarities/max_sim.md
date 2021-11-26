# MaxSim

> Auto-generated documentation for [fz_openqa.modeling.similarities.max_sim](blob/master/fz_openqa/modeling/similarities/max_sim.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Modeling](../index.md#modeling) / [Similarities](index.md#similarities) / MaxSim
    - [MaxSim](#maxsim)
        - [MaxSim().\_\_call\_\_](#maxsim__call__)

## MaxSim

[[find in source code]](blob/master/fz_openqa/modeling/similarities/max_sim.py#L7)

```python
class MaxSim(Similarity):
    def __init__(similarity_metric=str):
```

### MaxSim().\_\_call\_\_

[[find in source code]](blob/master/fz_openqa/modeling/similarities/max_sim.py#L12)

```python
def __call__(query: Tensor, document: Tensor) -> Tensor:
```

Compute the similarity between a batch of N queries and
M elements. Returns a similarity matrix S of shape N x M
where S_ij is the similarity between the query[i] and document[i]
`query` and `document` are global representation (shape [bs, h]).
If `query` or `document` is tensor of dimension 3, take the
first element (corresponds to the CLS token)
