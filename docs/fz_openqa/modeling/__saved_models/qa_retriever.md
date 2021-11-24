# QaRetriever

> Auto-generated documentation for [fz_openqa.modeling.__saved_models.qa_retriever](blob/master/fz_openqa/modeling/__saved_models/qa_retriever.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Modeling](../index.md#modeling) / [Saved Models](index.md#saved-models) / QaRetriever
    - [QaRetriever](#qaretriever)
        - [QaRetriever().check_input_features_with_key](#qaretrievercheck_input_features_with_key)
        - [QaRetriever().forward](#qaretrieverforward)
        - [QaRetriever().predict_step](#qaretrieverpredict_step)

## QaRetriever

[[find in source code]](blob/master/fz_openqa/modeling/__saved_models/qa_retriever.py#L19)

```python
class QaRetriever(Module):
    def __init__(
        tokenizer: PreTrainedTokenizerFast,
        bert: Union[BertPreTrainedModel, DictConfig],
        evaluator: Union[Module, DictConfig],
        corpus: Optional[Union[CorpusDataModule, DictConfig]] = None,
        hidden_size: int = 256,
        dropout: float = 0,
        **kwargs,
    ):
```

A Dense retriever model.

#### Attributes

- `model_logging_prefix` - prefix for the logged metrics: `'retriever/'`
- `pbar_metrics` - metrics to display: `['train/loss', 'train/Accuracy', 'validation/Accuracy']`

### QaRetriever().check_input_features_with_key

[[find in source code]](blob/master/fz_openqa/modeling/__saved_models/qa_retriever.py#L110)

```python
def check_input_features_with_key(batch, key):
```

### QaRetriever().forward

[[find in source code]](blob/master/fz_openqa/modeling/__saved_models/qa_retriever.py#L69)

```python
def forward(
    batch: Dict[str, Tensor],
    batch_idx: int = 0,
    dataloader_idx: int = None,
    model_key: str = 'document',
    **kwargs,
) -> torch.Tensor:
```

compute the document and question representations based on the argument `model_key`.
Return the representation of `x`: BERT(x)_CLS of shape [batch_size, h]

Future work:
Implement ColBERT interaction model, in that case the output
will be `conv(BERT(x))` of shape [batch_size, T, h]

### QaRetriever().predict_step

[[find in source code]](blob/master/fz_openqa/modeling/__saved_models/qa_retriever.py#L99)

```python
def predict_step(
    batch: Any,
    batch_idx: int,
    dataloader_idx: Optional[int] = None,
) -> Any:
```
