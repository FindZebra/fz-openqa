# MultipleChoiceQA

> Auto-generated documentation for [fz_openqa.modeling.__saved_models.multiple_choice_qa](blob/master/fz_openqa/modeling/__saved_models/multiple_choice_qa.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Modeling](../index.md#modeling) / [Saved Models](index.md#saved-models) / MultipleChoiceQA
    - [MultipleChoiceQA](#multiplechoiceqa)
        - [MultipleChoiceQA().configure_optimizers](#multiplechoiceqaconfigure_optimizers)
        - [MultipleChoiceQA().predict_step](#multiplechoiceqapredict_step)

## MultipleChoiceQA

[[find in source code]](blob/master/fz_openqa/modeling/__saved_models/multiple_choice_qa.py#L28)

```python
class MultipleChoiceQA(Module):
    def __init__(
        tokenizer: PreTrainedTokenizerFast,
        bert: Union[BertPreTrainedModel, DictConfig],
        reader: Union[DictConfig, MultipleChoiceQAReader],
        retriever: Union[DictConfig, QaRetriever],
        evaluator: Union[Module, DictConfig],
        corpus: Optional[Union[CorpusDataModule, DictConfig]] = None,
        end_to_end_evaluation: bool = False,
        **kwargs,
    ):
```

An end-to-end multiple choice openQA model with:
* a dense retriever
* a multiple-choice reader

### MultipleChoiceQA().configure_optimizers

[[find in source code]](blob/master/fz_openqa/modeling/__saved_models/multiple_choice_qa.py#L205)

```python
def configure_optimizers():
```

Choose what optimizers and learning-rate schedulers to use in your optimization.
Normally you'd need one. But in the case of GANs or similar you might have multiple.
See examples here:
    https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

### MultipleChoiceQA().predict_step

[[find in source code]](blob/master/fz_openqa/modeling/__saved_models/multiple_choice_qa.py#L124)

```python
def predict_step(
    batch: Any,
    batch_idx: int,
    dataloader_idx: Optional[int] = None,
) -> Any:
```
