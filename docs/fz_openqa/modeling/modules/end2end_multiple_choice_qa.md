# End2end Multiple Choice Qa

> Auto-generated documentation for [fz_openqa.modeling.modules.end2end_multiple_choice_qa](blob/master/fz_openqa/modeling/modules/end2end_multiple_choice_qa.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Modeling](../index.md#modeling) / [Modules](index.md#modules) / End2end Multiple Choice Qa
    - [EndToEndMultipleChoiceQaMaximumLikelihood](#endtoendmultiplechoiceqamaximumlikelihood)
        - [EndToEndMultipleChoiceQaMaximumLikelihood().compute_metrics](#endtoendmultiplechoiceqamaximumlikelihoodcompute_metrics)
        - [EndToEndMultipleChoiceQaMaximumLikelihood().reset_metrics](#endtoendmultiplechoiceqamaximumlikelihoodreset_metrics)
        - [EndToEndMultipleChoiceQaMaximumLikelihood.retrieve_documents](#endtoendmultiplechoiceqamaximumlikelihoodretrieve_documents)
        - [EndToEndMultipleChoiceQaMaximumLikelihood().step](#endtoendmultiplechoiceqamaximumlikelihoodstep)
        - [EndToEndMultipleChoiceQaMaximumLikelihood().step_end](#endtoendmultiplechoiceqamaximumlikelihoodstep_end)
        - [EndToEndMultipleChoiceQaMaximumLikelihood().update_metrics](#endtoendmultiplechoiceqamaximumlikelihoodupdate_metrics)
    - [argmax_select](#argmax_select)

## EndToEndMultipleChoiceQaMaximumLikelihood

[[find in source code]](blob/master/fz_openqa/modeling/modules/end2end_multiple_choice_qa.py#L24)

```python
class EndToEndMultipleChoiceQaMaximumLikelihood(Module):
    def __init__(n_documents: int, **kwargs):
```

End-to-end evaluation of an OpenQA model (retriever+reader) based on the reader accuracy.
    * Retrieve k documents from the corpus using the retriever model
    * Compute the relevance score for the k-documents using the reader model
    * Predict the answer based on the selected document

### EndToEndMultipleChoiceQaMaximumLikelihood().compute_metrics

[[find in source code]](blob/master/fz_openqa/modeling/modules/end2end_multiple_choice_qa.py#L202)

```python
def compute_metrics(split: Optional[Split] = None) -> Batch:
```

Compute the metrics for the given `split` else compute the metrics for all splits.
The metrics are return after computation.

#### See also

- [Batch](../../utils/datastruct.md#batch)

### EndToEndMultipleChoiceQaMaximumLikelihood().reset_metrics

[[find in source code]](blob/master/fz_openqa/modeling/modules/end2end_multiple_choice_qa.py#L195)

```python
def reset_metrics(split: Optional[Split] = None) -> None:
```

Reset the metrics corresponding to `split` if provided, else
reset all the metrics.

### EndToEndMultipleChoiceQaMaximumLikelihood.retrieve_documents

[[find in source code]](blob/master/fz_openqa/modeling/modules/end2end_multiple_choice_qa.py#L124)

```python
@staticmethod
def retrieve_documents(
    corpus: CorpusDataModule,
    query: Tensor,
    n_docs: int,
) -> Tuple[Batch, int]:
```

Retrieve `n_documents` from the corpus object given the `query`.

### EndToEndMultipleChoiceQaMaximumLikelihood().step

[[find in source code]](blob/master/fz_openqa/modeling/modules/end2end_multiple_choice_qa.py#L54)

```python
def step(
    model: nn.Module,
    batch: Batch,
    split: Split,
    **kwargs: Any,
) -> Batch:
```

Compute the forward pass for the question and the documents.

The input data is assumed to be of shape:
batch = {
'question.input_ids': [batch_size, L_q],
'document.input_ids': [batch_size, n_docs, L_q]

#### See also

- [Batch](../../utils/datastruct.md#batch)

### EndToEndMultipleChoiceQaMaximumLikelihood().step_end

[[find in source code]](blob/master/fz_openqa/modeling/modules/end2end_multiple_choice_qa.py#L159)

```python
def step_end(output: Batch, split: Split) -> Any:
```

Apply a post-processing step to the forward method.
The output is the output of the forward method.

This method is called after the `output` has been gathered
from each device. This method must aggregate the loss across
devices.

torchmetrics update() calls should be placed here.
The output must at least contains the `loss` key.

#### See also

- [Batch](../../utils/datastruct.md#batch)

### EndToEndMultipleChoiceQaMaximumLikelihood().update_metrics

[[find in source code]](blob/master/fz_openqa/modeling/modules/end2end_multiple_choice_qa.py#L188)

```python
def update_metrics(output: Batch, split: Split) -> None:
```

update the metrics of the given split.

#### See also

- [Batch](../../utils/datastruct.md#batch)

## argmax_select

[[find in source code]](blob/master/fz_openqa/modeling/modules/end2end_multiple_choice_qa.py#L210)

```python
def argmax_select(inputs: Tensor, key: Tensor) -> Dict[str, Tensor]:
```

Index all the tensor in the input based on the armax of the key
