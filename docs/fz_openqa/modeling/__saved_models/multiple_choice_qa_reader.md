# MultipleChoiceQAReader

> Auto-generated documentation for [fz_openqa.modeling.__saved_models.multiple_choice_qa_reader](blob/master/fz_openqa/modeling/__saved_models/multiple_choice_qa_reader.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Modeling](../index.md#modeling) / [Saved Models](index.md#saved-models) / MultipleChoiceQAReader
    - [MultipleChoiceQAReader](#multiplechoiceqareader)
        - [MultipleChoiceQAReader().concat_questions_and_documents](#multiplechoiceqareaderconcat_questions_and_documents)
        - [MultipleChoiceQAReader.expand_and_flatten](#multiplechoiceqareaderexpand_and_flatten)
        - [MultipleChoiceQAReader().forward](#multiplechoiceqareaderforward)

## MultipleChoiceQAReader

[[find in source code]](blob/master/fz_openqa/modeling/__saved_models/multiple_choice_qa_reader.py#L19)

```python
class MultipleChoiceQAReader(Module):
    def __init__(
        tokenizer: PreTrainedTokenizerFast,
        bert: Union[BertPreTrainedModel, DictConfig],
        evaluator: Union[Module, DictConfig],
        hidden_size: int = 256,
        dropout: float = 0,
        **kwargs,
    ):
```

A multiple-choice reader model.

### MultipleChoiceQAReader().concat_questions_and_documents

[[find in source code]](blob/master/fz_openqa/modeling/__saved_models/multiple_choice_qa_reader.py#L129)

```python
def concat_questions_and_documents(batch):
```

concatenate questions and documents such that
there is no padding between Q and D

### MultipleChoiceQAReader.expand_and_flatten

[[find in source code]](blob/master/fz_openqa/modeling/__saved_models/multiple_choice_qa_reader.py#L151)

```python
@staticmethod
def expand_and_flatten(x: Tensor, n: int) -> Tensor:
```

Expand a tensor of shape [bs, *dims] as
[bs, n, *dims] and flatten to [bs * n, *dims]

### MultipleChoiceQAReader().forward

[[find in source code]](blob/master/fz_openqa/modeling/__saved_models/multiple_choice_qa_reader.py#L69)

```python
def forward(
    batch: Dict[str, Tensor],
    **kwargs,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
```

Compute the multiple choice answer model:
 * log p(a_i | q, e)
where:
*  log p(a_i| q, e) = Sim(BERT_CLS([q;d]), BERT_CLS(a_i))

Future work:
This is a very simple model, it will be improved in the next iterations.
We can improve it either using a full interaction model
* BERT_CLS([q;d;a_i])
which requires roughly `N_q` times more GPU memory
Or using a local interaction model `Sim` much similar to ColBERT.

Input data:
batch = {
'document.input_ids': [batch_size * n_docs, L_d]
'question.input_ids': [batch_size * n_docs, L_q]
'answer.input_ids': [batch_size, N_q, L_q]
}
