# Medqa Retriever

> Auto-generated documentation for [examples.medqa_retriever](blob/master/examples/medqa_retriever.py) module.

- [Fz-openqa](../README.md#fz-openqa-index) / [Modules](../MODULES.md#fz-openqa-modules) / [Examples](index.md#examples) / Medqa Retriever
    - [randargmax](#randargmax)

#### Attributes

- `corpus` - load the corpus object: `MedQaCorpusBuilder(tokenizer=tokenizer, to_sent...`
- `dm` - load the QA dataset: `QaBuilder(tokenizer=tokenizer, train_batch_s...`
- `batch` - 1 do a pipe to concat question + answer option: `pipe(batch)`

## randargmax

[[find in source code]](blob/master/examples/medqa_retriever.py#L77)

```python
def randargmax(proposal_scores: np.array) -> np.array:
```

a random tie-breaking argmax
