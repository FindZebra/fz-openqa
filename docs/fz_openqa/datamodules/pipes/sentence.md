# Sentence

> Auto-generated documentation for [fz_openqa.datamodules.pipes.sentence](blob/master/fz_openqa/datamodules/pipes/sentence.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Sentence
    - [GenerateSentences](#generatesentences)
        - [GenerateSentences.generate_sentences](#generatesentencesgenerate_sentences)

## GenerateSentences

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/sentence.py#L11)

```python
class GenerateSentences(Pipe):
    def __init__(
        delimiter: Optional[str] = '.',
        required_keys: Optional[List[str]] = None,
        **kwargs,
    ):
```

A pipe to Extract sentences from a corpus of text.

replication of:
https://github.com/jind11/MedQA/blob/master/IR/scripts/insert_text_to_elasticsearch.py

### GenerateSentences.generate_sentences

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/sentence.py#L33)

```python
@staticmethod
def generate_sentences(
    examples: Dict[str, List[Any]],
    required_keys: List[str],
    delimiter: str,
) -> Batch:
```

This functions generates the sentences for each corpus article.

return:
    - output: Batch of data (`document.text` + `idx` (document id))

#### See also

- [Batch](../../utils/datastruct.md#batch)
