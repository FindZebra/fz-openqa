# Text

> Auto-generated documentation for [fz_openqa.datamodules.pipelines.preprocessing.text](blob/master/fz_openqa/datamodules/pipelines/preprocessing/text.py) module.

- [Fz-openqa](../../../../README.md#fz-openqa-index) / [Modules](../../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../../index.md#fz-openqa) / [Datamodules](../../index.md#datamodules) / [Pipelines](../index.md#pipelines) / [Preprocessing](index.md#preprocessing) / Text
    - [FormatAndTokenize](#formatandtokenize)

## FormatAndTokenize

[[find in source code]](blob/master/fz_openqa/datamodules/pipelines/preprocessing/text.py#L19)

```python
class FormatAndTokenize(Sequential):
    def __init__(
        prefix: str,
        text_formatter: TextFormatter,
        tokenizer: PreTrainedTokenizerFast,
        add_encoding_tokens: bool,
        max_length: Optional[int],
        spec_tokens: List,
        shape: Optional[List[int]],
        return_token_type_ids: bool = False,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        field: str = 'text',
    ):
```

create a pipeline to process the raw text:
    1. format text
    2. add special tokens
    3. tokenize
