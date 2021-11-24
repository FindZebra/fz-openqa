# Tokenizer

> Auto-generated documentation for [fz_openqa.datamodules.pipes.tokenizer](blob/master/fz_openqa/datamodules/pipes/tokenizer.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Tokenizer
    - [CleanupPadTokens](#cleanuppadtokens)
        - [CleanupPadTokens().filter_tokens](#cleanuppadtokensfilter_tokens)
    - [TokenizerPipe](#tokenizerpipe)
        - [TokenizerPipe().output_keys](#tokenizerpipeoutput_keys)

## CleanupPadTokens

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/tokenizer.py#L65)

```python
class CleanupPadTokens(Pipe):
    def __init__(tokenizer: PreTrainedTokenizerFast):
```

Remove pad tokens from all input_ids and corresponding attention_mask.
Quick and ugly fix, sorry about that!

### CleanupPadTokens().filter_tokens

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/tokenizer.py#L109)

```python
def filter_tokens(tokens, attn):
```

## TokenizerPipe

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/tokenizer.py#L13)

```python
class TokenizerPipe(Pipe):
    def __init__(
        tokenizer: PreTrainedTokenizerFast,
        drop_columns: bool = True,
        fields: Union[str, List[str]],
        max_length: Optional[int],
        return_token_type_ids: bool = False,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = True,
        **kwargs,
    ):
```

tokenize a batch of data

### TokenizerPipe().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/tokenizer.py#L41)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```
