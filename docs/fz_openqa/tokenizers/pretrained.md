# Pretrained

> Auto-generated documentation for [fz_openqa.tokenizers.pretrained](blob/master/fz_openqa/tokenizers/pretrained.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Tokenizers](index.md#tokenizers) / Pretrained
    - [init_pretrained_tokenizer](#init_pretrained_tokenizer)
    - [test_special_token_encoding](#test_special_token_encoding)

## init_pretrained_tokenizer

[[find in source code]](blob/master/fz_openqa/tokenizers/pretrained.py#L15)

```python
def init_pretrained_tokenizer(
    pretrained_model_name_or_path: str,
    **kwargs,
) -> PreTrainedTokenizerFast:
```

Load a HuggingFace Pretrained Tokenizer and add the special tokens.

## test_special_token_encoding

[[find in source code]](blob/master/fz_openqa/tokenizers/pretrained.py#L26)

```python
def test_special_token_encoding(tokenizer):
```
