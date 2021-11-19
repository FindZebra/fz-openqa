# TextFormatter

> Auto-generated documentation for [fz_openqa.datamodules.pipes.text_formatter](blob/master/fz_openqa/datamodules/pipes/text_formatter.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / TextFormatter
    - [TextFormatter](#textformatter)
        - [TextFormatter().clean](#textformatterclean)

## TextFormatter

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/text_formatter.py#L12)

```python
class TextFormatter(Pipe):
    def __init__(
        text_key: Optional[str] = None,
        remove_linebreaks: bool = True,
        remove_ref: bool = True,
        lowercase: bool = False,
        aggressive_cleaning: bool = False,
        remove_symbols: bool = False,
        **kwargs,
    ):
```

clean the text field (lower case, apply regex to remove special characters)

### TextFormatter().clean

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/text_formatter.py#L34)

```python
def clean(text: str) -> str:
```
