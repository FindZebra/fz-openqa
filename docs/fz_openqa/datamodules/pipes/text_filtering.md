# Text Filtering

> Auto-generated documentation for [fz_openqa.datamodules.pipes.text_filtering](blob/master/fz_openqa/datamodules/pipes/text_filtering.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Text Filtering
    - [MetaMapFilter](#metamapfilter)
    - [SciSpaCyFilter](#scispacyfilter)
        - [SciSpaCyFilter().filter_one](#scispacyfilterfilter_one)
    - [StopWordsFilter](#stopwordsfilter)
        - [StopWordsFilter().filter_one](#stopwordsfilterfilter_one)
    - [TextFilter](#textfilter)
        - [TextFilter().filter_one](#textfilterfilter_one)

## MetaMapFilter

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/text_filtering.py#L77)

```python
class MetaMapFilter(TextFilter):
    def __init__(**kwargs):
```

Build a Pipe to return a string of unique entities recognized
based on offline processed MetaMap heuristic

#### Arguments

- `MetaMapList` - A list of recognised entities inferred from the question query
- `Question` - query to be replaced by MetaMapList

#### See also

- [TextFilter](#textfilter)

## SciSpaCyFilter

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/text_filtering.py#L34)

```python
class SciSpaCyFilter(TextFilter):
    def __init__(spacy_model=None, **kwargs):
```

Build a Pipe to return a tuple of displacy image of named or
unnamed word entities and a set of unique entities recognized
based on scispacy model in use

#### Arguments

- `model` - A pretrained model from spaCy or scispaCy
- `document` - text data to be analysed

#### See also

- [TextFilter](#textfilter)

### SciSpaCyFilter().filter_one

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/text_filtering.py#L62)

```python
def filter_one(text: str) -> str:
```

## StopWordsFilter

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/text_filtering.py#L27)

```python
class StopWordsFilter(TextFilter):
```

Example: remove stop words from string

#### See also

- [TextFilter](#textfilter)

### StopWordsFilter().filter_one

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/text_filtering.py#L30)

```python
def filter_one(text: str) -> str:
```

## TextFilter

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/text_filtering.py#L12)

```python
class TextFilter(Pipe):
    def __init__(text_key: str, query_key, **kwargs):
```

### TextFilter().filter_one

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/text_filtering.py#L23)

```python
def filter_one(text: str) -> str:
```
