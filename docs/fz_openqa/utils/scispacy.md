# Scispacy

> Auto-generated documentation for [fz_openqa.utils.scispacy](blob/master/fz_openqa/utils/scispacy.py) module.

- [Fz-openqa](../../README.md#fz-openqa-index) / [Modules](../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../index.md#fz-openqa) / [Utils](index.md#utils) / Scispacy
    - [display_entities_pipe](#display_entities_pipe)

## display_entities_pipe

[[find in source code]](blob/master/fz_openqa/utils/scispacy.py#L6)

```python
def display_entities_pipe(model, document):
```

Build a Pipe to return a tuple of displacy image of named or unnamed
word entities and a set of unique entities recognized based on scispacy
model in use

#### Arguments

- `model` - A pretrained model from spaCy or scispaCy
- `document` - text data to be analysed
