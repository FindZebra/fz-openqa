# Relevance

> Auto-generated documentation for [fz_openqa.datamodules.pipes.relevance](blob/master/fz_openqa/datamodules/pipes/relevance.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Relevance
    - [AliasBasedMatch](#aliasbasedmatch)
        - [AliasBasedMatch().\_\_getstate\_\_](#aliasbasedmatch__getstate__)
        - [AliasBasedMatch().detect_acronym](#aliasbasedmatchdetect_acronym)
        - [AliasBasedMatch().dill_inspect](#aliasbasedmatchdill_inspect)
        - [AliasBasedMatch().extract_aliases](#aliasbasedmatchextract_aliases)
        - [AliasBasedMatch().extract_and_filters_entities](#aliasbasedmatchextract_and_filters_entities)
        - [AliasBasedMatch().fingerprint](#aliasbasedmatchfingerprint)
        - [AliasBasedMatch().get_linked_entities](#aliasbasedmatchget_linked_entities)
    - [ExactMatch](#exactmatch)
    - [LinkedEntity](#linkedentity)
    - [Pair](#pair)
    - [RelevanceClassifier](#relevanceclassifier)
        - [RelevanceClassifier().classify](#relevanceclassifierclassify)
        - [RelevanceClassifier().classify_and_interpret](#relevanceclassifierclassify_and_interpret)
        - [RelevanceClassifier().output_keys](#relevanceclassifieroutput_keys)
        - [RelevanceClassifier().preprocess](#relevanceclassifierpreprocess)
    - [ScispaCyMatch](#scispacymatch)
        - [ScispaCyMatch().preprocess](#scispacymatchpreprocess)
    - [find_all](#find_all)
    - [find_one](#find_one)

## AliasBasedMatch

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L215)

```python
class AliasBasedMatch(RelevanceClassifier):
    def __init__(
        filter_acronyms: Optional[bool] = True,
        model_name: Optional[str] = 'en_core_sci_lg',
        linker_name: str = 'umls',
        lazy_setup: bool = True,
        spacy_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
```

#### See also

- [RelevanceClassifier](#relevanceclassifier)

### AliasBasedMatch().\_\_getstate\_\_

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L248)

```python
def __getstate__():
```

this method is called when attempting pickling

### AliasBasedMatch().detect_acronym

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L315)

```python
def detect_acronym(alias: str) -> bool:
```

returns true if accronym is found in string
    example: "AbIA AoP U.S.A. USA"

### AliasBasedMatch().dill_inspect

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L262)

```python
def dill_inspect(reduce=True) -> Dict:
```

### AliasBasedMatch().extract_aliases

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L343)

```python
def extract_aliases(linked_entities: Iterable[LinkedEntity]) -> Iterable[str]:
```

### AliasBasedMatch().extract_and_filters_entities

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L327)

```python
def extract_and_filters_entities(doc: Doc) -> Iterable[str]:
```

### AliasBasedMatch().fingerprint

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L259)

```python
def fingerprint() -> Any:
```

### AliasBasedMatch().get_linked_entities

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L299)

```python
def get_linked_entities(entity: Entity) -> Iterable[LinkedEntity]:
```

## ExactMatch

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L198)

```python
class ExactMatch(RelevanceClassifier):
```

Match the lower-cased answer string in the document.

#### See also

- [RelevanceClassifier](#relevanceclassifier)

## LinkedEntity

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L34)

```python
dataclass
class LinkedEntity():
```


#### See also

- [AliasBasedMatch](#aliasbasedmatch)

```python
def preprocess(pairs: Iterable[Pair]) -> Iterable[Pair]:
```

Generate the field `pair.answer["aliases"]`

## Pair

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L41)

```python
dataclass
class Pair():
```

## RelevanceClassifier

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L107)

```python
class RelevanceClassifier(Pipe):
    def __init__(
        answer_prefix: str = 'answer.',
        document_prefix: str = 'document.',
        output_key: str = 'document.match_score',
        interpretable: bool = False,
        interpretation_key: str = 'document.match_on',
        id='relevance-classifier',
        **kwargs,
    ):
```

### RelevanceClassifier().classify

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L137)

```python
def classify(pair: Pair) -> int:
```

#### See also

- [Pair](#pair)

### RelevanceClassifier().classify_and_interpret

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L141)

```python
def classify_and_interpret(pair: Pair) -> Tuple[int, List[str]]:
```

#### See also

- [Pair](#pair)

### RelevanceClassifier().output_keys

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L125)

```python
def output_keys(input_keys: List[str]) -> List[str]:
```

### RelevanceClassifier().preprocess

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L145)

```python
def preprocess(pairs: Iterable[Pair]) -> Iterable[Pair]:
```

Preprocessing allows transforming all the pairs,
potentially in batch mode.

## ScispaCyMatch

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L391)

```python
class ScispaCyMatch(AliasBasedMatch):
    def __init__(**kwargs):
```

#### See also

- [AliasBasedMatch](#aliasbasedmatch)

### ScispaCyMatch().preprocess

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L395)

```python
def preprocess(pairs: Iterable[Pair]) -> Iterable[Pair]:
```

Generate the field `pair.answer["aliases"]`

## find_all

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L77)

```python
def find_all(
    text: str,
    queries: Sequence[Any],
    lower_case_queries: bool = True,
) -> List:
```

Find all matching queries in the document.
There are one returned item per match in the document.

## find_one

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/relevance.py#L46)

```python
def find_one(
    text: str,
    queries: Sequence[Any],
    sort_by: Optional[Callable] = None,
) -> bool:
```

check if one of the queries is in the input text
