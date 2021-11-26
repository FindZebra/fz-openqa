# ReaderMultipleChoice

> Auto-generated documentation for [fz_openqa.modeling.modules.reader_multiple_choice](blob/master/fz_openqa/modeling/modules/reader_multiple_choice.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Modeling](../index.md#modeling) / [Modules](index.md#modules) / ReaderMultipleChoice
    - [ReaderMultipleChoice](#readermultiplechoice)
        - [ReaderMultipleChoice().compute_metrics](#readermultiplechoicecompute_metrics)
        - [ReaderMultipleChoice().reset_metrics](#readermultiplechoicereset_metrics)
        - [ReaderMultipleChoice().update_metrics](#readermultiplechoiceupdate_metrics)

## ReaderMultipleChoice

[[find in source code]](blob/master/fz_openqa/modeling/modules/reader_multiple_choice.py#L24)

```python
class ReaderMultipleChoice(Module):
```

#### Attributes

- `pbar_metrics` - metrics to display: `['train/reader/Accuracy', 'validation/reader/Ac...`

### ReaderMultipleChoice().compute_metrics

[[find in source code]](blob/master/fz_openqa/modeling/modules/reader_multiple_choice.py#L261)

```python
def compute_metrics(split: Optional[Split] = None) -> Batch:
```

Compute the metrics for the given `split` else compute the metrics for all splits.
The metrics are return after computation.

#### See also

- [Batch](../../utils/datastruct.md#batch)

### ReaderMultipleChoice().reset_metrics

[[find in source code]](blob/master/fz_openqa/modeling/modules/reader_multiple_choice.py#L253)

```python
def reset_metrics(split: Optional[Split] = None) -> None:
```

Reset the metrics corresponding to `split` if provided, else
reset all the metrics.

### ReaderMultipleChoice().update_metrics

[[find in source code]](blob/master/fz_openqa/modeling/modules/reader_multiple_choice.py#L240)

```python
def update_metrics(output: Batch, split: Split) -> None:
```

update the metrics of the given split.

#### See also

- [Batch](../../utils/datastruct.md#batch)
