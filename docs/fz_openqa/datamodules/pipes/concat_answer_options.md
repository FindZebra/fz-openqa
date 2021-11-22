# Concat Answer Options

> Auto-generated documentation for [fz_openqa.datamodules.pipes.concat_answer_options](blob/master/fz_openqa/datamodules/pipes/concat_answer_options.py) module.

- [Fz-openqa](../../../README.md#fz-openqa-index) / [Modules](../../../MODULES.md#fz-openqa-modules) / [Fz Openqa](../../index.md#fz-openqa) / [Datamodules](../index.md#datamodules) / [Pipes](index.md#pipes) / Concat Answer Options
    - [ConcatQuestionAnswerOption](#concatquestionansweroption)

## ConcatQuestionAnswerOption

[[find in source code]](blob/master/fz_openqa/datamodules/pipes/concat_answer_options.py#L5)

```python
class ConcatQuestionAnswerOption(Pipe):
    def __init__(
        question_key: str = 'question.metamap',
        answer_key: str = 'answer.text',
        **kwargs,
    ):
```

Concat question text with answer text
