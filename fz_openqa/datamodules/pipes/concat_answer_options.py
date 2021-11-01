from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch


class ConcatQuestionAnswerOption(Pipe):
    """Concat question text with answer text"""

    def __init__(
        self,
        *,
        question_key: str = "question.metamap",
        answer_key: str = "answer.text",
        **kwargs,
    ):
        super(ConcatQuestionAnswerOption, self).__init__(**kwargs)
        self.question_key = question_key
        self.answer_key = answer_key

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        questions = batch[self.question_key]  # [bs,]
        answers = batch[self.answer_key]  # [bs, n_options]

        def _concat(q: str, a: str):
            return f"{q}, {a}"

        batch[self.question_key] = [
            [_concat(q, a) for a in a_options]
            for q, a_options in zip(questions, answers)
        ]

        return batch
