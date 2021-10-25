from typing import Optional

import datasets
import numpy as np
import rich
from rich.progress import track
from tqdm import tqdm

from fz_openqa.datamodules.corpus_dm import MedQaCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import PrintBatch
from fz_openqa.datamodules.pipes import SearchCorpus
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import UpdateWith
from fz_openqa.datamodules.pipes.nesting import AsFlatten
from fz_openqa.datamodules.pipes.nesting import infer_stride
from fz_openqa.datamodules.utils.filter_keys import KeyWithPrefix
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.train_utils import setup_safe_env

datasets.set_caching_enabled(True)
setup_safe_env()

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path="bert-base-cased"
)

# load the corpus object
corpus = MedQaCorpusDataModule(
    tokenizer=tokenizer,
    index=ElasticSearchIndex(
        index_key="document.row_idx",
        text_key="document.text",
        query_key="question.text",
        num_proc=4,
        filter_mode=None,
    ),
    verbose=False,
    num_proc=4,
    use_subset=True,
)

# load the QA dataset
dm = MedQaDataModule(
    tokenizer=tokenizer,
    train_batch_size=100,
    num_proc=4,
    num_workers=4,
    use_subset=True,
    verbose=True,
    corpus=corpus,
    relevance_classifier=ExactMatch(),
    compile_in_setup=False,
)

# prepare both the QA dataset and the corpus
dm.prepare_data()
dm.setup()

print(get_separator())
dm.build_index()
rich.print("[green]>> index is built.")
print(get_separator())


def concat_question_answer_options(
    batch: Batch, query_key: Optional[str] = "question.text"
):
    pass


class ConcatQuestionAnswerOption(Pipe):
    """Concat question text with answer text"""

    def __init__(
        self,
        *,
        question_key: str = "question.text",
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
            return f"{a}, {q}"

        batch[self.question_key] = [
            [_concat(q, a) for a in a_options]
            for q, a_options in zip(questions, answers)
        ]

        return batch


concat_pipe = ConcatQuestionAnswerOption()
select_fields = FilterKeys(lambda key: key == "question.text")
search_index = SearchCorpus(corpus=corpus, k=5)
flatten_and_search = AsFlatten(search_index)

pipe = UpdateWith(Sequential(concat_pipe, select_fields, flatten_and_search))
printer = PrintBatch()

for batch in track(
    dm.train_dataloader(), description="Iterating through the dataset..."
):

    # 1 do a pipe to concat question + answer option
    batch = pipe(batch)

    printer(batch)
    exit()


# print(
#    "accuracy is: {} / {} = {:.1f}%".format(
#        num_corrects, len(answers), num_corrects * 100.0 / len(answers)
#    )
# )
