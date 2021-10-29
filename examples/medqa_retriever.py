import datasets
import numpy as np
import rich
from rich.progress import track

from fz_openqa.datamodules.corpus_dm import MedQaCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import PrintBatch
from fz_openqa.datamodules.pipes import SearchCorpus
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.datamodules.pipes import UpdateWith
from fz_openqa.datamodules.pipes.concat_answer_options import (
    ConcatQuestionAnswerOption,
)
from fz_openqa.datamodules.pipes.nesting import AsFlatten
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.train_utils import setup_safe_env

datasets.set_caching_enabled(True)
setup_safe_env()

es_body = {
    "settings": {
        "analysis": {
            "analyzer": {
                "my_analyzer": {
                    "tokenizer": "standard",
                    "filter": ["snow"],
                }
            },
            "filter": {
                "snow": {
                    "type": "snowball",
                    "language": "English",
                }
            },
        }
    }
}

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path="bert-base-cased"
)

# load the corpus object
corpus = MedQaCorpusDataModule(
    tokenizer=tokenizer,
    to_sentences=True,
    index=ElasticSearchIndex(
        index_key="document.row_idx",
        text_key="document.text",
        query_key="question.metamap",
        num_proc=4,
        filter_mode=None,
        text_cleaner=TextFormatter(remove_symbols=True),
        es_body=es_body,
    ),
    verbose=False,
    num_proc=4,
    use_subset=False,
)

# load the QA dataset
dm = MedQaDataModule(
    tokenizer=tokenizer,
    train_batch_size=100,
    num_proc=4,
    num_workers=4,
    use_subset=False,
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

concat_pipe = ConcatQuestionAnswerOption()
select_fields = FilterKeys(lambda key: key == "question.metamap")
search_index = SearchCorpus(corpus=corpus, k=10)
flatten_and_search = AsFlatten(search_index)

pipe = UpdateWith(Sequential(concat_pipe, select_fields, flatten_and_search))
printer = PrintBatch()


def randargmax(retrieval_scores: np.array) -> np.array:
    """a random tie-breaking argmax"""
    return np.random.choice(
        np.flatnonzero(retrieval_scores == retrieval_scores.max())
    )


total = (
    len(dm.dataset["train"])
    + len(dm.dataset["validation"])
    + len(dm.dataset["test"])
)
num_corrects = 0
for batch in track(
    dm.train_dataloader(), description="Iterating through the dataset..."
):
    # pprint(batch)

    # 1 do a pipe to concat question + answer option
    batch = pipe(batch)
    # print(len(batch["document.retrieval_score"]))
    # print(len(batch["answer.target"]))
    # print(get_separator())
    predictions = [
        np.argmax(np.sum(question, axis=1))
        for question in batch["document.retrieval_score"]
    ]

    # predictions =[
    #    randargmax(np.sum(question, axis=1)
    #              ) for question in batch['document.retrieval_score']
    # ]

    # print(predictions)
    # print(batch["answer.target"])
    # print(get_separator())

    num_corrects += np.count_nonzero(
        predictions - batch["answer.target"].numpy() == 0
    )

print(
    "accuracy is: {} / {} = {:.1f}%".format(
        num_corrects, total, num_corrects * 100.0 / total
    )
)

# Accuracy metrics
# Query : "question.text" , Argmax : "No randomness"
# >> accuracy is: 2836 / 12723 = 22.3%
# Query : "question.metamap" , Argmax : "No randomness"
# >> accuracy is: 3157 / 12723 = 24.8%
# # Query : "question.metamap" , Argmax : "Random when tie-break"
# # >> accuracy is: 3157 / 12723 = 24.8%