import cProfile
import logging
import pstats
from timeit import Timer

import datasets
import numpy as np
import rich
from rich.logging import RichHandler

from fz_openqa.datamodules.__old.corpus_dm import MedQaCorpusDataModule
from fz_openqa.datamodules.__old.meqa_dm import MedQaDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.train_utils import setup_safe_env

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.CRITICAL,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)

datasets.set_caching_enabled(True)
setup_safe_env()

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path="bert-base-cased"
)

text_formatter = TextFormatter(lowercase=True)

# load the corpus object
corpus = MedQaCorpusDataModule(
    tokenizer=tokenizer,
    text_formatter=text_formatter,
    index=ElasticSearchIndex(
        index_key="document.row_idx",
        text_key="document.text",
        query_key="question.text",
        num_proc=4,
        filter_mode=None,
    ),
    verbose=False,
    num_proc=4,
    use_subset=False,
)

# load the QA dataset
dm = MedQaDataModule(
    tokenizer=tokenizer,
    text_formatter=text_formatter,
    train_batch_size=10,
    num_proc=4,
    num_workers=5,
    use_subset=True,
    verbose=True,
    corpus=corpus,
    # retrieve 100 documents for each question
    n_retrieved_documents=1000,
    # keep only one positive doc
    max_pos_docs=10,
    # keep only 10 docs (1 pos + 9 neg)
    n_documents=100,
    # simple exact match
    relevance_classifier=ExactMatch(interpretable=True),
)

# prepare both the QA dataset and the corpus
dm.subset_size = [1000, 100, 100]
dm.prepare_data()
dm.setup()


class GetBatch:
    def __init__(self, loader):
        self.it_loader = iter(loader)

    def __call__(self):
        return next(self.it_loader)


get_batch = GetBatch(dm.train_dataloader())

profiler = cProfile.Profile()
profiler.enable()
times = Timer(get_batch).repeat(20, 1)
profiler.disable()
stats = pstats.Stats(profiler).sort_stats("time")
stats.print_stats(20)
print(get_separator())
rich.print(
    f">> duration={np.mean(times):.3f}s/batch (std={np.std(times):.3f}s)"
)
# fetch docs in collate: >> duration=0.142s/batch (std=0.293s)
# fetch docs in __getitem__: >> duration=1.351s/batch (std=2.645s)make
