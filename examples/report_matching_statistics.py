from time import time
from copy import copy

import datasets
import numpy as np
import pandas as pd
import rich
from rich.progress import track
from rich.status import Status

from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule, MedQaCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.datamodules.pipes.relevance import ExactMatch, MetaMapMatch, ScispaCyMatch
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator, pprint_batch

datasets.set_caching_enabled(True)

FILTER_MODE = None
CORPUS_MODE = "FZ"
SUBSET_MODE = False

t0 = time()

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path='bert-base-cased')

if CORPUS_MODE == "FZ":
    # load the corpus object
    corpus = FzCorpusDataModule(tokenizer=tokenizer,
                                index=ElasticSearchIndex(index_key="idx",
                                                        text_key="document.text",
                                                        query_key="question.text",
                                                        filter_mode=FILTER_MODE,
                                                        verbose=False),
                                verbose=False,
                                num_proc=4,
                                use_subset=True,
                                train_batch_size=3)
elif CORPUS_MODE == "MedQA":
    # load the corpus object
    corpus = MedQaCorpusDataModule(tokenizer=tokenizer,
                                index=ElasticSearchIndex(index_key="idx",
                                                        text_key="document.text",
                                                        query_key="question.text",
                                                        filter_mode=FILTER_MODE,
                                                        verbose=False),
                                verbose=False,
                                num_proc=4,
                                use_subset=True,
                                train_batch_size=3)

# load the QA dataset
dm = MedQaDataModule(tokenizer=tokenizer,
                     num_proc=4,
                     use_subset=SUBSET_MODE,
                     verbose=True,
                     corpus=corpus,
                     # retrieve 100 documents for each question
                     n_documents=1000,
                     # retrieve the whole training set
                     train_batch_size=10)

# prepare both the QA dataset and the corpus
dm.prepare_data()
dm.setup()

print(get_separator())
dm.build_index()
rich.print(f"[green]>> index is built.")
print(get_separator())

with Status("Loading classifiers.."):
    classifiers = [ExactMatch(), MetaMapMatch()]

stats = []
batch_examples = {}

for cls in classifiers:
    count = 0 #number of questions with at least one postive
    total = 0 #number of questions ~12.000
    n_batches = 0 
    n_samples = 1000 #retrieved documents
    t0 = time()
    for batch in track(
            dm.train_dataloader(),
            total=min(n_samples // dm.train_batch_size,
                len(dm.train_dataloader())),
            description=f"Iterating through the datset..."):

        output = cls(copy(batch))

        at_least_one_positive = output['document.is_positive'].sum(1) > 0
        count += (at_least_one_positive > 0).float().sum()
        total += at_least_one_positive.shape[0]
        n_batches += 1
        if total > n_samples:
            break
    
    runtime = time()-t0

    batch_examples[type(cls).__name__] = output

    stats.append([
        type(cls).__name__,
        FILTER_MODE,
        "FZ",
        total,
        count.item(),
        n_batches,
        n_samples,
        runtime,
    ])

stats_table = pd.DataFrame(
    stats, 
    columns=[
    'Classifier',
    'Filter',
    'Corpus',
    'Questions', 
    'Positives',
    'Batches',
    'Retrv docs',
    'runtime (s)'])

#print output examples for each classifier
# for key in batch_examples.keys():
#     print(key)
#     for i in range(n_batches):
#         if batch_examples[key]['document.is_positive'][i].sum(1) > 0:
#             rich.print(f"[cyan]Question: {batch_examples[key]['question.text'][i]}")
#             rich.print(f"[red]Answer: {batch_examples[key]['answer.text'][i][batch_examples[key]['answer.target'][i]]}")
#             rich.print(f"[white]Document: {batch_examples[key]['document.text'][i][0]}")
#             print((batch_examples[key]['document.is_positive'][i]))


# print report matching statistics for the classifiers
rich.print(
    f">> Matching statistics for the classifiers:\n {stats_table}")