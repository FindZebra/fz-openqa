from time import time

import datasets
import rich
from rich.progress import track

from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule, MedQaCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.datamodules.pipes.relevance import ExactMatch, MetaMapMatch, SciSpacyMatch
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator, pprint_batch

datasets.set_caching_enabled(True)

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path='bert-base-cased')

# load the corpus object
corpus = FzCorpusDataModule(tokenizer=tokenizer,
                            index=ElasticSearchIndex(index_key="idx",
                                                     text_key="document.text",
                                                     query_key="question.text",
                                                     filter_mode=None,
                                                     verbose=False),
                            verbose=False,
                            num_proc=4,
                            use_subset=True)

# load the QA dataset
dm = MedQaDataModule(tokenizer=tokenizer,
                     num_proc=4,
                     use_subset=True,
                     verbose=True,
                     corpus=corpus,
                     # retrieve 1000 documents for each question
                     n_retrieved_documents=100,
                     # allow any number of positive documents
                     max_pos_docs=int(1e9),
                     # setting `n_documents` to None will effectively use `n_retrieved_documents`
                     n_documents=None,
                     # retrieve the whole training set
                     train_batch_size=10,
                     relevance_classifier=SciSpacyMatch(
                         answer_prefix='answer.',
                         document_prefix='document.',
                         output_key='document.is_positive'
                     ))

# prepare both the QA dataset and the corpus
dm.prepare_data()
dm.setup()

print(get_separator())
dm.build_index()
rich.print(f"[green]>> index is built.")
print(get_separator())

# iterate through the dataset and check the number of positive documents
count = 0
total = 0
n_batches = 0
n_samples = 100
t0 = time()
for batch in track(dm.train_dataloader(),
                   total=min(n_samples // dm.train_batch_size,
                             len(dm.train_dataloader())),
                   description=f"Iterating through the dataset.."):

    at_least_one_positive = batch['document.is_positive'].sum(1) > 0
    count += (at_least_one_positive > 0).float().sum()
    total += at_least_one_positive.shape[0]
    n_batches += 1
    if total > n_samples:
        break

runtime = time() - t0

# display a batch
print(get_separator())
pprint_batch(batch)

# display examples
print(get_separator())
for k in range(3):
    print(get_separator())
    rich.print(f"Question #{k}:[cyan] {batch['question.text'][k]}")
    ans_idx = batch['answer.target'][k]
    rich.print(f"answer: [green]{batch['answer.text'][k][ans_idx]}")
    for m in range(3):
        rich.print(
            f" - rank={m}: score={batch['document.retrieval_score'][k][m]:.2f}, is_positive={batch['document.is_positive'][k][m]}")
        print(batch['document.text'][k][m].strip().replace("\n", ""))

# display prop. of positive documents and runtime
print(get_separator())
rich.print(
    f">> Processing speed: {runtime / n_batches:.3f}s/batch")
rich.print(
    f">> Number of questions with at least one positive document: {count:.0f} ({100. * count / total:.2f}%)")
