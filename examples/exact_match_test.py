import datasets
import rich

from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.datamodules.pipes.relevance import ExactMatch, MetaMapMatch, SciSpacyMatch
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator, pprint_batch
from rich.progress import track

datasets.set_caching_enabled(True)

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path='bert-base-cased')

# load the corpus object
corpus = FzCorpusDataModule(tokenizer=tokenizer,
                            index=ElasticSearchIndex(index_key="idx",
                                                     text_key="document.text",
                                                     query_key="question.text",
                                                     filter_mode="scispacy",
                                                     verbose=False),
                            verbose=False,
                            num_proc=1,
                            use_subset=True,
                            train_batch_size=3)

# load the QA dataset
dm = MedQaDataModule(tokenizer=tokenizer,
                     num_proc=1,
                     use_subset=True,
                     verbose=True,
                     corpus=corpus,
                     # retrieve 100 documents for each question
                     n_documents=100,
                     # retrieve the whole training set
                     train_batch_size=100,
                     relevance_classifier=MetaMapMatch(
                         answer_prefix='answer.',
                         document_prefix='document.',
                         output_key='document.is_positive'
                         ))

# prepare both the QA dataset and the corpus
dm.prepare_data()
dm.setup()
corpus.prepare_data()
corpus.setup()

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