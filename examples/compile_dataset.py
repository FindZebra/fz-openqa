from time import time

import datasets
import rich
from rich.progress import track

from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.datamodules.pipes import ExactMatch
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator, pprint_batch

datasets.set_caching_enabled(False)

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path='bert-base-cased')

# load the corpus object
corpus = FzCorpusDataModule(tokenizer=tokenizer,
                            index=ElasticSearchIndex(index_key="idx",
                                                     text_key="document.text",
                                                     query_key="question.text",
                                                     num_proc=4,
                                                     filter_mode=None),
                            verbose=False,
                            num_proc=4,
                            use_subset=True)

# load the QA dataset
dm = MedQaDataModule(tokenizer=tokenizer,
                     num_proc=1, # todo: increase to 4
                     use_subset=True,
                     verbose=True,
                     corpus=corpus,
                     # retrieve 100 documents for each question
                     n_documents=10,
                     # retrieve the whole training set
                     train_batch_size=10,
                     use_corpus_sampler=False,
                     relevance_classifier=ExactMatch(
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

# Compile the dataset
dm.compile_dataset()
rich.print(f"=== Compiled Dataset ===")
rich.print(dm.compiled_dataset)
print(get_separator())
pprint_batch(dm.compiled_dataset["train"][0], "Compiled dataset example")

batch = next(iter(dm.train_dataloader()))
pprint_batch(batch, "compiled batch")
