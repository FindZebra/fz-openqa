import datasets

from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator, pprint_batch

datasets.set_caching_enabled(False)

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path='bert-base-cased')

# load the corpus object
corpus = FzCorpusDataModule(tokenizer=tokenizer,
                            index=ElasticSearchIndex(index_key="idx",
                                                     text_key="document.text",
                                                     filter_mode=None),
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
                     n_documents=3)

# prepare both the QA dataset and the corpus
dm.prepare_data()
dm.setup()

print(get_separator())
dm.build_index()
print(f">> index is built.")
print(get_separator())

batch = next(iter(dm.train_dataloader()))
pprint_batch(batch)
print(get_separator())
