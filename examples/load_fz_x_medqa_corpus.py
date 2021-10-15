import datasets
import rich
from fz_openqa.datamodules.corpus_dm import FZxMedQaCorpusDataModule, MedQaCorpusDataModule, FzCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator, pprint_batch

datasets.set_caching_enabled(False)

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path='bert-base-cased')

# load the corpus object
corpus = FzCorpusDataModule(tokenizer=tokenizer,
                            index=None,
                            verbose=False,
                            num_proc=4,
                            use_subset=False,
                            train_batch_size=10)

# full: 1 020 817
# MedQA: 230 180
# FZ: 790 637
# setup
corpus.prepare_data()
corpus.setup()

print(get_separator())
corpus.pprint()
print(get_separator())
corpus.display_sample()
