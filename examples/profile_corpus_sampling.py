import datasets
from timeit import Timer

import rich
from rich import status
from rich.progress import track
import numpy as np
from tqdm import tqdm
from rich.progress import track
from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator, pprint_batch
from fz_openqa.utils.train_utils import setup_safe_env

datasets.set_caching_enabled(True)
setup_safe_env()

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path='bert-base-cased')

# load the corpus object
corpus = FzCorpusDataModule(tokenizer=tokenizer,
                            index=ElasticSearchIndex(index_key="idx",
                                                     text_key="document.text",
                                                     filter_mode=None),
                            verbose=False,
                            num_proc=4,
                            use_subset=False,
                            train_batch_size=3)

# load the QA dataset
dm = MedQaDataModule(tokenizer=tokenizer,
                     num_proc=4,
                     use_subset=False,
                     verbose=True,
                     corpus=corpus,
                     num_workers=4,
                     train_batch_size=16,
                     n_documents=100)

# prepare both the QA dataset and the corpus
dm.prepare_data()
dm.setup()
corpus.prepare_data()
corpus.setup()

print(get_separator())
dm.build_index()
rich.print(f"[green]>> index is built.")
print(get_separator())

def fun():
    print(".", end="")
    batch = next(iter(dm.train_dataloader()))
    # access one value to make sure the batch is actually loaded
    return len(batch['document.input_ids'])

rich.print("[cyan]timing batch loading..")
time = Timer(fun).repeat(11, 1)
time = time[1:] # skip first iter

rich.print(f"> runtime={np.mean(time):.3f} (+- {np.std(time):.3f})")
rich.print(time)


# Results:
# with corpus sampling in collate:
