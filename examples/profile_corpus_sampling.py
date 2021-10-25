import cProfile
import pstats
from functools import partial
from time import time
from timeit import Timer

import datasets
import numpy as np
import rich

from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.train_utils import setup_safe_env

datasets.set_caching_enabled(True)
setup_safe_env()

MODE = "ctime"

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path="bert-base-cased"
)

# load the corpus object
corpus = FzCorpusDataModule(
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
    use_subset=False,
)

# load the QA dataset
dm = MedQaDataModule(
    tokenizer=tokenizer,
    num_proc=4,
    use_subset=False,
    verbose=True,
    corpus=corpus,
    num_workers=1,
    train_batch_size=16,
    n_documents=100,
    use_corpus_sampler=False,
)

# prepare both the QA dataset and the corpus
dm.prepare_data()
dm.setup()

print(get_separator())
dm.build_index()
rich.print("[green]>> index is built.")
print(get_separator())

dset_iter = iter(dm.train_dataloader())


def fun(dset_iter):
    batch = next(dset_iter)
    # access one value to make sure the batch is actually loaded
    print(f"> batch length={len(batch['document.input_ids'])}")


if MODE == "timeit":
    rich.print("[cyan]timing batch loading..")
    runtime = Timer(partial(fun, dset_iter)).repeat(11, 1)
    runtime = runtime[1:]  # skip first iter

    rich.print(f"> runtime={np.mean(runtime):.3f} (+- {np.std(runtime):.3f})")
    rich.print(time)
elif MODE == "ctime":

    def repeat_fun(times=1):
        return [fun(dset_iter) for _ in range(times)]

    repeats = 5
    profiler = cProfile.Profile()
    profiler.enable()
    t0 = time()
    repeat_fun(repeats)
    duration = time() - t0
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("time")
    stats.print_stats(20)
    print(get_separator())
    rich.print(f">> duration={duration / repeats:.3f}s/batch")
else:
    raise NotImplementedError

# Results: Corpus sampler is slower
# batch_size=16, num_workers=1, n_documents=100
# with corpus sampler (sample in __getitem__): >> duration=4.908s/batch
# without corpus sampler (sample in collate_fn): >> duration=2.947s/batch
# Comment: Corpus Sampler is expected to run faster if num_workers>1,
# this however crashes with JSONDecodeError
