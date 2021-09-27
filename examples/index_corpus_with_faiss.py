import sys
import datasets
import torch

from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule
from fz_openqa.datamodules.index import FaissIndex
from examples.utils import gen_example_query, display_search_results
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

datasets.set_caching_enabled(False)
print(f">> {sys.version}")

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path='bert-base-cased')

query = gen_example_query(tokenizer)

dm = FzCorpusDataModule(tokenizer=tokenizer,
                        index=FaissIndex(),
                        verbose=True,
                        num_proc=1,
                        use_subset=True,
                        train_batch_size=3)
dm.prepare_data()
dm.setup()

print(f">> indexing the dataset using Faiss")
model = lambda batch: torch.randn((batch['document.input_ids'].size(0), 10,))
dm.build_index(model=model)

print(f">> query the dataset using Faiss")
model = lambda batch: torch.randn((batch['question.input_ids'].size(0), 10,))
result = dm.search_index(query, k=3, model=model)

display_search_results(query, result)
