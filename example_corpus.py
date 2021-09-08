import numpy as np
import rich

from fz_openqa.datamodules.corpus_dm import FzCorpusDataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path='bert-base-cased')

corpus = FzCorpusDataModule(tokenizer=tokenizer,
                            passage_length=200,
                            passage_stride=100,
                            append_document_title=False,
                            num_proc=4,
                            use_subset=True,
                            verbose=False)
corpus.prepare_data()
corpus.setup()

print(f">> get dataset")
rich.print(corpus.dataset["train"])

print(f">> index the dataset using vectors")
corpus.index(model=lambda batch:  batch['document.input_ids'], index="faiss")

print(f">> index the dataset bm25")
corpus.index(model=lambda batch:  batch['document.input_ids'], index="bm25")

# query
output = corpus.query({'text': "Example query."}, k=1)
