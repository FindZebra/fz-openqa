import numpy as np
import rich

from fz_openqa.datamodules.corpus_dm import MedQaEnDataModule, FzCorpusDataModule
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

print(f">> index the dataset using bm25")
corpus.index(index="bm25")

#qst="A 59-year-old man presents to his primary care physician, accompanied by his wife, who requests treatment for his chronic pessimism. The patient admits to feeling tired and down most of the time for the past several years but insists that it is just part of getting old. His wife believes that he has become more self-critical and less confident than he used to be. Neither the patient nor his wife can identify any stressors or triggering events. He has continued to work as a librarian at a nearby college during this time and spends time with friends on the weekends. He sleeps 7 hours per night and eats 3 meals per day. He denies suicidal ideation or periods of elevated mood, excessive irritability, or increased energy. Physical exam reveals a well-dressed, well-groomed man without apparent abnormality. Basic neurocognitive testing and labs (CBC, BMP, TSH, cortisol, testosterone, and urine toxicology) are within normal limits. What is the most likely diagnosis?"
# query bm25
#scores, examples = corpus.query(index="bm25",query=qst, k=1)
#print(scores, examples)
# query faiss
#scores, examples = corpus.query("faiss", k=1)
