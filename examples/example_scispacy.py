import rich

from fz_openqa.datamodules.corpus_dm import MedQaCorpusDataModule
from fz_openqa.datamodules.pipes.text_filtering import SciSpacyFilter
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
import en_core_sci_scibert

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path='bert-base-cased')

corpus = MedQaCorpusDataModule(tokenizer=tokenizer,
                               passage_length=200,
                               passage_stride=100,
                               append_document_title=False,
                               num_proc=4,
                               use_subset=True,
                               verbose=False)
corpus.prepare_data()
corpus.setup()

sci = SciSpacyFilter(
    spacy_model=en_core_sci_scibert,
    text_key='document.text')

for i in range(10):
    row = corpus.dataset[i]
    rich.print(f"\n[cyan]ScispaCy entities for document #{row['idx']}:")
    print(sci.filter(text=row['document.text']))
