from fz_openqa.datamodules.corpus_dm import MedQaCorpusDataModule, FzCorpusDataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.datamodules.pipes.text_filtering import SciSpacyFilter
#from fz_openqa.utils.scispacy import display_entities_pipe
#import en_ner_bc5cdr_md
#import en_core_sci_md
import rich

tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path='bert-base-cased')

corpus = MedQaCorpusDataModule(tokenizer=tokenizer,
                            passage_length=200,
                            passage_stride=100,
                            append_document_title=False,
                            num_proc=4,
                            use_subset=True,
                            verbose=False)
corpus.prepare_data()
corpus.setup()

sci = SciSpacyFilter(text_key='document.text')

for i in corpus.dataset:
    print(f"\n ScispaCy entities for {i['idx']}:")
    rich.print(sci.filter(text=i['document.text']))
    