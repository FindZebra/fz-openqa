from fz_openqa.datamodules.corpus_dm import MedQaEnDataModule, FzCorpusDataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.scispacy import display_entities_pipe
import en_ner_bc5cdr_md
import rich

tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path='bert-base-cased')

corpus = MedQaEnDataModule(tokenizer=tokenizer,
                            passage_length=200,
                            passage_stride=100,
                            append_document_title=False,
                            num_proc=4,
                            use_subset=True,
                            verbose=False)
corpus.prepare_data()
corpus.setup()


for i in corpus.dataset:
    print(f"\n ScispaCy entities for {i['document.idx']}:")
    rich.print(display_entities_pipe(en_ner_bc5cdr_md, i['document.text']))
    # todo: remove (None, None) from print
