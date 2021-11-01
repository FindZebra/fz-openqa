from fz_openqa.datamodules.__old.corpus_dm import FzCorpusDataModule
from fz_openqa.datamodules.index import ElasticSearchIndex
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.scispacy import display_entities_pipe
import en_core_sci_md
import rich

tokenizer = init_pretrained_tokenizer(pretrained_model_name_or_path='bert-base-cased')

# load the corpus object
corpus = FzCorpusDataModule(tokenizer=tokenizer,
                            index=ElasticSearchIndex(index_key="idx",
                                                     text_key="document.text",
                                                     filter_mode=None),
                            verbose=False,
                            num_proc=1,
                            use_subset=True,
                            train_batch_size=3)


# prepare both the QA dataset and the corpus
corpus.prepare_data()
corpus.setup()


for i in corpus.dataset:
    print(f"\n ScispaCy entities for {i['document.idx']}:")
    rich.print(display_entities_pipe(en_core_sci_md, i['document.text'])[1])
