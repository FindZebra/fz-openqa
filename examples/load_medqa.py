import datasets

from fz_openqa.datamodules.qa_dm import QaDatamodule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

datasets.set_caching_enabled(False)

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path='bert-base-cased')

dm = QaDatamodule(tokenizer=tokenizer,
                  num_proc=1,
                  use_subset=True,
                  verbose=True)

dm.prepare_data()
dm.setup()
