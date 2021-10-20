import datasets
import rich

from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer

datasets.set_caching_enabled(False)

tokenizer = init_pretrained_tokenizer(
    pretrained_model_name_or_path="bert-base-cased"
)

dm = MedQaDataModule(
    tokenizer=tokenizer,
    num_proc=1,
    use_subset=True,
    verbose=True,
    n_retrieved_documents=0,
)

dm.prepare_data()
dm.setup()
