import datasets
import rich

from fz_openqa.datamodules.meqa_dm import MedQaDataModule
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.pretty import get_separator

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


print(get_separator())
rich.print(dm.dataset)
print(get_separator())


batch = next(iter(dm.train_dataloader()))


for idx in range(3):
    print(get_separator("-"))
    eg = Pipe.get_eg(batch, idx=idx)
    rich.print(
        f"Example #{1+idx}: \n"
        f" * answer=[magenta]{eg['answer.text'][eg['answer.target']]}[/magenta]\n"
        f" * question=[cyan]{eg['question.text']}[/cyan]\n"
    )
