from pathlib import Path

import dotenv
import hydra
import rich
import torch.nn
import transformers
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf
from transformers import BertTokenizerFast
from warp_pipes import get_console_separator

import fz_openqa.training.experiment  # type: ignore
from fz_openqa import configs
from fz_openqa.training.training import load_checkpoint
from fz_openqa.utils.config import print_config
from fz_openqa.utils.fingerprint import get_fingerprint
from scripts.utils import configure_env

dotenv.load_dotenv(Path(fz_openqa.__file__).parent.parent / ".env")


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="export_config.yaml",
)
def run(config):
    """This is an example script showing how to export a model to HuggingFace."""
    print_config(config)
    configure_env(config, silent_hf=False)

    # load the model
    checkpoint_manager = load_checkpoint(
        config.checkpoint,
        override_config=DictConfig({"sys": config.sys}),
        ref_config=config,
        silent=True,
    )
    model = checkpoint_manager.load_model(config.checkpoint_type)
    checkpoint_config = checkpoint_manager.config
    logger.info("Checkpoint config:")
    print(get_console_separator("-"))
    print_config(checkpoint_config)
    print(get_console_separator("-"))
    logger.info(f"Model fingerprint: {get_fingerprint(model)}")

    # get the tokenizer
    tokenizer = checkpoint_manager.load_tokenizer()

    # load the target HF model
    hf_model = transformers.AutoModel.from_pretrained(tokenizer.name_or_path)
    if hf_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        hf_model.resize_token_embeddings(len(tokenizer))

    # retrieve the weights from the trained model
    model_bert = model.module.backbone
    model_proj = model.module.retriever_head.q_head

    # override HF model with he trained weights
    hf_model.load_state_dict(model_bert.state_dict())
    hf_model.pooler.activation = torch.nn.Identity()
    hf_model.pooler.dense.weight.data = model_proj.weight.data
    if model_proj.bias is not None:
        hf_model.pooler.dense.bias.data = 0.
    else:
        hf_model.pooler.dense.bias = None

    # process batch and compare the outputs
    with torch.inference_mode():
        for field, key in [("document", "_hd_"), ("question", "_hq_")]:
            batch = tokenizer(["hello world", "Paris is the capital of France"], padding=True, truncation=True, return_tensors="pt")
            hf_output = hf_model(**batch)
            hf_output = hf_output.pooler_output
            output = model({f"{field}.{k}": v for k, v in batch.items()})
            output = output[key]

            diff = (hf_output - output)**2
            if not diff.sum().item() == 0:
                raise ValueError(f"Error in {field} field. Diff: {diff}")
            else:
                logger.info(f"output validated for {field} field.")


    # export the model to the hub
    hf_model.push_to_hub(config.export_model_id)
    tokenizer.push_to_hub(config.export_model_id)


if __name__ == "__main__":
    run()
