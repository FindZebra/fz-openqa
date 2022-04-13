import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())

import rich
from hydra.utils import instantiate

from fz_openqa.utils.pretty import pprint_batch, pretty_decode

import datasets
import hydra
import torch
from omegaconf import DictConfig

from fz_openqa import configs
from fz_openqa.inference.checkpoint import CheckpointLoader
from fz_openqa.modeling import Model
from fz_openqa.utils.fingerprint import get_fingerprint
from loguru import logger
import fz_openqa.training.experiment  # noqa

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


@torch.no_grad()
@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="inspect_config.yaml",
)
def run(config: DictConfig) -> None:
    """ "
    Rank the FindZebra queries using Indexes.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.set_caching_enabled(True)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    # avoid "too many open files" error
    torch.multiprocessing.set_sharing_strategy("file_system")

    # set the default checkpoint based on the environment
    if os.environ.get("USER") == "valv" and sys.platform != "darwin":
        DEFAULT_CKPT = "/scratch/valv/fz-openqa/runs/2022-04-12/12-28-19"
    else:
        DEFAULT_CKPT = "/Users/valv/Documents/Research/remote_data/12-28-19"

    # load a checkpoint
    loader = CheckpointLoader(config.get("checkpoint", DEFAULT_CKPT), override=config)
    loader.print_config()
    cache_dir = Path(loader.config.sys.cache_dir)
    logger.info(f"cache_dir = {cache_dir}, exists={cache_dir.exists()}")

    # load model
    logging.info("Loading model...")
    model: Model = loader.load_model(zero_shot=config.get("zero_shot", False))
    logger.info(f"Loaded model {type(model.module)}, fingerprint={get_fingerprint(model)}")

    # build all the datasets
    datamodule = loader.instantiate("datamodule")
    datamodule.setup()
    rich.print(f">> Datamodule: {datamodule}")

    # define the trainer
    trainer = instantiate(config.get("trainer", {}))
    rich.print(f">> trainer = {trainer}")

    # get batch of data
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    pprint_batch(batch, "Training batch")

    # process the data
    model.eval()
    output = model(batch, head="retriever")
    pprint_batch(output, "Output batch")

    # inspect the data
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Writing to: {output_dir.absolute()}")
    TERM_SIZE = shutil.get_terminal_size()[0]
    N_SAMPLES = 10
    N_DOCS = 3
    H_MAX = None
    W_MAX = None
    d_input_ids = batch.get("document.input_ids")
    q_input_ids = batch.get("question.input_ids")
    hq = output.get("_hq_")
    hd = output.get("_hd_")
    scores = torch.einsum("bmqh,bmkdh->bmkqd", hq, hd)
    targets = batch.get("answer.target")
    for i in range(min(N_SAMPLES, len(hq))):
        j = targets[i]
        d_input_ids_i = d_input_ids[i][j]
        q_input_ids_i = q_input_ids[i][j]
        q_input_ids_i_ = [q.item() for q in q_input_ids_i]
        if loader.tokenizer.pad_token_id in q_input_ids_i_:
            q_padding_idx = q_input_ids_i_.index(loader.tokenizer.pad_token_id)
        else:
            q_padding_idx = None
        q_input_ids_i = q_input_ids_i[:q_padding_idx]
        scores_i = scores[i][j][:q_padding_idx]
        # hq_i = hq[i][j]
        # hd_i = hd[i][j]
        msg = f" Question {i+1} "
        rich.print(f"{msg:=^{TERM_SIZE}}")
        rich.print(pretty_decode(q_input_ids_i, tokenizer=loader.tokenizer))

        for k in range(N_DOCS):
            # hd_ik = hd_i[k]
            scores_ik = scores_i[k]
            scores_ik = scores_ik.softmax(dim=-1)
            d_input_ids_ik = d_input_ids_i[k]
            msg = f" Document {i+1}:{k+1} "
            rich.print(f"{msg:=^{TERM_SIZE}}")
            rich.print(pretty_decode(d_input_ids_ik, tokenizer=loader.tokenizer, style="white"))
            rich.print(f"Scores: {scores_ik.shape}")

            # visualize the scores
            q_tokens = [loader.tokenizer.decode(t, skip_special_tokens=True) for t in q_input_ids_i]
            d_tokens = [
                loader.tokenizer.decode(t, skip_special_tokens=True) for t in d_input_ids_ik
            ]

            # print corresponding `max` scores
            for qi, q_token in enumerate(q_tokens):
                qj = scores_ik[i].argmax(dim=-1)
                rich.print(f"{q_token:>10} -> {d_tokens[qj]:<10}")

            # heatmap
            plt.figure(figsize=(20, 20))
            sns.heatmap(
                scores_ik[:H_MAX, :W_MAX],
                xticklabels=d_tokens[:W_MAX],
                yticklabels=q_tokens[:H_MAX],
            )
            plt.savefig(output_dir / f"heatmap-{i}-{k}.png")
            plt.close()

    logger.info(f"Written to: {output_dir.absolute()}")

    # # build the index
    # index_builder = instantiate(
    #     config.datamodule.index_builder,
    #     model=model,
    #     trainer=trainer,
    #     dataset=corpus,
    #     collate_pipe=corpus_builder.get_collate_pipe(),
    # )
    #
    # logger.info(f"Indexing index with {index_builder}..")
    # index: Index = index_builder()
    #
    # # query the index with the dataset
    # logger.info(f"Indexing dataset (pipe={index.fingerprint(reduce=True)})..")
    # indexed_dataset = index(
    #     dataset,
    #     collate_fn=dataset_builder.get_collate_pipe(
    #         columns=["question.input_ids", "question.attention_mask"]
    #     ),
    #     k=config.get("topk", 1000),
    #     batch_size=10,
    #     trainer=trainer,
    #     num_proc=4,
    #     cache_fingerprint=Path(cache_dir) / "rank_fz_queries_fingerprints",
    #     fingerprint_kwargs_exclude=["trainer"],
    #     set_new_fingerprint=True,
    # )


if __name__ == "__main__":
    run()
