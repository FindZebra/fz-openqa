import json
import logging
import os
import shutil
import sys
from collections import Counter
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

# PATH = "2022-04-13/16-46-28" # no norm
PATH = "2022-04-11/12-16-00"  # norm


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

    # output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True, parents=True)

    # set the default checkpoint based on the environment
    if os.environ.get("USER") == "valv" and sys.platform != "darwin":
        DEFAULT_CKPT = f"/scratch/valv/fz-openqa/runs/{PATH}"
    else:
        DEFAULT_CKPT = f"/Users/valv/Documents/Research/remote_data/{PATH.split()[-1]}"

    # load a checkpoint
    loader = CheckpointLoader(config.get("checkpoint", DEFAULT_CKPT), override=config)
    loader.print_config()
    cache_dir = Path(loader.config.sys.cache_dir)
    logger.info(f"cache_dir = {cache_dir}, exists={cache_dir.exists()}")

    # load model
    logging.info("Loading model...")
    model: Model = loader.load_model(zero_shot=config.get("zero_shot", False))
    logger.info(f"Loaded model {type(model.module)}, fingerprint={get_fingerprint(model)}")
    logger.info(f"Reader temperature: {model.module.reader_head.temperature()}")
    logger.info(f"Retriever temperature: {model.module.retriever_head.temperature()}")

    # define the trainer
    trainer = loader.instantiate("trainer")
    rich.print(f">> trainer = {trainer}")

    # build all the datasets
    datamodule = loader.instantiate("datamodule")
    if config.get("subset_size", None) is not None:
        datamodule.builder.dataset_builder.subset_size = config.subset_size
    if config.get("setup_with_model", False):
        logger.info("Setting up datasets with model...")
        datamodule.setup(trainer=trainer, model=model)
    else:
        logger.info("Setting up datasets without model...")
        datamodule.setup()

    # get batch of data
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    pprint_batch(batch, "Training batch")
    pids = batch["document.row_idx"]
    pid_freq = Counter(pids.view(-1).cpu().detach().numpy().tolist())
    rich.print({k: f for k, f in pid_freq.most_common(20)})
    with open(output_dir / "pid_freq.json", "w") as f:
        f.write(json.dumps({k: f for k, f in pid_freq.most_common()}, indent=2))

    # process the data
    model.eval()
    output = model(batch, head="retriever")
    pprint_batch(output, "Output batch")

    # inspect the data
    with open(output_dir / "outputs.txt", "w") as f:
        logger.info(f"Writing to: {output_dir.absolute()}")
        TERM_SIZE = shutil.get_terminal_size()[0]
        N_SAMPLES = 10
        N_DOCS = 10
        H_MAX = 100
        W_MAX = 100
        d_input_ids = batch.get("document.input_ids")
        q_input_ids = batch.get("question.input_ids")
        retrieval_score = batch.get("document.retrieval_score")
        hq = output.get("_hq_")
        hd = output.get("_hd_")
        scores = torch.einsum("bmqh,bmkdh->bmkqd", hq, hd)
        targets = batch.get("answer.target")
        for i in range(min(N_SAMPLES, len(hq))):
            j = targets[i]
            d_input_ids_i = d_input_ids[i][j]
            q_input_ids_i = q_input_ids[i][j]
            retrieval_score_i = retrieval_score[i][j]
            q_input_ids_i_ = [int(q.item()) for q in q_input_ids_i]

            if int(loader.tokenizer.pad_token_id) in q_input_ids_i_:
                q_padding_idx = q_input_ids_i_.index(int(loader.tokenizer.pad_token_id))
            else:
                q_padding_idx = None
            q_input_ids_i = q_input_ids_i[:q_padding_idx]
            scores_i = scores[i][j]
            # hq_i = hq[i][j]
            # hd_i = hd[i][j]
            msg = f" Question {i+1} "
            rich.print(f"{msg:=^{TERM_SIZE}}")
            u = pretty_decode(q_input_ids_i, tokenizer=loader.tokenizer)
            rich.print(u)
            f.write(f"{msg}\n{u}\n")

            for k in range(N_DOCS):
                # hd_ik = hd_i[k]
                retrieval_score_ik = retrieval_score_i[k]
                scores_ik = scores_i[k, :q_padding_idx, :]
                scores_ik = scores_ik.softmax(dim=-1)
                d_input_ids_ik = d_input_ids_i[k]
                msg = f" Document {i+1}-{k+1} : score={retrieval_score_ik:.2f} "
                rich.print(f"{msg:=^{TERM_SIZE}}")
                u = pretty_decode(d_input_ids_ik, tokenizer=loader.tokenizer, style="white")
                rich.print(u)
                f.write(f"{msg}\n{u}\n")

                # visualize the scores
                q_tokens = [
                    loader.tokenizer.decode(t, skip_special_tokens=True) for t in q_input_ids_i
                ]
                d_tokens = [
                    loader.tokenizer.decode(t, skip_special_tokens=True) for t in d_input_ids_ik
                ]

                # print corresponding `max` scores
                with open(output_dir / f"max_mapping-{i}-{k}.txt", 'w') as fmp:
                    for qi, q_token in enumerate(q_tokens):
                        qj = scores_ik[i].argmax(dim=-1)
                        fmp.write(f"{q_token:>10} -> {d_tokens[qj]:<10}\n")

                # heatmap
                plt.figure(figsize=(20, 20))
                y = scores_ik[:H_MAX, :W_MAX]

                # highlight the max score
                # ym = 1.2 * y.max().item()
                # _, ymi = y.max(dim=1)
                # y = y.scatter(dim=1, index=ymi[:, None], value=float(ym))

                sns.heatmap(
                    y,
                    xticklabels=d_tokens[:W_MAX],
                    yticklabels=q_tokens[:H_MAX],
                )
                plt.savefig(output_dir / f"heatmap-{i}-{k}.png")
                plt.close()

    logger.info(f"Written to: {output_dir.absolute()}")


if __name__ == "__main__":
    run()
