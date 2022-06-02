import logging
import os
import string
import sys
import time
from pathlib import Path

from tqdm import tqdm

from fz_openqa.utils.openai import AnsweringModel
from fz_openqa.utils.openai import MultipleChoiceTemplate

sys.path.append(Path(__file__).parent.parent.as_posix())

import datasets
import hydra
import rich
from omegaconf import DictConfig

from fz_openqa import configs
from fz_openqa.datamodules.builders.qa import QaBuilder
from fz_openqa.datamodules.builders.preprocessing.entity import EntityPreprocessing

from loguru import logger


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="script_config.yaml",
)
def run(config: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.set_caching_enabled(True)
    logging.getLogger("openai").setLevel(logging.WARNING)

    # preprocessing
    preprocessing_op = config.get("preprocessing", None)
    if isinstance(preprocessing_op, str):
        preprocessing_op = {"entity": EntityPreprocessing}[preprocessing_op]()

    # initialize the data module
    builder = QaBuilder(
        tokenizer=None,
        use_subset=config.get("use_subset", True),
        cache_dir=config.sys.cache_dir,
        num_proc=config.get("num_proc", 2),
        dset_name=config.get("dset_name", "medqa-us"),
        preprocessing_op=preprocessing_op,
    )
    builder.subset_size = [3, 3, 3]
    dataset = builder()
    rich.print(dataset)

    # setting up OpenAI API
    engine = config.get("engine", "text-davinci-002")
    model = AnsweringModel(
        engine=engine,
        prompt_mode="chain_of_thought",
        template=MultipleChoiceTemplate(),
    )

    output_dir = Path(os.getcwd()) / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    rich.print(f">> Logging to {output_dir}")
    rate_limit = 2
    t0 = time.time()
    splits = ["test"]
    results = {}
    for split in splits:
        dset = dataset[split]
        total = 0
        np = 0
        indices = list(range(len(dset)))
        rgn = np.random.RandomState(0)
        rgn.shuffle(indices)
        for i, row_idx in (
            pbar := tqdm(enumerate(indices), unit=" questions", total=len(indices))
        ) :
            row = dset[row_idx]
            # header = f" Question {row_idx + 1} "
            # rich.print(f"[blue]{header:=^60}")
            question = row["question.text"]
            options = row["answer.text"]
            answer_idx = row["answer.target"].item()
            answer = string.ascii_uppercase[answer_idx]
            # rich.print(f"Question: {question}")
            # rich.print(f"Options: {options}")
            # rich.print(f"Answer: {answer}")

            model_answer, meta = model(question, options)
            try:
                model_answer_idx = string.ascii_uppercase.index(model_answer)
            except Exception as exc:
                logger.warning(f"{exc}")
                model_answer_idx = -1
            # prefix = "[green] (V)" if model_answer == answer else "[red] (X)"
            # rich.print(f"{prefix} Model answer: {model_answer} ({meta['answer']}).
            # Reasoning: \n{meta['reasoning']}")

            # update the tracking of the accuracy
            total += 1
            if model_answer_idx == answer_idx:
                np += 1
            pbar.set_description(f"({split}) Accuracy: {np / total:.2f}")

            with open(output_dir / f"{split}_{row_idx}.txt", "w") as f:
                outcome = "correct" if model_answer == answer else "incorrect"
                formatted_options = ",".join(
                    [f"{string.ascii_uppercase[i]}) {option}" for i, option in enumerate(options)]
                )
                output_str = f"""\
                Outcome: {outcome}\n
                Question ({split}#{row_idx}): {question}\n
                Options: {formatted_options}\n
                Answer: {answer}: {options[answer_idx]}\n
                Model answer: {model_answer}) {options[model_answer_idx]}\n\n
                Reasoning: \n{meta['completed_prompt']}
                """
                f.write(output_str)

            if time.time() - t0 < rate_limit:
                t = rate_limit - (time.time() - t0)
                time.sleep(max(t, 0))
            t0 = time.time()

        results[split] = np / total

    for split, accuracy in results.items():
        rich.print(f">> {split}: {accuracy:.3%}")
    rich.print(f">> Logged to {output_dir}")


if __name__ == "__main__":
    run()
