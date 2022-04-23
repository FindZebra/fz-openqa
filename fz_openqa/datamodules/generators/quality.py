import json
from pathlib import Path

import datasets
from datasets import Split

_CITATION = """\
@article{pang2021quality,
  title={{QuALITY}: Question Answering with Long Input Texts, Yes!},
  author={Pang, Richard Yuanzhe and Parrish, Alicia and Joshi, Nitish and Nangia,
  Nikita and Phang, Jason and Chen, Angelica and Padmakumar, Vishakh and Ma, Johnny and Thompson,
  Jana and He, He and Bowman, Samuel R.},
  journal={arXiv preprint arXiv:2112.08608},
  year={2021}
}
}
"""

_DESCRIPTION = ""

_HOMEPAGE = "https://github.com/nyu-mll/quality"

_LICENSE = ""
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URL = "https://github.com/nyu-mll/quality/raw/main/data/QuALITY.v0.9.zip"


# _URL = "/Users/valv/Downloads/QuALITY.v0.9.zip"


class QualityConfig(datasets.BuilderConfig):
    """BuilderConfig for Quality"""

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(QualityConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class QuALITY(datasets.GeneratorBasedBuilder):
    """Quality: A Dataset for Biomedical Research Question Answering"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "question.idx": datasets.Value("int32"),
                    "document.uid": datasets.Value("string"),
                    "question.text": datasets.Value("string"),
                    "document.title": datasets.Value("string"),
                    "document.text": datasets.Value("string"),
                    "answer.text": datasets.Sequence(datasets.Value("string")),
                    "answer.target": datasets.Value("int32"),
                    "question.difficulty": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = dl_manager.download_and_extract(_URL)
        root = Path(downloaded_files)

        def get_split(root, end_pattern):
            matches = [p for p in root.iterdir() if str(p).endswith(end_pattern)]
            assert len(matches) == 1
            return matches[0]

        splits = {
            k: get_split(root, ptrn)
            for k, ptrn in [
                (Split.TRAIN, "htmlstripped.train"),
                (Split.VALIDATION, "htmlstripped.dev"),
                (Split.TEST, "htmlstripped.test"),
            ]
        }

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"filepath": file, "split": split},
            )
            for split, file in splits.items()
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        with open(filepath, "r") as f:
            q_idx = 0
            for doc_idx, line in enumerate(f.readlines()):
                data = json.loads(line)
                for question in data["questions"]:
                    target = question.get("gold_label", -1)
                    row = {
                        "question.text": question["question"],
                        "question.idx": q_idx,
                        "question.difficulty": "hard" if question["difficult"] else "easy",
                        "answer.text": question["options"],
                        "answer.target": target - 1,
                        "document.uid": data["article_id"],
                        "document.text": data["article"],
                        "document.title": data["title"],
                    }

                    yield q_idx, row
                    q_idx += 1
