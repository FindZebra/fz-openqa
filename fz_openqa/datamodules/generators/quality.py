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
# todo: _URL = "https://github.com/nyu-mll/quality/raw/main/data/QuALITY.v0.9.zip"
_URL = "/Users/valv/Downloads/QuALITY.v0.9.zip"


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
    BUILDER_CONFIGS = [
        QualityConfig(
            name="full",
            description="QuALITY dataset with questions and articles",
        ),
        QualityConfig(
            name="questions",
            description="QuALITY dataset - only the questions",
        ),
        QualityConfig(
            name="documents",
            description="QuALITY dataset - only the articles",
        ),
    ]

    def _info(self):
        if self.config.name == "full":
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                        "question.idx": datasets.Value("int32"),
                        "question.text": datasets.Value("string"),
                        "document.idx": datasets.Value("int32"),
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
        elif self.config.name == "questions":
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                        "question.idx": datasets.Value("int32"),
                        "question.text": datasets.Value("string"),
                        "document.idx": datasets.Value("int32"),
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
        elif self.config.name == "documents":
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                        "document.idx": datasets.Value("int32"),
                        "document.text": datasets.Value("string"),
                        "document.title": datasets.Value("string"),
                    }
                ),
                supervised_keys=None,
                homepage=_HOMEPAGE,
                license=_LICENSE,
                citation=_CITATION,
            )
        else:
            raise ValueError(f"Unknown config name={self.config.name}")

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
        if self.config.name == "documents":
            with open(filepath, "r") as f:
                for doc_idx, line in enumerate(f.readlines()):
                    data = json.loads(line)
                    row = {
                        "document.idx": doc_idx,
                        "document.text": data["article"],
                        "document.title": data["title"],
                    }
                    yield doc_idx, row
        else:
            with open(filepath, "r") as f:
                q_idx = 0
                for doc_idx, line in enumerate(f.readlines()):
                    data = json.loads(line)
                    for question in data["questions"]:
                        target = question.get("gold_label", -1)
                        row = {
                            "question.text": question["question"],
                            "question.idx": q_idx,
                            "document.idx": doc_idx,
                            "question.difficulty": "hard" if question["difficult"] else "easy",
                            "answer.text": question["options"],
                            "answer.target": target - 1,
                        }
                        if self.config.name == "full":
                            row["document.text"] = data["article"]
                            row["document.title"] = data["title"]

                        yield q_idx, row
                        q_idx += 1
