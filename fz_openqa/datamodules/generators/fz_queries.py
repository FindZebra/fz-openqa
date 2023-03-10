import csv
from pathlib import Path

import datasets
import rich

TXT_PATTERN = r"^.*\.txt$"


class FzQueriesConfig(datasets.BuilderConfig):
    """BuilderConfig for the MedQa English Corpus objecxt."""

    def __init__(self, **kwargs):
        """BuilderConfig for the Corpus object.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FzQueriesConfig, self).__init__(**kwargs)


_DESCRIPTION = "Load the FindZebra queries"
_VERSION = "0.0.1"
_HOMEPAGE = "https://github.com/vlievin/fz-openqa"
_CITATION = ""
_URL = (
    "https://f001.backblazeb2.com/file/FindZebraData/fz-openqa/datasets/fz-retrieval-validation.csv"
)


class FzQueriesGenerator(datasets.GeneratorBasedBuilder):
    """FzCorpus Dataset. Version 0.0.1"""

    VERSION = datasets.Version(_VERSION)
    force = False

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "question.idx": datasets.Value("int32"),
                    "question.text": datasets.Value("string"),
                    "question.cui": datasets.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": Path(downloaded_file)},
            )
        ]

    def _generate_examples(self, filepath: str):
        """Yields examples."""

        header = None
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for idx, line in enumerate(reader):
                if idx == 0:
                    header = line
                else:
                    idx = idx - 1

                    # retrieve CUIs
                    cuis = [line[header.index("answer_cui")]]
                    alternative_cui = line[header.index("alternative_cui")]
                    if alternative_cui != "":
                        cuis.append(alternative_cui)

                    # format output
                    row = {
                        "question.idx": idx,
                        "question.text": line[header.index("question")],
                        "question.cui": cuis,
                    }

                    yield idx, row
