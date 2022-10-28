import json
import re
from pathlib import Path

import datasets

TXT_PATTERN = r"^.*\.txt$"


class FzCorpusConfig(datasets.BuilderConfig):
    """BuilderConfig for the MedQa English Corpus objecxt."""

    def __init__(self, **kwargs):
        """BuilderConfig for the Corpus object.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FzCorpusConfig, self).__init__(**kwargs)


_DESCRIPTION = "A class to load the FindZebra corpus"
_VERSION = "0.0.1"
_HOMEPAGE = "https://github.com/vlievin/fz-openqa"
_CITATION = ""
_URL = "https://f001.backblazeb2.com/file/FindZebraData/fz-openqa/datasets/fz-corpus.zip"


class FzCorpusGenerator(datasets.GeneratorBasedBuilder):
    """FzCorpus Dataset. Version 0.0.1"""

    VERSION = datasets.Version(_VERSION)
    force = False

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "document.idx": datasets.Value("int32"),
                    "document.text": datasets.Value("string"),
                    "document.title": datasets.Value("string"),
                    "document.cui": datasets.Value("string"),
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
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": Path(downloaded_file) / "fz-corpus"},
            )
        ]

    def _generate_examples(self, filepath: str):
        """Yields examples."""
        dirpath = Path(filepath)

        cleanr = re.compile(r"(<.*?>)|(\[.*?\])")

        for idx, path in enumerate(dirpath.glob("**/*.json")):
            article = json.loads(open(path, "r").read())

            # cleanup the text
            text = re.sub(cleanr, "", article["raw_content"])

            # get cui
            cui = article["cui"]
            if cui is None:
                cui = "<unknown-cui>"

            # yield the data
            yield idx, {
                "document.text": str(text),
                "document.title": str(article["title"]),
                "document.cui": str(cui),
                "document.idx": idx,
            }
