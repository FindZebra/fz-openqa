import json
import re

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


_DESCRIPTION = "A class to load the english MedQA corpus"
_VERSION = "0.0.1"
_HOMEPAGE = "https://github.com/MotzWanted/Open-Domain-MedQA"
_CITATION = ""
_URL = "https://drive.google.com/file/d/1665FL0D-QZwW-8os8xmxSXubw3jn1-ki/view?usp=sharing"


class FzCorpusDataset(datasets.GeneratorBasedBuilder):
    """FzCorpus Dataset. Version 0.0.1"""

    VERSION = datasets.Version(_VERSION)
    force = False

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "idx": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    "title": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    @staticmethod
    def _get_drive_url(url):
        base_url = "https://drive.google.com/uc?id="
        split_url = url.split("/")
        return base_url + split_url[5]

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(
            self._get_drive_url(_URL)
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_file},
            )
        ]

    def _generate_examples(self, filepath: str):
        """Yields examples."""

        cleanr = re.compile(r"(<.*?>)|(\[.*?\])")

        with open(filepath, "r") as f:
            data = json.load(f)
            for idx, article in enumerate(data):
                text = re.sub(cleanr, "", article["raw_content"])
                yield idx, {
                    "text": text,
                    "title": article["title"],
                    "idx": idx,
                }
