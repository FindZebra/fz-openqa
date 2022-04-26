import os
import re
from pathlib import Path

import datasets
import rich

TXT_PATTERN = r"^.*\.txt$"


class MedWikipediaCorpusConfig(datasets.BuilderConfig):
    """BuilderConfig for the MedQa English Corpus objecxt."""

    def __init__(self, **kwargs):
        """BuilderConfig for the Corpus object.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MedWikipediaCorpusConfig, self).__init__(**kwargs)


_DESCRIPTION = "A class to load the Wikipedia-based MedQA corpus"
_VERSION = "0.0.1"
_HOMEPAGE = "https://github.com/MotzWanted/Open-Domain-MedQA"
_CITATION = ""
_URLS = {
    "v1": "https://f001.backblazeb2.com/file/FindZebraData/fz-openqa/datasets/"
    "medqa_x_wiki_corpus_v1.zip",
    "v2": "https://f001.backblazeb2.com/file/FindZebraData/fz-openqa/datasets/"
    "medqa_x_wiki_corpus_v2.zip",
    "v3-us": "https://f001.backblazeb2.com/file/FindZebraData/fz-openqa/datasets/"
    "medqa_x_wiki_corpus_v3_us.zip",
    "v3-tw": "https://f001.backblazeb2.com/file/FindZebraData/fz-openqa/datasets/"
    "medqa_x_wiki_corpus_v3_tw.zip",
    "v3": "https://f001.backblazeb2.com/file/FindZebraData/fz-openqa/datasets/"
    "medqa_x_wiki_corpus_v3_us_tw.zip",
}


class MedWikipediaCorpusGenerator(datasets.GeneratorBasedBuilder):
    """MedWikipediaCorpus Dataset. Version 0.0.1"""

    VERSION = datasets.Version(_VERSION)
    BUILDER_CONFIGS = [
        MedWikipediaCorpusConfig(
            name="v1",
            description="Subset of Wikipedia built for the US Medqa, 10 queries per option",
        ),
        MedWikipediaCorpusConfig(
            name="v2",
            description="Subset of Wikipedia built for the US Medqa, 5 queries per option",
        ),
        MedWikipediaCorpusConfig(
            name="v3",
            description="Subset of Wikipedia built for the US+TW Medqa, 10 queries per option",
        ),
        MedWikipediaCorpusConfig(
            name="v3-us",
            description="Subset of Wikipedia built for the US Medqa, 10 queries per option",
        ),
        MedWikipediaCorpusConfig(
            name="v3-tw",
            description="Subset of Wikipedia built for the TW Medqa, 10 queries per option",
        ),
    ]
    force = False

    def _info(self):
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
            citation=_CITATION,
        )

    @staticmethod
    def _get_drive_url(url):
        if "drive.google.com" in url:
            base_url = "https://drive.google.com/uc?id="
            split_url = url.split("/")
            return base_url + split_url[5]
        else:
            return url

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        url = _URLS[self.config.name]
        downloaded_file = dl_manager.download_and_extract(self._get_drive_url(url))
        if not Path(downloaded_file).is_dir():
            raise Exception(
                f"Could not download the dataset Content of `downloaded_file`:"
                f"{open(downloaded_file, 'r').read()}"
            )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"output_dir": Path(downloaded_file)},
            )
        ]

    def _generate_examples(self, output_dir: str):
        # enter the subdirectory
        files = list(Path(output_dir).iterdir())
        rich.print(f">> Found {len(files)} files in {output_dir}: {files[:10]}")

        # move in the subdirectory
        path = [
            p for p in Path(output_dir).iterdir() if p.is_dir() and p.name.startswith("med_x_wiki")
        ][0]

        # list files
        data_files = [os.path.join(path, p) for p in os.listdir(path) if re.findall(TXT_PATTERN, p)]

        # iterate and yield documents
        for i, fn in enumerate(data_files):
            with open(fn, "r") as f:
                # the first line is the title
                title = f.readline()
                text = f.read()
                yield i, {"document.text": text, "document.idx": i, "document.title": title}
