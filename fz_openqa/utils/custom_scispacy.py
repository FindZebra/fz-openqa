import json
from spacy.language import Language


@Language.factory("my_component")
class CustomComponent:
    def __init__(self):
        self.data = []

    def __call__(self, doc):
        # Do something to the doc here
        return doc

    def add(self, data):
        # Add something to the component's data
        self.data.append(data)

    def to_disk(self, path, exclude=tuple()):
        # This will receive the directory path + /my_component
        data_path = path / "data.json"
        with data_path.open("w", encoding="utf8") as f:
            f.write(json.dumps(self.data))

    def from_disk(self, path, exclude=tuple()):
        # This will receive the directory path + /my_component
        data_path = path / "data.json"
        with data_path.open("r", encoding="utf8") as f:
            self.data = json.loads(f)
        return self