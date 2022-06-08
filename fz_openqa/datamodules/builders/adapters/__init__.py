from fz_openqa.datamodules.builders.adapters.medmcqa import MedMCQAAdapter
from fz_openqa.datamodules.builders.adapters.quality import QualityAdapter
from fz_openqa.datamodules.builders.adapters.race import RaceAdapter

DATASET_ADAPTERS = {
    "race": RaceAdapter,
    "quality": QualityAdapter,
    "medmcqa": MedMCQAAdapter,
}
