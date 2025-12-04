"""
"""
from dataclasses import dataclass, asdict

import pandas as pd


@dataclass
class DictLike:
    """
    A base class that mimics dictionary behavior for dataclasses.
    """
    def to_dict(self) -> dict:
        return asdict(self)

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()

    def values(self):
        return self.to_dict().values()

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.to_dict())

    def get(self, key, default=None):
        return self.to_dict().get(key, default)

    def __contains__(self, key):
        return key in self.to_dict()

    def __or__(self, other):
        if isinstance(other, dict):
            return {**self.to_dict(), **other}
        return NotImplemented

@dataclass
class LabelMap(DictLike):
    """
    Stores label mappings for breast cancer subtypes.

    Attributes:
        triple_negative (str): Label for triple negative subtype.
        luminal_a (str): Label for luminal A subtype.
        luminal_b (str): Label for luminal B subtype.
        her2_enriched (str): Label for HER2-enriched subtype.
    """
    triple_negative: str
    luminal_a: str
    luminal_b: str
    her2_enriched: str

@dataclass
class DataSet(DictLike):
    """
    Stores datasets and scalers for pain intensity classification.

    Attributes:
        train_df (pd.DataFrame): Training dataset.
        test_df (pd.DataFrame): Testing dataset.
    """
    train_df: pd.DataFrame
    test_df: pd.DataFrame
