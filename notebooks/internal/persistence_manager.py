"""
Utility class for persisting datasets and scalers using joblib.
"""
import os
from typing import Literal, Union

import joblib
from internal.data_types import DataSet


class PersistenceManager:
    """
    Manages the persistence of datasets and scalers using joblib.
    """
    # full path to the current directory
    NOTEBOOKS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = f'{NOTEBOOKS_PATH}/processed/dataset.joblib'

    @staticmethod
    def _makedirs_for_file(file_path: str) -> None:
        """
        Ensures that the directory for the given file path exists.
        :param file_path: The file path for which to create directories.
        """
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def load_dataset() -> DataSet:
        """
        Loads the dataset containing arrays and scalers from the specified path.
        :return: DataSet object with loaded data.
        """
        PersistenceManager._makedirs_for_file(PersistenceManager.DATASET_PATH)
        return (
                print(f"Arrays and scalers loaded successfully from: {PersistenceManager.DATASET_PATH}") or
                DataSet(**joblib.load(PersistenceManager.DATASET_PATH))
        )

    @staticmethod
    def save_dataset(data: Union[DataSet, dict], compress: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]=0) -> None:
        """
        Saves the dataset containing arrays and scalers to the specified path.
        :param data: DataSet object to be saved.
        :param compress: Compression level for joblib dump (0-9). Default is 0 (no compression).
        """
        PersistenceManager._makedirs_for_file(PersistenceManager.DATASET_PATH)
        joblib.dump(
            dict(data) if isinstance(data, DataSet) else data,
            PersistenceManager.DATASET_PATH,
            compress=compress
        )
        print(f"Arrays and scalers saved successfully to: {PersistenceManager.DATASET_PATH}")