# src/datasets/__init__.py

import logging
from typing import Dict, Any

from .phonebook import PhoneBookLoader
from .base_loader import CsvLoader, JsonlLoader

# Register available dataset loaders
AVAILABLE_DATASETS = {
    "phonebook": PhoneBookLoader,
    "counterfact": CsvLoader,
    "thesis": CsvLoader,
    "quality": JsonlLoader,
    "paperqa": JsonlLoader,
    "narrativeqa": JsonlLoader,
}

def get_dataset_loader(config: Dict[str, Any]):
    dataset_config = config.get('dataset', {})
    dataset_name = dataset_config.get('name')

    if not dataset_name:
        raise ValueError("'dataset.name' is not specified in the configuration.")

    logging.info(f"Initializing dataset loader for '{dataset_name}'.")

    loader_class = AVAILABLE_DATASETS.get(dataset_name)
    if loader_class:
        return loader_class(config)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")