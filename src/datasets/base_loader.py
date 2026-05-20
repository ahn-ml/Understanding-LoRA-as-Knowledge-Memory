# src/datasets/base_loader.py

import os
import logging
import json
from typing import List, Dict, Any

import pandas as pd
from datasets import load_dataset, Dataset

class CsvLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_config = config.get('dataset', {})
        self.data_file_path = self.dataset_config.get('data_file_path')

    def load_data(self) -> List[Dict[str, Any]]:
        if not self.data_file_path or not os.path.exists(self.data_file_path):
            logging.warning(f"Data file not found: {self.data_file_path}. Returning empty list.")
            return []
        
        df = pd.read_csv(self.data_file_path)
        # NTPTrainer expects a list of dictionaries in the format {'text': ...}
        return df.to_dict('records')

class JsonlLoader:
    """
    Loader for .jsonl files.
    Each line must be a JSON object, ideally in the format {"text": "...", "metadata": {...}}.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_config = config.get('dataset', {})
        self.data_file_path = self.dataset_config.get('data_file_path')

    def load_data(self) -> List[Dict[str, Any]]:
        """Reads a JSON Lines file and extracts only the 'text' field."""
        if not self.data_file_path or not os.path.exists(self.data_file_path):
            logging.warning(f"Data file not found: {self.data_file_path}. Returning empty list.")
            return []
        
        data_records = []
        with open(self.data_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Only pass dictionaries with the 'text' key to match NTPTrainer's input format
                    if 'text' in record:
                        data_records.append({'text': record['text']})
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed JSON line in {self.data_file_path}: {line.strip()}")
        return data_records