# src/trainers/__init__.py

import logging
from .dcd_trainer import DCDTrainer
from .ntp_trainer import NTPTrainer

def get_trainer(config, **kwargs):
    """
    Returns the appropriate Trainer instance based on 'training.method' in the config.
    """
    method = config['training']['method']
    logging.info(f"Initializing trainer for '{method}' training strategy.")

    if method == "dcd":
        return DCDTrainer(config, **kwargs)
    elif method == "ntp":
        return NTPTrainer(config, **kwargs)
    else:
        raise ValueError(f"Unsupported training method: {method}")
