import yaml
import joblib
import logging
import json
import numpy as np
from datetime import datetime, date
from pathlib import Path

def load_config(config_path: Path) -> dict:
    print(f"Cargando configuración YAML desde {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_artifact(obj: object, file_path: Path):
    print(f"Guardando artefacto en {file_path}...")
    joblib.dump(obj, file_path)

def load_artifact(file_path: Path) -> object:
    print(f"Cargando artefacto desde {file_path}...")
    return joblib.load(file_path)

def save_run_metadata(metadata: dict, file_path: Path):
    print(f"Guardando metadatos (run.json) en {file_path}...")
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error al crear el directorio {file_path.parent}: {e}")
        return
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=4, cls=NumpyJSONEncoder)

def setup_logging(log_path: Path):
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error al crear el directorio para logs {log_path.parent}: {e}")
        return
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging configurado.")

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (datetime, date)):
             return obj.isoformat()
        return super(NumpyJSONEncoder, self).default(obj)