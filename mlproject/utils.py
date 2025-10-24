import yaml
import joblib
import logging
import json
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
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def setup_logging(log_path: Path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging configurado.")