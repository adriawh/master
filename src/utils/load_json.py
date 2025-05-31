import json
import os
from typing import Any
from utils.logger import setup_logger

logger = setup_logger()

def load_json(base_path: str, filename: str) -> Any:
    file_path = os.path.join(base_path, filename)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"❌ Error loading {filename}: {e}")
        return {}
