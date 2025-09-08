# logging_utils.py
from __future__ import annotations
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Any, Dict, Optional

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "time": datetime.utcfromtimestamp(record.created).isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "name": record.name,
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "msg": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False)

def setup_logging(
    level: Optional[str] = None,
    logfile: Optional[str] = None,
    json_format: bool = True,
) -> None:
    """
    Configura logging global con:
      - Nivel por env: TFM_LOG_LEVEL (por defecto INFO)
      - Fichero por env: TFM_LOG_FILE (por defecto 'run.log')
      - Rotación 5MB x 3
      - Formato JSON por línea (opcional)
    """
    lvl = (level or os.getenv("TFM_LOG_LEVEL") or "INFO").upper()
    file_path = logfile or os.getenv("TFM_LOG_FILE") or "run.log"

    root = logging.getLogger()
    if root.handlers:
        return

    try:
        numeric_level = getattr(logging, lvl)
    except AttributeError:
        numeric_level = logging.INFO

    root.setLevel(numeric_level)

    file_handler = RotatingFileHandler(file_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    if json_format:
        file_handler.setFormatter(JsonFormatter())
    else:
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
    root.addHandler(file_handler)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)
