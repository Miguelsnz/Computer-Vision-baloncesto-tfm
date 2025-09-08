# run_manager.py
from __future__ import annotations
import os, sys, json, time, random, platform, importlib
from datetime import datetime
from typing import Optional, Dict

try:
    import numpy as np
except Exception:
    np = None  # el módulo sigue funcionando aunque numpy no esté disponible

# Estado global mínimo del "run" actual
_RUN: Dict[str, str] = {}

def _timestamp() -> str:
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    ms = f"{int((time.time() % 1) * 1000):03d}"
    return f"{base}-{ms}"

def init_run(root: str = "runs", run_id: Optional[str] = None, seed: int = 42) -> str:
    """
    Crea la estructura runs/<run_id>/{plots,metrics,video}, fija semillas y
    guarda metadatos en metrics/run_info.json. Devuelve la ruta del run.
    """
    global _RUN
    if _RUN:
        return _RUN["run_dir"]

    rid = run_id or _timestamp()
    run_dir = os.path.join(root, rid)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "video"), exist_ok=True)
    _RUN = {"run_dir": run_dir}

    set_seeds(seed)

    meta = {
        "run_id": rid,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": sys.argv,
        "seed": seed,
        "libs": {
            "numpy": _ver("numpy"),
            "opencv": _ver_cv2(),
            "torch": _ver("torch"),
            "detectron2": _ver("detectron2"),
            "matplotlib": _ver("matplotlib"),
        },
        # Variables de entorno del proyecto (opcional)
        "env": {k: v for k, v in os.environ.items() if k.startswith("TFM_")},
    }
    _dump_json(meta, os.path.join(run_dir, "metrics", "run_info.json"))
    return run_dir

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # No forzamos determinismo estricto porque puede chocar con OpenCV/Detectron2
        # torch.use_deterministic_algorithms(False)
    except Exception:
        pass

def run_dir() -> str:
    if not _RUN:
        raise RuntimeError("init_run() no ha sido llamado todavía.")
    return _RUN["run_dir"]

def path(*parts: str) -> str:
    """Construye una ruta dentro del run actual."""
    return os.path.join(run_dir(), *parts)

def plots_dir() -> str:
    return path("plots")

def metrics_dir() -> str:
    return path("metrics")

def video_dir() -> str:
    return path("video")

def video_path(filename: str = "demo.mp4") -> str:
    return os.path.join(video_dir(), filename)

def save_metrics(payload: dict, name: str = "metrics.json") -> str:
    """Guarda un JSON de métricas en runs/<id>/metrics/<name>."""
    out = os.path.join(metrics_dir(), name)
    _dump_json(payload, out)
    return out

def append_scalar(name: str, value, file: str = "scalars.jsonl") -> str:
    """
    Añade una línea JSON con un escalar (serie temporal simple).
    Útil para contadores por frame, FPS, etc.
    """
    out = os.path.join(metrics_dir(), file)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"name": name, "value": value, "t": time.time()}) + "\n")
    return out

# --- Utilidades internas ---
def _ver(mod: str) -> Optional[str]:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", None)
    except Exception:
        return None

def _ver_cv2() -> Optional[str]:
    try:
        import cv2
        return cv2.__version__
    except Exception:
        return None

def _dump_json(obj: dict, fpath: str) -> None:
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
