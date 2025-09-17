# Author: wujiahang
"""
Centralized configuration for database credentials.
Priority: Environment variables -> config.json -> defaults.
"""
from __future__ import annotations
import os, json

DEFAULTS = {"host": "127.0.0.1", "port": 3306, "user": "root", "password": "", "name": "posecoach"}

def _load_json():
    path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("db", {})
        except Exception:
            return {}
    return {}

def get_db_settings():
    cfg = DEFAULTS.copy()
    cfg.update(_load_json())
    cfg["host"] = os.getenv("POSE_DB_HOST", cfg["host"])
    cfg["port"] = int(os.getenv("POSE_DB_PORT", cfg["port"]))
    cfg["user"] = os.getenv("POSE_DB_USER", cfg["user"])
    cfg["password"] = os.getenv("POSE_DB_PASSWORD", cfg["password"])
    cfg["name"] = os.getenv("POSE_DB_NAME", cfg["name"])
    return cfg

DB_SETTINGS = get_db_settings()
