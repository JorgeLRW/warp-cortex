from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
LOCAL_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.local.yaml"
_ENV_VAR_RE = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def project_root() -> Path:
    return PROJECT_ROOT


def resolve_project_path(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def get_setting(settings: Dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    value: Any = settings
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def load_settings(config_path: Optional[str] = None) -> Dict[str, Any]:
    settings: Dict[str, Any] = {}

    settings = _deep_merge(settings, _load_yaml(DEFAULT_SETTINGS_PATH))

    if config_path:
        settings = _deep_merge(settings, _load_yaml(Path(config_path)))
    elif LOCAL_SETTINGS_PATH.exists():
        settings = _deep_merge(settings, _load_yaml(LOCAL_SETTINGS_PATH))

    settings = _expand_env(settings)

    hf_cache = resolve_project_path(get_setting(settings, "paths.huggingface_cache"))
    if hf_cache:
        settings.setdefault("paths", {})["huggingface_cache"] = hf_cache

    return settings


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in settings file: {path}")

    return loaded


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(_replace_env_var, value)
    return value


def _replace_env_var(match: re.Match[str]) -> str:
    env_name = match.group(1)
    fallback = match.group(2) or ""
    return os.environ.get(env_name, fallback)