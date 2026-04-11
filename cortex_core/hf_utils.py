import os
from pathlib import Path
from typing import Optional


def _ensure_usable_dir(path_str: str) -> bool:
    if not path_str:
        return False

    try:
        path = Path(path_str).expanduser()
        if path.drive and not Path(f"{path.drive}{os.sep}").exists():
            return False
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    return True


def _normalize_cache_root(path_str: str) -> str:
    candidate_path = Path(path_str).expanduser()
    if candidate_path.name.lower() in {"hub", "transformers", "datasets"}:
        return str(candidate_path.parent)
    return str(candidate_path)


def prepare_hf_cache(base_dir: str, preferred_root: Optional[str] = None) -> str:
    """
    Return a usable Hugging Face cache directory.

    If the current environment points at a missing drive or inaccessible path,
    fall back to a repo-local cache under local_artifacts/ so standalone scripts
    remain runnable on clean machines.
    """
    cache_root = None
    if preferred_root and _ensure_usable_dir(preferred_root):
        cache_root = _normalize_cache_root(preferred_root)

    if cache_root is None:
        env_candidates = [
            os.environ.get("HF_HOME"),
            os.environ.get("HUGGINGFACE_HUB_CACHE"),
            os.environ.get("HF_HUB_CACHE"),
            os.environ.get("TRANSFORMERS_CACHE"),
            os.environ.get("HF_DATASETS_CACHE"),
        ]

        for candidate in env_candidates:
            if not candidate or not _ensure_usable_dir(candidate):
                continue
            cache_root = _normalize_cache_root(candidate)
            break

    if cache_root is None:
        cache_root = os.path.join(base_dir, "local_artifacts", "huggingface")

    hub_dir = os.path.join(cache_root, "hub")
    transformers_dir = os.path.join(cache_root, "transformers")
    datasets_dir = os.path.join(cache_root, "datasets")
    _ensure_usable_dir(hub_dir)
    _ensure_usable_dir(transformers_dir)
    _ensure_usable_dir(datasets_dir)

    os.environ["HF_HOME"] = cache_root
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_dir
    os.environ["HF_HUB_CACHE"] = hub_dir
    os.environ["TRANSFORMERS_CACHE"] = transformers_dir
    os.environ["HF_DATASETS_CACHE"] = datasets_dir
    return cache_root