# phase3/utils/config_loader.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any, Optional

from utils.common import PROTEIN_WORDS

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None
    _yaml_err = e
else:
    _yaml_err = None

# Paths
_THIS_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _THIS_DIR.parent / "config"

# Defaults if YAML is missing
_DEFAULT_KW = {
    "whitelist": sorted(PROTEIN_WORDS),
    "aliases": {},  # e.g., {"eaas": "eaa"}
}
_DEFAULT_UI = {
    "title": "ProteinScope",
    "tagline": "Protein Market Trends Dashboard",
    "theme": {"base": "light", "primaryColor": "#4C78A8"},
    "cards": {
        "reddit_label": "Reddit",
        "journals_label": "Journals",
        "blogs_label": "Blogs",
    },
}


def _read_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    if yaml is None:
        raise ImportError(
            f"PyYAML not available: {_yaml_err}\n"
            "Install with: pip install pyyaml"
        )
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data


@lru_cache(maxsize=1)
def load_keywords(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Load keyword config (whitelist, aliases).
    Returns dict with keys: 'whitelist' (list[str]), 'aliases' (dict).
    """
    p = Path(path).expanduser() if path else (_CONFIG_DIR / "keywords_protein.yaml")
    data = _read_yaml(p)
    wl = data.get("whitelist") or data.get("keywords") or _DEFAULT_KW["whitelist"]
    al = data.get("aliases") or _DEFAULT_KW["aliases"]
    # sanitize
    wl = sorted({str(x).strip().lower() for x in wl if str(x).strip()})
    al = {str(k).strip().lower(): str(v).strip().lower() for k, v in dict(al).items()}
    return {"whitelist": wl, "aliases": al}


@lru_cache(maxsize=1)
def load_ui(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Load UI config (title, theme, labels…).
    """
    p = Path(path).expanduser() if path else (_CONFIG_DIR / "ui.yaml")
    data = _read_yaml(p)
    if not data:
        return dict(_DEFAULT_UI)
    # merge shallow (defaults <- file)
    out = dict(_DEFAULT_UI)
    out.update(data)
    # nested 'cards'
    if "cards" in data and isinstance(data["cards"], dict):
        out["cards"] = {**_DEFAULT_UI["cards"], **data["cards"]}
    return out


# Convenience accessors used across components
def protein_keywords() -> List[str]:
    """Return the protein keyword whitelist (lowercased)."""
    return load_keywords().get("whitelist", [])


def keyword_aliases() -> Dict[str, str]:
    """Return alias mapping (lowercased)."""
    return load_keywords().get("aliases", {})


def ui_config() -> Dict[str, Any]:
    """Return the merged UI config."""
    return load_ui()


__all__ = [
    "load_keywords",
    "load_ui",
    "protein_keywords",
    "keyword_aliases",
    "ui_config",
]