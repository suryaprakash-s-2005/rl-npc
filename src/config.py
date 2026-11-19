"""Configuration loading and command-line override support."""
import copy
import os
import sys

try:
    import yaml
except Exception:  # fallback if pyyaml unavailable
    yaml = None

from .defaults import DEFAULTS


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_override(parts):
    """Parse flat key=value pairs into a nested dict."""
    result = {}
    for pair in parts:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                low = str(value).strip().lower()
                if low in ("true", "yes", "on"):
                    value = True
                elif low in ("false", "no", "off"):
                    value = False
                elif low in ("none", "null"):
                    value = None
                else:
                    value = value
        node = result
        keys = key.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return result


def load_config(argv=None):
    """Build config: defaults merged with cli overrides.

    Supports `--key value` style overrides such as
        --rl.episodes 300 --seed 1
    """
    argv = sys.argv[1:] if argv is None else argv
    overrides = {}
    # support --key value and --key=value
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            key = tok[2:]
            value = None
            if "=" in key:
                key, value = key.split("=", 1)
            else:
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    value = argv[i + 1]
                    i += 1
                else:
                    value = True
            overrides = _merge(overrides, _parse_override([f"{key}={value}"]))
        i += 1
    config = _merge(DEFAULTS, overrides)
    return config
