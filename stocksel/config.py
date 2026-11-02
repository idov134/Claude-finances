"""Single-file configuration loader with validation and hashing.

The whole system reads one YAML file. ``Config`` gives attribute access
(``cfg.portfolio.top_n``) while staying a plain nested dict underneath so it can
be hashed and logged into the trial registry.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


class Config(dict):
    """dict with attribute access; nested dicts are wrapped lazily."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - attribute error path
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self, default=str))

    def copy_with(self, overrides: dict) -> "Config":
        """Deep copy with dotted-key overrides, e.g. {"portfolio.top_n": 30}."""
        new = copy.deepcopy(self.to_dict())
        for dotted, val in overrides.items():
            node = new
            parts = dotted.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = val
        cfg = Config(new)
        validate(cfg)
        return cfg

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: str | Path = "config.yaml", overrides: dict | None = None) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(raw)
    if overrides:
        cfg = cfg.copy_with(overrides)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Hard rules from the research brief. Violations raise instead of warn."""
    if cfg.validation.holdout_months < 24:
        raise ConfigError("validation.holdout_months must be >= 24 (two-year one-shot holdout)")
    if not 0 < cfg.portfolio.kelly_fraction <= 0.25:
        raise ConfigError("portfolio.kelly_fraction must be in (0, 0.25] (at most quarter Kelly)")
    if not 0 < cfg.capital.max_drawdown < 1:
        raise ConfigError("capital.max_drawdown must be a fraction in (0, 1)")
    if cfg.portfolio.max_gross > 1.0 + 1e-9:
        raise ConfigError("portfolio.max_gross > 1 means leverage; not supported by default")
    if not 0 <= cfg.signals.haircut < 1:
        raise ConfigError("signals.haircut must be in [0, 1)")
    if cfg.models.baseline not in cfg.models.candidates:
        raise ConfigError("the baseline model must be among models.candidates")
    if cfg.universe.min_market_cap < 3e8:
        raise ConfigError("universe.min_market_cap below $300M admits micro-caps; refused")
    for name in cfg.features.sets.get("all", []):
        if name not in cfg.signals.registry:
            raise ConfigError(f"feature {name!r} lacks signals.registry metadata (needed for the haircut)")


def vol_target(cfg: Config) -> float:
    vt = cfg.portfolio.get("vol_target")
    return float(vt) if vt else float(cfg.capital.max_drawdown) / 2.0


def kill_drawdown(cfg: Config) -> float:
    dd = cfg.risk.kill_switch.get("drawdown")
    return float(dd) if dd else float(cfg.capital.max_drawdown)
