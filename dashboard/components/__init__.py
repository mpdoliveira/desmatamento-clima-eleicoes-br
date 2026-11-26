"""Componentes modulares das abas do dashboard (importação preguiçosa)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "render_climate_tab",
    "render_elections_tab",
    "render_deforestation_tab",
    "render_combined_tab",
]

_TAB_MODULE_MAP = {
    "render_climate_tab": "dashboard.components.climate_tab",
    "render_elections_tab": "dashboard.components.elections_tab",
    "render_deforestation_tab": "dashboard.components.deforestation_tab",
    "render_combined_tab": "dashboard.components.combined_tab",
}


def __getattr__(name: str) -> Any:
    module_path = _TAB_MODULE_MAP.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = __import__(module_path, fromlist=[name])
    return getattr(module, name)
