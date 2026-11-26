"""Modelos tipados e metadados das variáveis climáticas do dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AggregationMode = Literal["mean", "sum"]


@dataclass(frozen=True)
class GlobalFilters:
    """Filtros globais compartilhados entre as abas do dashboard."""

    years: tuple[int, ...] = ()
    regions: tuple[str, ...] = ()
    ufs: tuple[str, ...] = ()
    stations: tuple[str, ...] = ()
    only_brazil_territory: bool = True


@dataclass(frozen=True)
class MetricSpec:
    """Metadados de exibição e agregação para uma variável climática."""

    column: str
    label: str
    short_label: str
    unit: str
    default_agg: AggregationMode
    supports_sum: bool
    colorscale: str


CLIMATE_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        column="temp_c",
        label="Temperatura Média do Ar (°C)",
        short_label="Temperatura",
        unit="°C",
        default_agg="mean",
        supports_sum=False,
        colorscale="RdYlBu_r",
    ),
    MetricSpec(
        column="humidity_pct",
        label="Umidade Relativa Média (%)",
        short_label="Umidade Relativa",
        unit="%",
        default_agg="mean",
        supports_sum=False,
        colorscale="YlGnBu",
    ),
    MetricSpec(
        column="precip_mm",
        label="Precipitação — Soma Mensal (mm)",
        short_label="Precipitação (Soma Mensal)",
        unit="mm",
        default_agg="mean",
        supports_sum=True,
        colorscale="Blues",
    ),
    MetricSpec(
        column="radiation_kj_m2",
        label="Radiação Global — Soma Mensal (kJ/m²)",
        short_label="Radiação Global (Soma Mensal)",
        unit="kJ/m²",
        default_agg="mean",
        supports_sum=True,
        colorscale="YlOrRd",
    ),
)

CLIMATE_METRIC_MAP: dict[str, MetricSpec] = {
    spec.column: spec for spec in CLIMATE_METRICS
}

MONTH_LABELS_PT: dict[int, str] = {
    1: "Jan",
    2: "Fev",
    3: "Mar",
    4: "Abr",
    5: "Mai",
    6: "Jun",
    7: "Jul",
    8: "Ago",
    9: "Set",
    10: "Out",
    11: "Nov",
    12: "Dez",
}
