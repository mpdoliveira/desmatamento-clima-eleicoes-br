"""Funções analíticas puras (sem acoplamento ao Streamlit) para a aba meteorológica."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dashboard.models import AggregationMode, CLIMATE_METRICS, GlobalFilters

CLIMATE_NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    spec.column for spec in CLIMATE_METRICS
)


def _safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def _safe_sum(series: pd.Series) -> float | None:
    if series.empty:
        return None
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.sum())


def filter_climate_dataframe(
    df: pd.DataFrame,
    filters: GlobalFilters,
) -> pd.DataFrame:
    """Aplica os filtros globais ao DataFrame climático com segurança para colunas opcionais."""
    if df.empty:
        return df.copy()

    mask = pd.Series(True, index=df.index)

    if filters.only_brazil_territory and "is_continental_br" in df.columns:
        mask &= df["is_continental_br"].fillna(False).astype(bool)

    if filters.years and "ANO" in df.columns:
        mask &= df["ANO"].isin(list(filters.years))

    if filters.regions and "REGIAO" in df.columns:
        mask &= df["REGIAO"].isin(list(filters.regions))

    if filters.ufs and "UF" in df.columns:
        mask &= df["UF"].isin(list(filters.ufs))

    if filters.stations and "station_label" in df.columns:
        mask &= df["station_label"].isin(list(filters.stations))

    return df.loc[mask].copy()


def compute_climate_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Calcula KPIs resumidos do recorte climático atual, tratando NaNs e DataFrames vazios."""
    if df.empty:
        return {
            "records_count": 0,
            "stations_count": 0,
            "years_count": 0,
            "ufs_count": 0,
            "temp_mean": None,
            "humidity_mean": None,
            "precip_monthly_mean": None,
            "precip_total": None,
            "radiation_monthly_mean": None,
        }

    if "station_id" in df.columns:
        stations_count = int(df["station_id"].dropna().nunique())
    elif {"LATITUDE", "LONGITUDE"}.issubset(df.columns):
        stations_count = int(df[["LATITUDE", "LONGITUDE"]].dropna().drop_duplicates().shape[0])
    else:
        stations_count = 0

    years_count = int(df["ANO"].dropna().nunique()) if "ANO" in df.columns else 0
    ufs_count = int(df["UF"].dropna().nunique()) if "UF" in df.columns else 0

    return {
        "records_count": int(len(df)),
        "stations_count": stations_count,
        "years_count": years_count,
        "ufs_count": ufs_count,
        "temp_mean": _safe_mean(df["temp_c"]) if "temp_c" in df.columns else None,
        "humidity_mean": _safe_mean(df["humidity_pct"]) if "humidity_pct" in df.columns else None,
        "precip_monthly_mean": _safe_mean(df["precip_mm"]) if "precip_mm" in df.columns else None,
        "precip_total": _safe_sum(df["precip_mm"]) if "precip_mm" in df.columns else None,
        "radiation_monthly_mean": (
            _safe_mean(df["radiation_kj_m2"]) if "radiation_kj_m2" in df.columns else None
        ),
    }


def _ensure_station_key(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "station_id" not in working.columns:
        if {"LATITUDE", "LONGITUDE"}.issubset(working.columns):
            working["station_id"] = (
                working["LATITUDE"].round(4).astype(str)
                + ","
                + working["LONGITUDE"].round(4).astype(str)
            )
        else:
            working["station_id"] = working.index.astype(str)
    return working


def aggregate_stations_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega dados por estação e ano para o mapa Folium com camadas anuais."""
    expected_cols = [
        "ANO",
        "station_id",
        "LATITUDE",
        "LONGITUDE",
        "months_count",
        "temp_c",
        "humidity_pct",
        "precip_mm_sum",
        "precip_mm_mean",
        "radiation_kj_m2_sum",
        "radiation_kj_m2_mean",
    ]
    if df.empty or not {"ANO", "LATITUDE", "LONGITUDE"}.issubset(df.columns):
        return pd.DataFrame(columns=expected_cols)

    working = _ensure_station_key(df).dropna(subset=["ANO", "LATITUDE", "LONGITUDE"])
    if working.empty:
        return pd.DataFrame(columns=expected_cols)

    group_cols = ["ANO", "station_id", "LATITUDE", "LONGITUDE"]
    agg_spec: dict[str, tuple[str, Any]] = {}

    if "MES" in working.columns:
        agg_spec["months_count"] = ("MES", "nunique")
    else:
        agg_spec["months_count"] = ("ANO", "size")

    if "temp_c" in working.columns:
        agg_spec["temp_c"] = ("temp_c", "mean")
    if "humidity_pct" in working.columns:
        agg_spec["humidity_pct"] = ("humidity_pct", "mean")
    if "precip_mm" in working.columns:
        agg_spec["precip_mm_sum"] = ("precip_mm", lambda s: s.sum(min_count=1))
        agg_spec["precip_mm_mean"] = ("precip_mm", "mean")
    if "radiation_kj_m2" in working.columns:
        agg_spec["radiation_kj_m2_sum"] = ("radiation_kj_m2", lambda s: s.sum(min_count=1))
        agg_spec["radiation_kj_m2_mean"] = ("radiation_kj_m2", "mean")

    for opt_col in ("ESTACAO", "UF", "REGIAO", "WMO", "source_file", "station_label"):
        if opt_col in working.columns:
            agg_spec[opt_col] = (opt_col, "first")

    grouped = working.groupby(group_cols, as_index=False, dropna=False).agg(**agg_spec)
    for col in ("temp_c", "humidity_pct", "precip_mm_sum", "precip_mm_mean", "radiation_kj_m2_sum", "radiation_kj_m2_mean"):
        if col not in grouped.columns:
            grouped[col] = np.nan

    return grouped.sort_values(["ANO", "station_id"]).reset_index(drop=True)


def aggregate_stations_spatial(
    df: pd.DataFrame,
    precip_agg: AggregationMode = "mean",
    radiation_agg: AggregationMode = "mean",
) -> pd.DataFrame:
    """Agrega todo o período filtrado por estação física para os mapas espaciais das 4 variáveis."""
    expected_cols = [
        "station_id",
        "LATITUDE",
        "LONGITUDE",
        "months_count",
        "years_count",
        "temp_c",
        "humidity_pct",
        "precip_mm",
        "precip_mm_mean",
        "precip_mm_sum",
        "radiation_kj_m2",
        "radiation_kj_m2_mean",
        "radiation_kj_m2_sum",
    ]
    if df.empty or not {"LATITUDE", "LONGITUDE"}.issubset(df.columns):
        return pd.DataFrame(columns=expected_cols)

    working = _ensure_station_key(df).dropna(subset=["LATITUDE", "LONGITUDE"])
    if working.empty:
        return pd.DataFrame(columns=expected_cols)

    if "ANO" in working.columns:
        working = working.sort_values("ANO", ascending=False)

    agg_spec: dict[str, tuple[str, Any]] = {
        "LATITUDE": ("LATITUDE", "first"),
        "LONGITUDE": ("LONGITUDE", "first"),
        "months_count": ("MES", "count") if "MES" in working.columns else ("LATITUDE", "size"),
    }
    if "ANO" in working.columns:
        agg_spec["years_count"] = ("ANO", "nunique")

    if "temp_c" in working.columns:
        agg_spec["temp_c"] = ("temp_c", "mean")
    if "humidity_pct" in working.columns:
        agg_spec["humidity_pct"] = ("humidity_pct", "mean")
    if "precip_mm" in working.columns:
        agg_spec["precip_mm_mean"] = ("precip_mm", "mean")
        agg_spec["precip_mm_sum"] = ("precip_mm", lambda s: s.sum(min_count=1))
    if "radiation_kj_m2" in working.columns:
        agg_spec["radiation_kj_m2_mean"] = ("radiation_kj_m2", "mean")
        agg_spec["radiation_kj_m2_sum"] = ("radiation_kj_m2", lambda s: s.sum(min_count=1))

    for opt_col in ("ESTACAO", "UF", "REGIAO", "WMO", "station_label"):
        if opt_col in working.columns:
            agg_spec[opt_col] = (opt_col, "first")

    grouped = working.groupby("station_id", as_index=False, dropna=False).agg(**agg_spec)
    if "years_count" not in grouped.columns:
        grouped["years_count"] = 1

    for col in (
        "temp_c",
        "humidity_pct",
        "precip_mm_mean",
        "precip_mm_sum",
        "radiation_kj_m2_mean",
        "radiation_kj_m2_sum",
    ):
        if col not in grouped.columns:
            grouped[col] = np.nan

    grouped["precip_mm"] = (
        grouped["precip_mm_sum"] if precip_agg == "sum" else grouped["precip_mm_mean"]
    )
    grouped["radiation_kj_m2"] = (
        grouped["radiation_kj_m2_sum"]
        if radiation_agg == "sum"
        else grouped["radiation_kj_m2_mean"]
    )
    if "station_label" not in grouped.columns:
        grouped["station_label"] = grouped["station_id"].astype(str)

    return grouped.reset_index(drop=True)


def compute_monthly_seasonality(
    df: pd.DataFrame,
    variable: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (série por ANO x MES, média climatológica por MES) para a variável escolhida."""
    if df.empty or variable not in df.columns or not {"ANO", "MES"}.issubset(df.columns):
        return (
            pd.DataFrame(columns=["ANO", "MES", "value", "stations_count"]),
            pd.DataFrame(columns=["MES", "mean_value"]),
        )

    valid = df.dropna(subset=["ANO", "MES", variable]).copy()
    if valid.empty:
        return (
            pd.DataFrame(columns=["ANO", "MES", "value", "stations_count"]),
            pd.DataFrame(columns=["MES", "mean_value"]),
        )

    by_year_month = (
        valid.groupby(["ANO", "MES"], as_index=False)
        .agg(
            value=(variable, "mean"),
            stations_count=(variable, "count"),
        )
        .sort_values(["ANO", "MES"])
    )
    by_year_month["ANO_STR"] = by_year_month["ANO"].astype(int).astype(str)

    climatology = (
        by_year_month.groupby("MES", as_index=False)
        .agg(mean_value=("value", "mean"))
        .sort_values("MES")
    )
    return by_year_month, climatology


def compute_annual_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a média anual das variáveis climáticas e a cobertura de estações."""
    if df.empty or "ANO" not in df.columns:
        return pd.DataFrame(columns=["ANO", "stations_count", *CLIMATE_NUMERIC_COLUMNS])

    working = _ensure_station_key(df).dropna(subset=["ANO"])
    if working.empty:
        return pd.DataFrame(columns=["ANO", "stations_count", *CLIMATE_NUMERIC_COLUMNS])

    agg_spec: dict[str, tuple[str, Any]] = {"stations_count": ("station_id", "nunique")}
    for col in CLIMATE_NUMERIC_COLUMNS:
        if col in working.columns:
            agg_spec[col] = (col, "mean")

    trend = working.groupby("ANO", as_index=False).agg(**agg_spec).sort_values("ANO")
    for col in CLIMATE_NUMERIC_COLUMNS:
        if col not in trend.columns:
            trend[col] = np.nan
    return trend.reset_index(drop=True)


def compute_uf_summary(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Calcula resumo por UF para a variável informada (seguro caso UF esteja ausente)."""
    if df.empty or "UF" not in df.columns or variable not in df.columns:
        return pd.DataFrame(columns=["UF", "mean_value", "stations_count", "records_count"])

    working = _ensure_station_key(df)
    valid = working.dropna(subset=["UF", variable])
    if valid.empty:
        return pd.DataFrame(columns=["UF", "mean_value", "stations_count", "records_count"])

    summary = (
        valid.groupby("UF", as_index=False)
        .agg(
            mean_value=(variable, "mean"),
            stations_count=("station_id", "nunique"),
            records_count=(variable, "count"),
        )
        .sort_values("mean_value", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def compute_correlation_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula a matriz de correlação de Pearson e a contagem de pares válidos entre variáveis."""
    available = [col for col in CLIMATE_NUMERIC_COLUMNS if col in df.columns]
    if df.empty or len(available) < 2:
        return pd.DataFrame(), pd.DataFrame()

    numeric_df = df[available].apply(pd.to_numeric, errors="coerce")
    non_empty_cols = [col for col in available if numeric_df[col].notna().sum() >= 2]
    if len(non_empty_cols) < 2:
        return pd.DataFrame(), pd.DataFrame()

    numeric_df = numeric_df[non_empty_cols]
    corr = numeric_df.corr(method="pearson", min_periods=2)

    valid_mask = numeric_df.notna().astype(int)
    pair_counts = valid_mask.T @ valid_mask
    return corr, pair_counts
