"""Carregamento e enriquecimento dos dados meteorológicos do INMET."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.paths import PROCESSED_DATA_DIR

INMET_PARQUET_PATH = PROCESSED_DATA_DIR / "inmet" / "inmet.parquet"
INMET_METADATA_PATH = PROCESSED_DATA_DIR / "inmet" / "metadata.json"

_YEAR_SUFFIX_RE = re.compile(r"(\d{4})$")


def parse_brazilian_float(value: Any) -> float | None:
    """Converte strings numéricas com vírgula decimal para float, retornando None se inválido."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "null", "none", "-9999"}:
        return None
    normalized = text.replace(",", ".")
    try:
        parsed = float(normalized)
    except ValueError:
        return None
    return None if np.isnan(parsed) else parsed


def extract_year_from_metadata_key(key: str) -> int | None:
    """Extrai o ano final da chave de metadados do INMET (ex.: '..._31-12-2019')."""
    match = _YEAR_SUFFIX_RE.search(key.strip())
    if not match:
        return None
    return int(match.group(1))


def _first_present(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        val = mapping.get(key)
        if val is not None:
            cleaned = str(val).strip()
            if cleaned:
                return cleaned
    return None


def build_metadata_dataframe(raw_meta: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Transforma o dicionário metadata.json do INMET em DataFrame normalizado."""
    columns = [
        "source_file",
        "ANO",
        "LATITUDE",
        "LONGITUDE",
        "ALTITUDE_M",
        "UF",
        "REGIAO",
        "ESTACAO",
        "WMO",
        "DATA_FUNDACAO",
    ]
    if not raw_meta:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for key, payload in raw_meta.items():
        if not isinstance(payload, dict):
            continue
        year = extract_year_from_metadata_key(key)
        lat = parse_brazilian_float(payload.get("LATITUDE"))
        lon = parse_brazilian_float(payload.get("LONGITUDE"))
        alt = parse_brazilian_float(payload.get("ALTITUDE"))
        rows.append(
            {
                "source_file": key,
                "ANO": year,
                "LATITUDE": lat,
                "LONGITUDE": lon,
                "ALTITUDE_M": alt,
                "UF": _first_present(payload, "UF"),
                "REGIAO": _first_present(payload, "REGIÃO", "REGIAO", "REGI?O"),
                "ESTACAO": _first_present(payload, "ESTAÇÃO", "ESTACAO", "ESTAC?O"),
                "WMO": _first_present(payload, "CODIGO (WMO)"),
                "DATA_FUNDACAO": _first_present(
                    payload,
                    "DATA DE FUNDAÇÃO (YYYY-MM-DD)",
                    "DATA DE FUNDACAO",
                    "DATA DE FUNDAC?O",
                ),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    meta_df = pd.DataFrame(rows, columns=columns)
    meta_df["ANO"] = pd.to_numeric(meta_df["ANO"], errors="coerce").astype("Int64")
    return meta_df


def enrich_inmet_with_metadata(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """Enriquece os dados mensais do INMET com informações cadastrais das estações."""
    result = df.copy()
    if result.empty:
        for col in ("source_file", "UF", "REGIAO", "ESTACAO", "WMO", "ALTITUDE_M", "station_id", "station_label"):
            if col not in result.columns:
                result[col] = pd.Series(dtype="object")
        result["is_continental_br"] = pd.Series(dtype="bool")
        return result

    for coord_col in ("LATITUDE", "LONGITUDE"):
        if coord_col in result.columns:
            result[coord_col] = pd.to_numeric(result[coord_col], errors="coerce")

    if "ANO" in result.columns:
        result["ANO"] = pd.to_numeric(result["ANO"], errors="coerce").astype("Int64")
    if "MES" in result.columns:
        result["MES"] = pd.to_numeric(result["MES"], errors="coerce").astype("Int64")

    meta_cols = ["source_file", "UF", "REGIAO", "ESTACAO", "WMO", "ALTITUDE_M"]

    if not meta_df.empty and {"ANO", "LATITUDE", "LONGITUDE"}.issubset(result.columns):
        dedup_meta = (
            meta_df.dropna(subset=["ANO", "LATITUDE", "LONGITUDE"])
            .drop_duplicates(subset=["ANO", "LATITUDE", "LONGITUDE"])
            .copy()
        )
        dedup_meta["ANO"] = dedup_meta["ANO"].astype("Int64")

        result = result.merge(
            dedup_meta[["ANO", "LATITUDE", "LONGITUDE", *meta_cols]],
            on=["ANO", "LATITUDE", "LONGITUDE"],
            how="left",
        )

        # Fallback por coordenada arredondada caso haja diferença mínima de ponto flutuante
        missing_mask = result["WMO"].isna() & result["ESTACAO"].isna()
        if missing_mask.any():
            dedup_meta["_lat_r"] = dedup_meta["LATITUDE"].round(4)
            dedup_meta["_lon_r"] = dedup_meta["LONGITUDE"].round(4)
            fallback_meta = dedup_meta.drop_duplicates(subset=["ANO", "_lat_r", "_lon_r"])

            unmatched = result.loc[missing_mask, ["ANO", "LATITUDE", "LONGITUDE"]].copy()
            unmatched["_orig_idx"] = unmatched.index
            unmatched["_lat_r"] = unmatched["LATITUDE"].round(4)
            unmatched["_lon_r"] = unmatched["LONGITUDE"].round(4)
            matched = unmatched.merge(
                fallback_meta[["ANO", "_lat_r", "_lon_r", *meta_cols]],
                on=["ANO", "_lat_r", "_lon_r"],
                how="left",
            ).set_index("_orig_idx")

            for col in meta_cols:
                result.loc[missing_mask, col] = result.loc[missing_mask, col].fillna(
                    matched[col]
                )
    else:
        for col in meta_cols:
            if col not in result.columns:
                result[col] = pd.NA

    coord_fallback = (
        result["LATITUDE"].round(4).astype(str)
        + ","
        + result["LONGITUDE"].round(4).astype(str)
    )
    wmo_series = result["WMO"].astype("string").str.strip()
    result["station_id"] = wmo_series.where(
        wmo_series.notna() & (wmo_series != ""),
        coord_fallback,
    )

    station_name = result["ESTACAO"].astype("string").fillna("Estação sem nome").str.strip()
    uf_clean = result["UF"].astype("string").fillna("").str.strip()
    wmo_clean = result["WMO"].astype("string").fillna("").str.strip()

    label = station_name.copy()
    has_wmo = wmo_clean != ""
    has_uf = uf_clean != ""
    label = label.where(~has_wmo, label + " (" + wmo_clean + ")")
    label = label.where(~has_uf, label + " - " + uf_clean)
    result["station_label"] = label

    if {"LATITUDE", "LONGITUDE"}.issubset(result.columns):
        result["is_continental_br"] = (
            result["LATITUDE"].between(-34.5, 5.5, inclusive="both")
            & result["LONGITUDE"].between(-74.5, -28.0, inclusive="both")
        ).fillna(False)
    else:
        result["is_continental_br"] = False

    return result


def load_inmet_data_from_disk(
    parquet_path: Path = INMET_PARQUET_PATH,
    metadata_path: Path = INMET_METADATA_PATH,
) -> pd.DataFrame:
    """Lê os arquivos processados do INMET do disco e retorna DataFrame enriquecido."""
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Arquivo de dados processados não encontrado: {parquet_path}"
        )

    df = pd.read_parquet(parquet_path)

    if metadata_path.exists():
        raw_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        meta_df = build_metadata_dataframe(raw_meta)
    else:
        meta_df = build_metadata_dataframe({})

    return enrich_inmet_with_metadata(df, meta_df)
