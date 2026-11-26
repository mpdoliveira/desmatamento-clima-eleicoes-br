"""Ponto de entrada do dashboard Streamlit para análise de Clima, Desmatamento e Eleições."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Garante resolução de `src` e `dashboard` quando executado via `streamlit run dashboard/app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.climate_tab import render_climate_tab  # noqa: E402
from dashboard.components.combined_tab import render_combined_tab  # noqa: E402
from dashboard.components.deforestation_tab import render_deforestation_tab  # noqa: E402
from dashboard.components.elections_tab import render_elections_tab  # noqa: E402
from dashboard.data_loader import load_inmet_data_from_disk  # noqa: E402
from dashboard.models import GlobalFilters  # noqa: E402


@st.cache_data(show_spinner="Carregando dados meteorológicos do INMET...")
def get_inmet_dataset() -> pd.DataFrame:
    """Carrega e mantém em cache o dataset meteorológico enriquecido."""
    return load_inmet_data_from_disk()


def _render_sidebar_filters(df: pd.DataFrame) -> GlobalFilters:
    st.sidebar.header("Filtros Globais")

    only_brazil = st.sidebar.checkbox(
        "Restringir ao território brasileiro",
        value=True,
        help="Exclui estações fora da área continental/insular brasileira (ex.: módulo antártico Criosfera).",
    )

    scoped = df
    if only_brazil and "is_continental_br" in df.columns:
        scoped = df[df["is_continental_br"].fillna(False)]

    available_years = (
        sorted(scoped["ANO"].dropna().astype(int).unique().tolist())
        if "ANO" in scoped.columns
        else []
    )

    year_mode = st.sidebar.radio(
        "Recorte temporal (Anos)",
        options=["2026", "Últimos 5 anos", "Todos os anos", "Seleção manual"],
        index=0,
    )
    if year_mode == "2026":
        if 2026 in available_years:
            selected_years = [2026]
        else:
            selected_years = available_years[-1:] if available_years else []
    elif year_mode == "Últimos 5 anos":
        selected_years = available_years[-5:] if available_years else []
    elif year_mode == "Todos os anos":
        selected_years = available_years
    else:
        default_manual = available_years[-3:] if available_years else []
        selected_years = st.sidebar.multiselect(
            "Anos",
            options=available_years,
            default=default_manual,
        )

    if selected_years and "ANO" in scoped.columns:
        scoped = scoped[scoped["ANO"].isin(selected_years)]

    available_regions = (
        sorted(scoped["REGIAO"].dropna().astype(str).unique().tolist())
        if "REGIAO" in scoped.columns
        else []
    )
    selected_regions = (
        st.sidebar.multiselect(
            "Regiões",
            options=available_regions,
            default=[],
            placeholder="Todas as regiões",
        )
        if available_regions
        else []
    )
    if selected_regions and "REGIAO" in scoped.columns:
        scoped = scoped[scoped["REGIAO"].isin(selected_regions)]

    available_ufs = (
        sorted(scoped["UF"].dropna().astype(str).unique().tolist())
        if "UF" in scoped.columns
        else []
    )
    selected_ufs = (
        st.sidebar.multiselect(
            "Estados (UF)",
            options=available_ufs,
            default=[],
            placeholder="Todas as UFs",
        )
        if available_ufs
        else []
    )
    if selected_ufs and "UF" in scoped.columns:
        scoped = scoped[scoped["UF"].isin(selected_ufs)]

    available_stations = (
        sorted(scoped["station_label"].dropna().astype(str).unique().tolist())
        if "station_label" in scoped.columns
        else []
    )
    selected_stations = (
        st.sidebar.multiselect(
            "Estações meteorológicas",
            options=available_stations,
            default=[],
            placeholder="Todas as estações",
        )
        if available_stations
        else []
    )

    st.sidebar.divider()
    st.sidebar.caption(
        f"Base INMET carregada: {len(df):,} linhas | "
        f"{len(available_years)} anos disponíveis neste filtro.".replace(",", ".")
    )

    return GlobalFilters(
        years=tuple(selected_years),
        regions=tuple(selected_regions),
        ufs=tuple(selected_ufs),
        stations=tuple(selected_stations),
        only_brazil_territory=only_brazil,
    )


def main() -> None:
    """Configura a página Streamlit, filtros globais e abas modulares."""
    st.set_page_config(
        page_title="Clima, Desmatamento e Eleições — Brasil",
        page_icon="🌎",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Painel Integrado: Clima, Desmatamento e Eleições no Brasil")
    st.caption(
        "Exploração interativa da régua meteorológica (INMET) preparada para integração "
        "com dados eleitorais (TSE) e de desmatamento."
    )

    try:
        inmet_df = get_inmet_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    filters = _render_sidebar_filters(inmet_df)

    tab_climate, tab_elections, tab_deforestation, tab_combined = st.tabs(
        [
            "🌦️ Clima (INMET)",
            "🗳️ Eleições (TSE)",
            "🌳 Desmatamento",
            "🔗 Análise Combinada",
        ]
    )

    with tab_climate:
        render_climate_tab(inmet_df, filters)

    with tab_elections:
        render_elections_tab(filters)

    with tab_deforestation:
        render_deforestation_tab(filters)

    with tab_combined:
        render_combined_tab(filters)


if __name__ == "__main__":
    main()
