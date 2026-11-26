"""Placeholder modular para a futura aba de Desmatamento."""

from __future__ import annotations

import streamlit as st

from dashboard.models import GlobalFilters
from src.paths import PROCESSED_DATA_DIR


def render_deforestation_tab(filters: GlobalFilters) -> None:
    """Renderiza placeholder estruturado para a aba de Desmatamento."""
    st.subheader("Desmatamento (PRODES / DETER / MapBiomas)")
    st.info(
        "Esta aba receberá as séries e camadas geoespaciais de desmatamento processadas em "
        f"`{PROCESSED_DATA_DIR}`."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Visualizações planejadas:**
            - Área desmatada anual por bioma, UF e município
            - Mapas de calor e focos acumulados por janela temporal
            - Taxas de variação e tendências regionais
            """
        )
    with col2:
        years_txt = ", ".join(map(str, filters.years)) if filters.years else "Todos"
        ufs_txt = ", ".join(filters.ufs) if filters.ufs else "Todas"
        st.markdown(
            f"""
            **Filtros globais prontos para reutilização:**
            - Anos selecionados: `{years_txt}`
            - UFs selecionadas: `{ufs_txt}`
            """
        )
