"""Placeholder modular para a futura aba de dados eleitorais (TSE)."""

from __future__ import annotations

import streamlit as st

from dashboard.models import GlobalFilters
from src.paths import PROCESSED_DATA_DIR


def render_elections_tab(filters: GlobalFilters) -> None:
    """Renderiza placeholder estruturado para a aba de Eleições (TSE)."""
    st.subheader("Eleições Brasileiras (TSE)")
    st.info(
        "Esta aba receberá os indicadores eleitorais processados a partir de "
        f"`{PROCESSED_DATA_DIR / 'tse'}` (`src/preprocessing/tse.py`)."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Visualizações planejadas:**
            - Mapa de resultados e comparecimento por UF / município
            - Evolução histórica por ciclo eleitoral (`ANO`)
            - Distribuição regional e perfil de votos válidos
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
