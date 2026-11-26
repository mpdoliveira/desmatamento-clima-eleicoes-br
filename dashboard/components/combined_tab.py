"""Placeholder modular para a futura aba de Análise Combinada."""

from __future__ import annotations

import streamlit as st

from dashboard.models import GlobalFilters


def render_combined_tab(filters: GlobalFilters) -> None:
    """Renderiza placeholder estruturado para a aba de Análise Combinada."""
    st.subheader("Análise Combinada: Clima × Desmatamento × Eleições")
    st.info(
        "Esta aba integrará a 'régua' climática do INMET com os indicadores de "
        "desmatamento e comportamento eleitoral por UF, município e ano."
    )
    years_txt = ", ".join(map(str, filters.years)) if filters.years else "Todos"
    ufs_txt = ", ".join(filters.ufs) if filters.ufs else "Todas"
    st.markdown(
        f"""
        **Integrações planejadas (chaves `ANO`, `UF` e coordenadas):**
        - Correlações entre anomalias de temperatura/precipitação, desmatamento e resultados eleitorais
        - Painel comparativo multivariado por estado e ciclo eleitoral
        - Filtros globais ativos herdados da barra lateral: Anos (`{years_txt}`), UFs (`{ufs_txt}`)
        """
    )
