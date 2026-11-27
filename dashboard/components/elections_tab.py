"""Aba de Eleições (TSE): prefeitos eleitos por alinhamento ideológico."""
 
from __future__ import annotations
 
import pandas as pd
import plotly.express as px
import streamlit as st
 
from dashboard.models import GlobalFilters
from src.paths import PROCESSED_DATA_DIR
 
TSE_PATH = PROCESSED_DATA_DIR / "tse" / "tse.parquet"
ORDER = ["far_left", "left", "center_left", "center",
         "center_right", "right", "far_right", "other"]
COLORS = {
    "far_left": "#B2182B",
    "left": "#D6604D",
    "center_left": "#F4A582",
    "center": "#D9D9D9",
    "center_right": "#92C5DE",
    "right": "#4393C3",
    "far_right": "#2166AC",
    "other": "#808080"
} 
 
@st.cache_data(show_spinner="Carregando dados do TSE...")
def load_tse() -> pd.DataFrame:
    if not TSE_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(TSE_PATH)
 
 
def render_elections_tab(filters: GlobalFilters) -> None:
    st.subheader("Eleições Municipais (TSE) — Prefeitos eleitos")
 
    df = load_tse()
    if df.empty:
        st.warning(
            f"Arquivo não encontrado: {TSE_PATH}. "
            "Rode `python -m src.preprocessing.tse`."
        )
        return
 
    # Filtro de ano próprio (anos eleitorais != anos do INMET)
    years = sorted(df["election_year"].unique().tolist())
    sel_years = st.multiselect("Ano da eleição", years, default=years)
    df = df[df["election_year"].isin(sel_years)]
 
    # Filtro global de UF (só funciona se o parquet tiver a coluna 'uf')
    if "uf" not in df.columns:
        st.info("O parquet não tem a coluna `uf`: rode o `tse.py` atualizado para filtrar por estado.")
    elif filters.ufs:
        df = df[df["uf"].isin(filters.ufs)]
 
    if df.empty:
        st.info("Nenhum registro para os filtros selecionados.")
        return
 
    c1, c2, c3 = st.columns(3)
    c1.metric("Prefeitos eleitos", f"{len(df):,}".replace(",", "."))
    c2.metric("Partido mais frequente", df["party_acronym"].mode().iat[0])
    c3.metric("Alinhamento mais frequente", df["alignment"].mode().iat[0])
 
    by_year = (df.groupby(["election_year", "alignment"]).size()
                 .reset_index(name="prefeitos"))
    fig = px.bar(
    by_year,
    x="election_year",
    y="prefeitos",
    color="alignment",
    color_discrete_map=COLORS,
    category_orders={"alignment": ORDER},
    barmode="stack",
    title="Prefeitos eleitos por alinhamento e ano"
)
        
    st.plotly_chart(fig, use_container_width=True)
 
    if "uf" in df.columns:
        by_uf = df.groupby(["uf", "alignment"]).size().reset_index(name="prefeitos")
        fig2 = px.bar(
    by_uf,
    x="uf",
    y="prefeitos",
    color="alignment",
    color_discrete_map=COLORS,
    category_orders={"alignment": ORDER},
    title="Distribuição por UF"
)
        st.plotly_chart(fig2, use_container_width=True)
 