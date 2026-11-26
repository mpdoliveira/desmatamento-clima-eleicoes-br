"""Componente visual da aba de análise meteorológica (INMET)."""

from __future__ import annotations

import json
from typing import Any

import folium
from folium.plugins import FastMarkerCluster
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from dashboard.components.climate_analytics import (
    aggregate_stations_by_year,
    aggregate_stations_spatial,
    compute_annual_trend,
    compute_climate_kpis,
    compute_correlation_matrix,
    compute_monthly_seasonality,
    compute_uf_summary,
    filter_climate_dataframe,
)
from dashboard.models import (
    AggregationMode,
    CLIMATE_METRICS,
    CLIMATE_METRIC_MAP,
    GlobalFilters,
    MONTH_LABELS_PT,
    MetricSpec,
)

_BRAZIL_CENTER = {"lat": -14.235, "lon": -51.9253}

_YEAR_PALETTE: tuple[str, ...] = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#17becf",
    "#e377c2",
    "#8c564b",
    "#bcbd22",
    "#0f766e",
)


def _fmt_number(value: Any, decimals: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    formatted = f"{float(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{formatted}{suffix}"


def _build_cluster_callback(color: str) -> str:
    """Retorna callback JS leve para renderizar marcadores coloridos com tooltip e popup."""
    safe_color = json.dumps(color)
    return f"""
    function (row) {{
        var esc = function(v) {{
            if (v === null || v === undefined) return 'N/D';
            return String(v)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;');
        }};
        var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {{
            radius: 5,
            color: {safe_color},
            weight: 1.5,
            fill: true,
            fillColor: {safe_color},
            fillOpacity: 0.75
        }});
        var tooltip = esc(row[2]) + ' (' + esc(row[3]) + ') | Ano ' + esc(row[6]) +
            ' | T: ' + esc(row[8]) + ' | P: ' + esc(row[10]);
        var popup = "<div style='font-family:sans-serif;font-size:12px;line-height:1.45;'>" +
            "<strong style='font-size:13px;'>" + esc(row[2]) + "</strong><br>" +
            "<b>Ano:</b> " + esc(row[6]) + " &nbsp;|&nbsp; <b>UF/Região:</b> " + esc(row[3]) + " / " + esc(row[4]) + "<br>" +
            "<b>WMO:</b> " + esc(row[5]) + " &nbsp;|&nbsp; <b>Meses com registro:</b> " + esc(row[7]) + "<br>" +
            "<b>Coordenadas:</b> " + esc(row[0].toFixed(4)) + ", " + esc(row[1].toFixed(4)) + "<br>" +
            "<hr style='margin:6px 0;border:none;border-top:1px solid #ddd;'>" +
            "<b>Temp. média:</b> " + esc(row[8]) + "<br>" +
            "<b>Umidade média:</b> " + esc(row[9]) + "<br>" +
            "<b>Precipitação (acum. no ano):</b> " + esc(row[10]) + " (soma mensal: " + esc(row[11]) + ")<br>" +
            "<b>Radiação (soma mensal):</b> " + esc(row[12]) + "<br>" +
            "<span style='color:#666;font-size:11px;'><b>Arquivo:</b> " + esc(row[13]) + "</span>" +
            "</div>";
        marker.bindTooltip(tooltip);
        marker.bindPopup(popup, {{maxWidth: 340}});
        return marker;
    }}
    """


def _prepare_cluster_rows(year_df: pd.DataFrame, year: int) -> list[list[Any]]:
    """Serializa apenas os campos essenciais da estação para o FastMarkerCluster."""
    rows: list[list[Any]] = []
    for _, row in year_df.iterrows():
        est_name = str(row.get("ESTACAO") or row.get("station_label") or "Estação INMET")
        uf = str(row.get("UF") or "N/D")
        regiao = str(row.get("REGIAO") or "N/D")
        wmo = str(row.get("WMO") or "N/D")
        months = int(row["months_count"]) if pd.notna(row.get("months_count")) else 0
        source_file = str(row.get("source_file") or "N/D")
        rows.append(
            [
                round(float(row["LATITUDE"]), 5),
                round(float(row["LONGITUDE"]), 5),
                est_name,
                uf,
                regiao,
                wmo,
                year,
                months,
                _fmt_number(row.get("temp_c"), 1, " °C"),
                _fmt_number(row.get("humidity_pct"), 1, " %"),
                _fmt_number(row.get("precip_mm_sum"), 1, " mm"),
                _fmt_number(row.get("precip_mm_mean"), 1, " mm/mês"),
                _fmt_number(row.get("radiation_kj_m2_mean"), 0, " kJ/m²/mês"),
                source_file,
            ]
        )
    return rows


def _render_kpis(filtered_df: pd.DataFrame) -> None:
    kpis = compute_climate_kpis(filtered_df)
    cols = st.columns(6)
    cols[0].metric("Estações", f"{kpis['stations_count']:,}".replace(",", "."))
    cols[1].metric("Registros (Estação-Mês)", f"{kpis['records_count']:,}".replace(",", "."))
    cols[2].metric("Temp. Média", _fmt_number(kpis["temp_mean"], 1, " °C"))
    cols[3].metric("Umidade Média", _fmt_number(kpis["humidity_mean"], 1, " %"))
    cols[4].metric(
        "Precip. (Soma Mensal)",
        _fmt_number(kpis["precip_monthly_mean"], 1, " mm"),
        help="Média das somas mensais de precipitação registradas por estação no recorte ativo.",
    )
    cols[5].metric(
        "Radiação (Soma Mensal)",
        _fmt_number(kpis["radiation_monthly_mean"], 0, " kJ/m²"),
        help="Média das somas mensais de radiação global registradas por estação no recorte ativo.",
    )


def _render_yearly_folium_map(filtered_df: pd.DataFrame) -> None:
    st.subheader("Mapa de Estações por Ano (Camadas Alternáveis)")
    st.caption(
        "Cada ano selecionado forma uma camada independente no controle no canto superior direito do mapa, "
        "permitindo ativar, desativar e sobrepor a rede de estações ao longo do tempo."
    )

    yearly_stations = aggregate_stations_by_year(filtered_df)
    if yearly_stations.empty:
        st.info("Não há estações com coordenadas válidas para montar o mapa por ano.")
        return

    years = sorted(yearly_stations["ANO"].dropna().astype(int).unique().tolist())

    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        default_active_mode = st.radio(
            "Camadas ativas inicialmente no mapa",
            options=["Apenas o ano mais recente", "Todos os anos filtrados"],
            index=0 if len(years) > 3 else 1,
            horizontal=True,
            key="folium_initial_layers_mode",
        )
    with ctrl_col2:
        st.metric("Camadas anuais disponíveis", len(years))

    map_center = [
        float(yearly_stations["LATITUDE"].median()),
        float(yearly_stations["LONGITUDE"].median()),
    ]
    if not (-35.0 <= map_center[0] <= 6.0 and -75.0 <= map_center[1] <= -28.0):
        map_center = [_BRAZIL_CENTER["lat"], _BRAZIL_CENTER["lon"]]

    fmap = folium.Map(
        location=map_center,
        zoom_start=4,
        tiles="OpenStreetMap",
        prefer_canvas=True,
        control_scale=True,
    )

    latest_year = years[-1]
    show_all = default_active_mode == "Todos os anos filtrados"

    cluster_options = {
        "chunkedLoading": True,
        "maxClusterRadius": 55,
        "disableClusteringAtZoom": 8,
        "spiderfyOnMaxZoom": True,
        "showCoverageOnHover": False,
    }

    for idx, year in enumerate(years):
        year_df = yearly_stations[yearly_stations["ANO"].astype(int) == year]
        color = _YEAR_PALETTE[idx % len(_YEAR_PALETTE)]
        FastMarkerCluster(
            data=_prepare_cluster_rows(year_df, year),
            callback=_build_cluster_callback(color),
            name=f"Ano {year} ({len(year_df)} estações)",
            show=show_all or (year == latest_year),
            options=cluster_options,
        ).add_to(fmap)

    folium.LayerControl(collapsed=len(years) > 10).add_to(fmap)
    st_folium(
        fmap,
        height=540,
        use_container_width=True,
        returned_objects=[],
        key="inmet_yearly_folium_map",
    )


def _create_plotly_spatial_figure(
    spatial_df: pd.DataFrame,
    spec: MetricSpec,
    agg_mode: AggregationMode,
    clip_outliers: bool,
) -> go.Figure | None:
    if spatial_df.empty or spec.column not in spatial_df.columns:
        return None

    plot_df = spatial_df.dropna(subset=[spec.column, "LATITUDE", "LONGITUDE"]).copy()
    if plot_df.empty:
        return None

    agg_label = (
        "Soma acumulada no período"
        if (spec.supports_sum and agg_mode == "sum")
        else ("Média das somas mensais" if spec.supports_sum else "Média no período")
    )
    unit_label = spec.unit if agg_mode == "sum" or not spec.supports_sum else f"{spec.unit}/mês"
    title = f"{spec.short_label} — {agg_label} ({unit_label})"

    values = plot_df[spec.column].astype(float)
    range_color: list[float] | None = None
    if clip_outliers and len(values) >= 10:
        q_low, q_high = float(values.quantile(0.02)), float(values.quantile(0.98))
        if np.isfinite(q_low) and np.isfinite(q_high) and q_high > q_low:
            range_color = [q_low, q_high]

    hover_data: dict[str, Any] = {
        "LATITUDE": ":.4f",
        "LONGITUDE": ":.4f",
        spec.column: ":.2f",
        "months_count": True,
        "years_count": True,
    }
    if "UF" in plot_df.columns:
        hover_data["UF"] = True
    if spec.column == "precip_mm":
        hover_data["precip_mm_mean"] = ":.1f"
        hover_data["precip_mm_sum"] = ":.1f"
    elif spec.column == "radiation_kj_m2":
        hover_data["radiation_kj_m2_mean"] = ":.0f"
        hover_data["radiation_kj_m2_sum"] = ":.0f"

    labels = {
        spec.column: f"{spec.short_label} ({unit_label})",
        "months_count": "Meses",
        "years_count": "Anos",
        "precip_mm_mean": "Precip. soma mensal média (mm/mês)",
        "precip_mm_sum": "Precip. acumulada no período (mm)",
        "radiation_kj_m2_mean": "Radiação soma mensal média (kJ/m²/mês)",
        "radiation_kj_m2_sum": "Radiação acumulada no período (kJ/m²)",
    }

    scatter_fn = getattr(px, "scatter_map", None) or px.scatter_mapbox
    style_kwarg = "map_style" if hasattr(px, "scatter_map") else "mapbox_style"

    kwargs: dict[str, Any] = {
        "data_frame": plot_df,
        "lat": "LATITUDE",
        "lon": "LONGITUDE",
        "color": spec.column,
        "hover_name": "station_label",
        "hover_data": hover_data,
        "color_continuous_scale": spec.colorscale,
        "range_color": range_color,
        "labels": labels,
        "zoom": 3.1,
        "center": _BRAZIL_CENTER,
        "height": 430,
        "title": title,
        style_kwarg: "carto-positron",
    }

    fig = scatter_fn(**kwargs)
    fig.update_traces(marker={"size": 8, "opacity": 0.85})
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        coloraxis_colorbar={"title": unit_label, "thickness": 14},
    )
    return fig


def _render_spatial_maps(filtered_df: pd.DataFrame) -> None:
    st.subheader("Distribuição Espacial das 4 Variáveis Climáticas")
    st.caption(
        "Temperatura e Umidade representam médias registradas no período. "
        "Precipitação e Radiação Global correspondem à soma mensal por estação — "
        "escolha entre a média das somas mensais (comparável entre estações com históricos distintos) "
        "ou a soma acumulada no período."
    )

    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    with c1:
        precip_mode_label = st.radio(
            "Agregação de Precipitação (Soma Mensal)",
            options=["Soma mensal (mm/mês)", "Acumulado no período (mm)"],
            index=0,
            horizontal=True,
            key="spatial_precip_agg",
        )
    with c2:
        rad_mode_label = st.radio(
            "Agregação de Radiação Global (Soma Mensal)",
            options=["Soma mensal (kJ/m²/mês)", "Acumulado no período (kJ/m²)"],
            index=0,
            horizontal=True,
            key="spatial_radiation_agg",
        )
    with c3:
        clip_outliers = st.checkbox(
            "Suavizar extremos na escala de cor (p2–p98)",
            value=True,
            help="Evita que leituras extremas de sensores comprimam o gradiente de cores no mapa; o valor real permanece no hover.",
            key="spatial_clip_outliers",
        )

    precip_agg: AggregationMode = "sum" if precip_mode_label.startswith("Acumulado") else "mean"
    rad_agg: AggregationMode = "sum" if rad_mode_label.startswith("Acumulado") else "mean"

    spatial_df = aggregate_stations_spatial(
        filtered_df,
        precip_agg=precip_agg,
        radiation_agg=rad_agg,
    )
    if spatial_df.empty:
        st.info("Sem estações válidas para plotar os mapas espaciais.")
        return

    agg_by_col: dict[str, AggregationMode] = {
        "temp_c": "mean",
        "humidity_pct": "mean",
        "precip_mm": precip_agg,
        "radiation_kj_m2": rad_agg,
    }

    for row_start in (0, 2):
        cols = st.columns(2)
        for offset in (0, 1):
            spec = CLIMATE_METRICS[row_start + offset]
            with cols[offset]:
                fig = _create_plotly_spatial_figure(
                    spatial_df=spatial_df,
                    spec=spec,
                    agg_mode=agg_by_col[spec.column],
                    clip_outliers=clip_outliers,
                )
                if fig is None:
                    st.warning(f"Sem dados válidos para {spec.short_label} no filtro atual.")
                else:
                    st.plotly_chart(fig, width="stretch")


def _render_seasonality_and_trends(filtered_df: pd.DataFrame) -> None:
    st.subheader("Perfil Sazonal Mensal e Série Histórica")

    metric_options = {spec.column: spec.label for spec in CLIMATE_METRICS}
    selected_col = st.selectbox(
        "Variável para análise temporal e regional",
        options=list(metric_options.keys()),
        format_func=lambda c: metric_options[c],
        key="climate_extra_metric",
    )
    spec = CLIMATE_METRIC_MAP[selected_col]

    by_ym, climatology = compute_monthly_seasonality(filtered_df, selected_col)
    if by_ym.empty:
        st.info(f"Não há dados válidos de {spec.short_label} para o perfil sazonal.")
    else:
        by_ym = by_ym.copy()
        by_ym["MES_NOME"] = by_ym["MES"].astype(int).map(MONTH_LABELS_PT)
        climatology = climatology.copy()
        climatology["MES_NOME"] = climatology["MES"].astype(int).map(MONTH_LABELS_PT)

        fig_season = px.line(
            by_ym,
            x="MES",
            y="value",
            color="ANO_STR",
            markers=True,
            labels={
                "MES": "Mês",
                "value": f"{spec.short_label} ({spec.unit})",
                "ANO_STR": "Ano",
            },
            title=f"Perfil Sazonal Mensal por Ano — {spec.label}",
        )
        fig_season.add_trace(
            go.Scatter(
                x=climatology["MES"],
                y=climatology["mean_value"],
                mode="lines+markers",
                name="Média do recorte",
                line={"color": "#111827", "width": 3, "dash": "dash"},
                marker={"size": 7},
            )
        )
        fig_season.update_xaxes(
            tickmode="array",
            tickvals=list(MONTH_LABELS_PT.keys()),
            ticktext=list(MONTH_LABELS_PT.values()),
        )
        fig_season.update_layout(height=420, legend_title_text="Ano")
        st.plotly_chart(fig_season, width="stretch")

    col_left, col_right = st.columns(2)
    with col_left:
        trend_df = compute_annual_trend(filtered_df)
        if trend_df.empty or trend_df[selected_col].dropna().empty:
            st.info("Sem série anual suficiente para exibir evolução temporal.")
        else:
            unit_suffix = f"{spec.unit}/mês" if spec.supports_sum else spec.unit
            y_axis_label = (
                f"Soma mensal média ({unit_suffix})"
                if spec.supports_sum
                else f"Média entre estações ({unit_suffix})"
            )
            fig_trend = px.line(
                trend_df.dropna(subset=[selected_col]),
                x="ANO",
                y=selected_col,
                markers=True,
                hover_data={"stations_count": True},
                labels={
                    "ANO": "Ano",
                    selected_col: y_axis_label,
                    "stations_count": "Estações ativas",
                },
                title=f"Evolução Anual — {spec.short_label} (Régua Temporal)",
            )
            fig_trend.update_layout(height=380)
            st.plotly_chart(fig_trend, width="stretch")

    with col_right:
        uf_df = compute_uf_summary(filtered_df, selected_col)
        if uf_df.empty:
            st.info("Coluna de UF indisponível ou sem registros válidos no recorte atual.")
        else:
            uf_value_label = (
                f"Soma mensal média ({spec.unit})"
                if spec.supports_sum
                else f"Média ({spec.unit})"
            )
            fig_uf = px.bar(
                uf_df,
                x="UF",
                y="mean_value",
                hover_data={"stations_count": True, "records_count": True},
                color="mean_value",
                color_continuous_scale=spec.colorscale,
                labels={
                    "UF": "UF",
                    "mean_value": uf_value_label,
                    "stations_count": "Estações",
                    "records_count": "Registros",
                },
                title=f"Comparativo por UF — {spec.short_label}",
            )
            fig_uf.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig_uf, width="stretch")


def _render_correlation_matrix(filtered_df: pd.DataFrame) -> None:
    st.subheader("Matriz de Correlação entre Variáveis Climáticas")
    corr, pair_counts = compute_correlation_matrix(filtered_df)
    if corr.empty:
        st.info(
            "São necessários pelo menos dois indicadores climáticos com valores válidos suficientes para calcular a matriz de correlação."
        )
        return

    labels = [
        CLIMATE_METRIC_MAP[col].short_label if col in CLIMATE_METRIC_MAP else col
        for col in corr.columns
    ]

    text_matrix = [
        [
            "N/D"
            if pd.isna(corr.iloc[i, j])
            else f"{corr.iloc[i, j]:.2f}<br><span style='font-size:10px'>n={int(pair_counts.iloc[i, j]):,}</span>"
            for j in range(len(corr.columns))
        ]
        for i in range(len(corr.index))
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
            text=text_matrix,
            texttemplate="%{text}",
            hovertemplate="%{y} × %{x}<br>Correlação de Pearson: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Correlação de Pearson (observações mensais por estação)",
        height=420,
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(fig, width="stretch")


def _render_station_table(filtered_df: pd.DataFrame) -> None:
    with st.expander("Tabela Resumo por Estação e Exportação CSV", expanded=False):
        summary = aggregate_stations_spatial(filtered_df, precip_agg="mean", radiation_agg="mean")
        if summary.empty:
            st.info("Nenhuma estação disponível para exibição tabular.")
            return

        display_cols = [
            col
            for col in (
                "station_label",
                "UF",
                "REGIAO",
                "WMO",
                "LATITUDE",
                "LONGITUDE",
                "years_count",
                "months_count",
                "temp_c",
                "humidity_pct",
                "precip_mm_mean",
                "precip_mm_sum",
                "radiation_kj_m2_mean",
            )
            if col in summary.columns
        ]
        table_df = summary[display_cols].sort_values(
            "months_count", ascending=False
        )
        st.dataframe(table_df, width="stretch", hide_index=True)

        csv_bytes = table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar resumo das estações (CSV)",
            data=csv_bytes,
            file_name="inmet_resumo_estacoes.csv",
            mime="text/csv",
        )


def render_climate_tab(df: pd.DataFrame, filters: GlobalFilters) -> None:
    """Renderiza toda a aba de análise meteorológica respeitando os filtros globais."""
    if df.empty:
        st.warning("O conjunto de dados meteorológicos está vazio.")
        return

    filtered_df = filter_climate_dataframe(df, filters)
    if filtered_df.empty:
        st.warning(
            "Nenhum registro meteorológico corresponde aos filtros selecionados. "
            "Ajuste os filtros na barra lateral."
        )
        return

    _render_kpis(filtered_df)
    st.divider()
    _render_yearly_folium_map(filtered_df)
    st.divider()
    _render_spatial_maps(filtered_df)
    st.divider()
    _render_seasonality_and_trends(filtered_df)
    st.divider()
    _render_correlation_matrix(filtered_df)
    _render_station_table(filtered_df)
