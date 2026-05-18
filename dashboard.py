import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


HEDEF_HIZMET_DUZEYI = 0.80
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EXCEL = BASE_DIR / "tüketim_v3.xlsx"
DEFAULT_CACHE = BASE_DIR / "enm412_cache.pkl"


st.set_page_config(
    page_title="ENM412 Stok ve Talep Dashboard",
    page_icon="📦",
    layout="wide",
)


def first_existing(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def fmt_pct(value):
    if pd.isna(value):
        return "-"
    return f"%{value * 100:.1f}"


@st.cache_data(show_spinner=False)
def load_excel(path):
    xl = pd.ExcelFile(path)
    sheets = {}
    for sheet in xl.sheet_names:
        sheets[sheet] = pd.read_excel(xl, sheet_name=sheet)
    return sheets


@st.cache_data(show_spinner=False)
def load_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


st.title("ENM412 Stok ve Talep Dashboard")
st.caption("Talep tahmini, segmentasyon, model metrikleri ve %80 hizmet düzeyi takibi")

with st.sidebar:
    st.header("Veri Kaynağı")
    excel_path = st.text_input("Excel dosyası", value=str(DEFAULT_EXCEL))
    cache_path = st.text_input("Pipeline cache", value=str(DEFAULT_CACHE))
    st.divider()
    st.markdown("**Hedef hizmet düzeyi**")
    st.progress(HEDEF_HIZMET_DUZEYI)
    st.write(fmt_pct(HEDEF_HIZMET_DUZEYI))


excel_file = Path(excel_path)
cache_file = Path(cache_path)

if not excel_file.exists():
    st.error(f"Excel dosyası bulunamadı: {excel_file}")
    st.stop()

try:
    sheets = load_excel(excel_file)
except Exception as exc:
    st.error(f"Excel okunamadı: {exc}")
    st.stop()

ml_df = sheets.get("ML_Hazir_Veri", pd.DataFrame())
abc_df = sheets.get("ABC_XYZ_Segmentasyon", pd.DataFrame())
opt_df = sheets.get("Optimizasyon_Parametreleri", pd.DataFrame())

cache = {}
if cache_file.exists():
    try:
        cache = load_cache(cache_file)
    except Exception as exc:
        st.warning(f"Cache okunamadı, sadece Excel verisi gösteriliyor: {exc}")

batch_df = cache.get("batch_df", pd.DataFrame()) if isinstance(cache, dict) else pd.DataFrame()
analiz_df = cache.get("analiz_df", pd.DataFrame()) if isinstance(cache, dict) else pd.DataFrame()

parca_col = first_existing(ml_df, ["Parça_Kodu", "ParÃ§a_Kodu", "Parca_Kodu"])
talep_col = first_existing(ml_df, ["Talep_ham", "Talep", "Talep_Ham"])
split_col = first_existing(ml_df, ["Split"])
tarih_col = first_existing(ml_df, ["Tarih"])
segment_col = first_existing(ml_df, ["talep_segment", "Segment"])

toplam_parca = ml_df[parca_col].nunique() if parca_col else 0
toplam_satir = len(ml_df)
toplam_talep = ml_df[talep_col].sum() if talep_col else np.nan
ortalama_talep = ml_df[talep_col].mean() if talep_col else np.nan

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Toplam Parça", f"{toplam_parca:,.0f}")
with col2:
    metric_card("Veri Satırı", f"{toplam_satir:,.0f}")
with col3:
    metric_card("Toplam Talep", f"{toplam_talep:,.0f}" if not pd.isna(toplam_talep) else "-")
with col4:
    metric_card("Hedef Hizmet", fmt_pct(HEDEF_HIZMET_DUZEYI))

tab_veri, tab_model, tab_hizmet, tab_parca = st.tabs(
    ["Veri Özeti", "Model Sonuçları", "Hizmet Düzeyi", "Parça Detayı"]
)

with tab_veri:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Segment Dağılımı")
        if segment_col:
            seg = ml_df[[parca_col, segment_col]].drop_duplicates()
            seg_counts = seg[segment_col].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Parça Sayısı"]
            fig = px.bar(seg_counts, x="Segment", y="Parça Sayısı", text="Parça Sayısı")
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Segment kolonu bulunamadı.")

    with right:
        st.subheader("Train / Test Dağılımı")
        if split_col:
            split_counts = ml_df[split_col].value_counts().reset_index()
            split_counts.columns = ["Split", "Satır Sayısı"]
            fig = px.pie(split_counts, names="Split", values="Satır Sayısı", hole=0.45)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Split kolonu bulunamadı.")

    st.subheader("Talep Zaman Serisi")
    if tarih_col and talep_col:
        ts = ml_df.groupby(tarih_col, as_index=False)[talep_col].sum()
        fig = px.line(ts, x=tarih_col, y=talep_col, markers=True)
        fig.update_layout(height=360, xaxis_title="Tarih", yaxis_title="Toplam Talep")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tarih veya Talep kolonu bulunamadı.")

    with st.expander("Excel sayfaları"):
        st.write(list(sheets.keys()))

with tab_model:
    if batch_df.empty:
        st.info("Model sonuçları için önce pipeline çalıştırılıp enm412_cache.pkl oluşturulmalı.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card("Tahminlenen Parça", f"{len(batch_df):,.0f}")
        with m2:
            metric_card("Medyan WAPE", f"%{batch_df['WAPE'].median():.1f}" if "WAPE" in batch_df else "-")
        with m3:
            metric_card("Medyan MAE", f"{batch_df['MAE'].median():.1f}" if "MAE" in batch_df else "-")
        with m4:
            ml_oran = (batch_df.get("Sampiyon_Tip", pd.Series(dtype=str)).eq("ML").mean())
            metric_card("ML Şampiyon Oranı", fmt_pct(ml_oran))

        left, right = st.columns([1, 1])
        with left:
            if "Sampiyon" in batch_df:
                samp = batch_df["Sampiyon"].value_counts().reset_index()
                samp.columns = ["Model", "Parça Sayısı"]
                fig = px.bar(samp, x="Model", y="Parça Sayısı", text="Parça Sayısı")
                fig.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        with right:
            metric_cols = [c for c in ["MAE", "RMSE", "WAPE", "sMAPE"] if c in batch_df]
            if metric_cols:
                fig = px.box(batch_df[metric_cols], points=False)
                fig.update_layout(height=380, yaxis_title="Metrik Değeri")
                st.plotly_chart(fig, use_container_width=True)

        st.dataframe(batch_df, use_container_width=True, height=360)

with tab_hizmet:
    st.subheader("Hizmet Düzeyi Takibi")
    st.write("Hedef: en az ortalama %80 hizmet düzeyi.")

    hz_source = None
    if "sim_HZ" in batch_df:
        hz_source = batch_df
    elif "sim_HZ" in opt_df:
        hz_source = opt_df

    if hz_source is None:
        st.info("Toplu optimizasyon sonucu içinde sim_HZ kolonu bulunursa burada gerçekleşen hizmet düzeyi gösterilir.")
        st.metric("Hedef Hizmet Düzeyi", fmt_pct(HEDEF_HIZMET_DUZEYI))
    else:
        hz = pd.to_numeric(hz_source["sim_HZ"], errors="coerce")
        ort_hz = hz.mean()
        alt_adet = int((hz < HEDEF_HIZMET_DUZEYI).sum())
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Ortalama Hizmet", fmt_pct(ort_hz))
        with c2:
            metric_card("Hedef Altı Parça", f"{alt_adet:,}")
        with c3:
            metric_card("Hedef", fmt_pct(HEDEF_HIZMET_DUZEYI))

        plot_df = hz_source.copy()
        plot_df["hedef_HZ"] = HEDEF_HIZMET_DUZEYI
        fig = px.histogram(plot_df, x="sim_HZ", nbins=30)
        fig.add_vline(x=HEDEF_HIZMET_DUZEYI, line_dash="dash", line_color="red")
        fig.update_layout(height=380, xaxis_tickformat=".0%", xaxis_title="Hizmet Düzeyi")
        st.plotly_chart(fig, use_container_width=True)

with tab_parca:
    if not parca_col:
        st.info("Parça kolonu bulunamadı.")
    else:
        parcalar = sorted(ml_df[parca_col].dropna().astype(str).unique().tolist())
        secili = st.selectbox("Parça seç", parcalar)
        pdf = ml_df[ml_df[parca_col].astype(str) == secili].copy()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Toplam Talep", f"{pdf[talep_col].sum():,.0f}" if talep_col else "-")
        with c2:
            metric_card("Ortalama Talep", f"{pdf[talep_col].mean():,.1f}" if talep_col else "-")
        with c3:
            if segment_col:
                metric_card("Segment", str(pdf[segment_col].iloc[0]))
        with c4:
            if split_col:
                metric_card("Satır", f"{len(pdf):,.0f}")

        if tarih_col and talep_col:
            fig = px.line(pdf.sort_values(tarih_col), x=tarih_col, y=talep_col, color=split_col if split_col else None, markers=True)
            fig.update_layout(height=390, xaxis_title="Tarih", yaxis_title="Talep")
            st.plotly_chart(fig, use_container_width=True)

        if not batch_df.empty:
            b_parca = first_existing(batch_df, ["Parça_Kodu", "ParÃ§a_Kodu", "Parca_Kodu"])
            if b_parca:
                brow = batch_df[batch_df[b_parca].astype(str) == secili]
                if not brow.empty:
                    st.subheader("Tahmin Sonucu")
                    st.dataframe(brow, use_container_width=True)

        st.subheader("Ham Kayıtlar")
        st.dataframe(pdf, use_container_width=True, height=320)
