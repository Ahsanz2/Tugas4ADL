"""Dashboard perbandingan model peramalan beban listrik Panama (24 jam ke depan).
Jalankan:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Beban Listrik Panama", layout="wide", initial_sidebar_state="expanded")
DATA = Path(__file__).parent / "data"

# ---------- warna tetap per keluarga metode (sama di semua grafik) ----------
WARNA = {"Model fondasi (zero-shot)": "#2a78d6", "Model fondasi (fine-tuned)": "#6250d6", "Deep learning": "#eb6834",
         "Machine learning": "#1baf7a", "Statistik": "#eda100", "Patokan": "#898781"}
SERI = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e87ba4"]
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def f(x, d=2):
    """Angka dengan koma desimal gaya Indonesia."""
    return "–" if x is None or pd.isna(x) else f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def tata(fig, tinggi=380, judul_y=None, judul_x=None, legenda=True):
    fig.update_layout(height=tinggi, margin=dict(l=10, r=10, t=30, b=10), separators=",.", hovermode="x unified" if legenda == "x" else "closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), showlegend=bool(legenda),
                      yaxis_title=judul_y, xaxis_title=judul_x, font=dict(size=13))
    return fig


# ---------- pemuatan data ----------
@st.cache_data
def muat():
    model = pd.read_csv(DATA / "model.csv")
    z = np.load(DATA / "ramalan.npz", allow_pickle=False)
    ram = {k[3:]: z[k] for k in z.files if k.startswith("m::")}
    lo = {k[4:]: z[k] for k in z.files if k.startswith("lo::")}
    hi = {k[4:]: z[k] for k in z.files if k.startswith("hi::")}
    tgl = pd.to_datetime(z["tanggal"])
    beban = pd.read_parquet(DATA / "beban.parquet")
    studi = pd.read_csv(DATA / "studi.csv").fillna("")
    kont = pd.read_csv(DATA / "kontaminasi.csv")
    info = json.load(open(DATA / "info.json"))
    return model, ram, lo, hi, tgl, z["aktual"], int(z["n_normal"]), beban, studi, kont, info


MODEL, RAM, LO, HI, TGL, AKT, N_NORMAL, BEBAN, STUDI, KONT, INFO = muat()
KEL = dict(zip(MODEL.model, MODEL.keluarga))
PERIODE = {"Seluruh data uji (Jul 2019 – Jun 2020)": slice(0, len(TGL)), "Uji periode normal (Jul 2019 – Feb 2020)": slice(0, N_NORMAL),
           "Uji periode COVID (Mar – Jun 2020)": slice(N_NORMAL, len(TGL))}


@st.cache_data
def galat(awal, akhir):
    """Ukuran galat tiap model pada rentang hari uji [awal, akhir)."""
    a = AKT[awal:akhir]; baris = []
    for n, p in RAM.items():
        e = p[awal:akhir] - a
        baris.append({"Model": n, "Keluarga": KEL[n], "MAPE (%)": float(np.mean(np.abs(e) / a) * 100), "MAE (MWh)": float(np.abs(e).mean()),
                      "RMSE (MWh)": float(np.sqrt((e ** 2).mean())), "Bias (MWh)": float(e.mean()),
                      "Galat puncak (%)": float(np.mean(np.abs(p[awal:akhir].max(1) - a.max(1)) / a.max(1)) * 100)})
    return pd.DataFrame(baris)


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### Beban Listrik Panama")
    periode = st.selectbox("Periode penilaian", list(PERIODE))
    keluarga = st.multiselect("Keluarga metode", list(WARNA), default=list(WARNA))
    ukuran = st.selectbox("Ukuran galat", ["MAPE (%)", "MAE (MWh)", "RMSE (MWh)"])
    st.caption("Data: 3 Jan 2015 – 27 Jun 2020, per jam. Tiap hari diramal pukul 00.00 untuk 24 jam berikutnya. "
               "Semua ukuran galat: makin kecil makin bagus.")

sl = PERIODE[periode]
G = galat(sl.start, sl.stop)
GF = G[G.Keluarga.isin(keluarga)].sort_values(ukuran).reset_index(drop=True)
G_SEMUA, G_NORMAL, G_COVID = galat(0, len(TGL)), galat(0, N_NORMAL), galat(N_NORMAL, len(TGL))

st.title("Perbandingan Model Peramalan Beban")
st.caption(f"{len(MODEL)} model dari 4 keluarga metode · 362 hari uji · target: nat_demand (beban nasional per jam)")
t_ring, t_eda, t_ram, t_input, t_dek, t_lain, t_coba = st.tabs(
    ["Ringkasan", "Eksplorasi data", "Ramalan vs aktual", "Studi input", "Dekomposisi", "Studi lain", "Coba ramal sendiri"])

# =====================================================================================
with t_ring:
    if GF.empty:
        st.warning("Pilih minimal satu keluarga metode di sidebar.")
    else:
        b = GF.iloc[0]; bn = G_NORMAL.sort_values("MAPE (%)").iloc[0]; bc = G_COVID.sort_values("MAPE (%)").iloc[0]
        naif = G[G.Model == "Naif musiman"].iloc[0]
        d = 1 if "MWh" in ukuran else 2; sat = "" if "MWh" in ukuran else "%"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{ukuran.split(' ')[0]} terkecil, periode terpilih", f(b[ukuran], d) + sat, b.Model, delta_color="off", delta_arrow="off")
        c2.metric("MAPE terkecil saat normal", f(bn["MAPE (%)"]) + "%", bn.Model, delta_color="off", delta_arrow="off")
        c3.metric("MAPE terkecil saat COVID", f(bc["MAPE (%)"]) + "%", bc.Model, delta_color="off", delta_arrow="off")
        c4.metric(f"Patokan naif, periode terpilih", f(naif[ukuran], d) + sat, "jam yang sama minggu lalu", delta_color="off", delta_arrow="off")
        hemat = (1 - b["MAPE (%)"] / naif["MAPE (%)"]) * 100
        st.markdown(f"Pada periode terpilih, **{b.Model}** ({b.Keluarga.lower()}) punya galat terkecil. "
                    f"MAPE-nya {f(hemat, 0)}% lebih rendah dari patokan naif musiman.")

        k1, k2 = st.columns([1.05, 1])
        with k1:
            st.subheader(f"{ukuran} per model")
            fig = go.Figure()
            for kel in [k for k in WARNA if k in set(GF.Keluarga)]:
                s = GF[GF.Keluarga == kel]
                fig.add_bar(y=s.Model, x=s[ukuran], orientation="h", name=kel, marker_color=WARNA[kel], text=[f(v, d) for v in s[ukuran]],
                            textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>" + ukuran + ": %{x:.2f}<extra>" + kel + "</extra>")
            fig.update_yaxes(categoryorder="array", categoryarray=list(GF.Model[::-1]))
            st.plotly_chart(tata(fig, max(360, 24 * len(GF) + 90), judul_x=ukuran), width="stretch")
        with k2:
            st.subheader("Tabel hasil")
            tab = GF.merge(MODEL[["model", "mase", "dm_p"]].rename(columns={"model": "Model", "mase": "MASE (seluruh uji)", "dm_p": "p uji DM"}), on="Model")
            tab.insert(0, "#", range(1, len(tab) + 1))
            st.dataframe(tab, hide_index=True, width="stretch", height=max(360, 24 * len(GF) + 90),
                         column_config={c: st.column_config.NumberColumn(format="%.2f") for c in tab.columns if tab[c].dtype.kind == "f"} |
                                       {"p uji DM": st.column_config.NumberColumn(format="%.3f", help="Uji Diebold-Mariano terhadap model terbaik di seluruh data uji. Di atas 0,05: selisihnya belum terbukti nyata.")})
            st.download_button("Unduh tabel (CSV)", tab.to_csv(index=False).encode("utf-8"), "perbandingan_model.csv", "text/csv")

        st.subheader("Periode normal lawan periode COVID")
        st.caption("Titik penuh: MAPE periode normal. Lingkaran kosong: MAPE periode COVID. Garis panjang berarti model memburuk saat pola beban berubah.")
        dd = G_NORMAL[["Model", "Keluarga", "MAPE (%)"]].merge(G_COVID[["Model", "MAPE (%)"]], on="Model", suffixes=("_n", "_c"))
        dd = dd[dd.Keluarga.isin(keluarga)].sort_values("MAPE (%)_n", ascending=False)
        fig = go.Figure()
        for _, r in dd.iterrows():
            fig.add_scatter(x=[r["MAPE (%)_n"], r["MAPE (%)_c"]], y=[r.Model, r.Model], mode="lines", line=dict(color="#c3c2b7", width=3), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=[None], y=[None], mode="markers", name="periode normal", marker=dict(size=11, color="#52514e"))
        fig.add_scatter(x=[None], y=[None], mode="markers", name="periode COVID", marker=dict(size=11, color="white", line=dict(width=2.5, color="#52514e")))
        fig.add_scatter(x=dd["MAPE (%)_n"], y=dd.Model, mode="markers", showlegend=False, marker=dict(size=11, color=[WARNA[k] for k in dd.Keluarga]),
                        hovertemplate="%{y}<br>normal: %{x:.2f}%<extra></extra>")
        fig.add_scatter(x=dd["MAPE (%)_c"], y=dd.Model, mode="markers", showlegend=False, marker=dict(size=11, color="white", line=dict(width=2.5, color=[WARNA[k] for k in dd.Keluarga])),
                        hovertemplate="%{y}<br>COVID: %{x:.2f}%<extra></extra>")
        st.plotly_chart(tata(fig, max(360, 22 * len(dd) + 90), judul_x="MAPE (%)"), width="stretch")
        setara = MODEL[(MODEL.dm_p > 0.05) & (MODEL.peringkat > 1)].model.tolist()
        st.info(f"**Cara membaca peringkat dengan jujur.** Di seluruh data uji, {len(setara)} model tidak berbeda nyata dari model teratas menurut uji Diebold-Mariano: "
                + ", ".join(setara) + ". Jadi yang didukung data adalah adanya kelompok teratas yang setara, bukan satu pemenang mutlak.")

# =====================================================================================
with t_eda:
    st.subheader("Beban harian dan pembagian data")
    h = BEBAN.set_index("waktu")
    harian = h["beban"].resample("D").mean()
    fig = go.Figure(go.Scatter(x=harian.index, y=harian.values, mode="lines", line=dict(color=SERI[0], width=1.2), name="rata-rata harian", hovertemplate="%{x|%d %b %Y}<br>%{y:.0f} MWh<extra></extra>"))
    for a, b_, lab in (("2015-02-15", "2018-12-31", "latih"), ("2019-01-01", "2019-06-30", "validasi"), ("2019-07-01", "2020-02-29", "uji normal"), ("2020-03-01", "2020-06-26", "uji COVID")):
        fig.add_vline(x=pd.Timestamp(a), line=dict(color="#c3c2b7", width=1))
        fig.add_annotation(x=pd.Timestamp(a) + (pd.Timestamp(b_) - pd.Timestamp(a)) / 2, y=1.0, yref="paper", text=lab, showarrow=False, font=dict(size=12, color="#52514e"))
    st.plotly_chart(tata(fig, 330, judul_y="MWh", legenda=False), width="stretch")

    tahun = st.multiselect("Tahun yang dianalisis di bawah ini", sorted(h.index.year.unique()), default=[2015, 2016, 2017, 2018])
    hs = h[h.index.year.isin(tahun)] if tahun else h
    e1, e2 = st.columns(2)
    with e1:
        st.subheader("Pola jam × hari")
        pv = hs.pivot_table(index=hs.index.dayofweek, columns=hs.index.hour, values="beban", aggfunc="mean")
        fig = go.Figure(go.Heatmap(z=pv.values, x=[f"{j:02d}" for j in pv.columns], y=[HARI[i] for i in pv.index], colorscale="Blues", colorbar=dict(title="MWh"),
                                   hovertemplate="%{y}, pukul %{x}.00<br>%{z:.0f} MWh<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(tata(fig, 340, judul_x="jam", legenda=False), width="stretch")
        st.caption("Beban tertinggi terjadi siang hari kerja; Sabtu dan terutama Minggu jauh lebih rendah. Inilah alasan lag musiman (minggu lalu) menjadi input penting.")
    with e2:
        st.subheader("Hari libur melawan hari biasa")
        kerja = hs[(hs.index.dayofweek < 5)]
        pb = kerja[kerja.libur == 0].groupby(kerja[kerja.libur == 0].index.hour)["beban"].mean()
        pl = kerja[kerja.libur == 1].groupby(kerja[kerja.libur == 1].index.hour)["beban"].mean()
        fig = go.Figure()
        fig.add_scatter(x=pb.index, y=pb.values, mode="lines", name="hari kerja biasa", line=dict(color=SERI[0], width=2.5))
        fig.add_scatter(x=pl.index, y=pl.values, mode="lines", name="hari libur (Senin–Jumat)", line=dict(color=SERI[1], width=2.5))
        st.plotly_chart(tata(fig, 340, judul_y="MWh", judul_x="jam", legenda="x"), width="stretch")
        if len(pl):
            st.caption(f"Pada jam 13, hari libur rata-rata {f(pb.get(13, np.nan) - pl.get(13, np.nan), 0)} MWh lebih rendah daripada hari kerja biasa.")
    e3, e4 = st.columns(2)
    with e3:
        st.subheader("Suhu melawan beban")
        kota = st.selectbox("Kota", {"suhu_toc": "Tocumen (Kota Panama)", "suhu_san": "Santiago", "suhu_dav": "David"}.items(), format_func=lambda t: t[1])[0]
        smp = hs.sample(min(4000, len(hs)), random_state=0)
        fig = go.Figure(go.Scattergl(x=smp[kota], y=smp["beban"], mode="markers", marker=dict(size=4, color=smp.index.hour, colorscale="Blues", showscale=True, colorbar=dict(title="jam")),
                                     hovertemplate="%{x:.1f} °C<br>%{y:.0f} MWh<extra></extra>"))
        st.plotly_chart(tata(fig, 340, judul_y="MWh", judul_x="suhu (°C)", legenda=False), width="stretch")
        st.caption(f"Korelasi suhu dengan beban: {f(hs[kota].corr(hs['beban']))}. Warna titik menunjukkan jam; sebagian besar hubungan ini sebenarnya efek jam (siang lebih panas dan lebih sibuk).")
    with e4:
        st.subheader("Rata-rata beban per bulan")
        bl = h["beban"].resample("MS").mean()
        fig = go.Figure(go.Scatter(x=bl.index, y=bl.values, mode="lines+markers", line=dict(color=SERI[0], width=2), marker=dict(size=5), hovertemplate="%{x|%b %Y}<br>%{y:.0f} MWh<extra></extra>"))
        fig.add_vline(x=pd.Timestamp("2020-03-25"), line=dict(color="#d03b3b", dash="dash", width=1.5))
        fig.add_annotation(x=pd.Timestamp("2020-03-25"), y=1.0, yref="paper", text="karantina 25 Mar 2020", showarrow=False, xanchor="right", font=dict(size=12, color="#52514e"))
        st.plotly_chart(tata(fig, 340, judul_y="MWh", legenda=False), width="stretch")
        st.caption("Beban naik perlahan tiap tahun, lalu anjlok saat karantina COVID-19. Data juga memuat 8 jam beban di bawah 600 MWh (pemadaman) yang diinterpolasi sebelum pemodelan.")

# =====================================================================================
with t_ram:
    st.subheader("Ramalan melawan beban aktual")
    r1, r2, r3 = st.columns([1, 1, 2.4])
    mulai = r1.date_input("Mulai tanggal", value=pd.Timestamp("2020-04-06"), min_value=TGL[0].date(), max_value=TGL[-1].date())
    n_hari = r2.selectbox("Jumlah hari", [1, 3, 7, 14], index=2)
    bawaan = [MODEL[MODEL.keluarga == k].model.iloc[0] for k in WARNA if k in set(MODEL.keluarga) and k not in ("Patokan", "Model fondasi (fine-tuned)")]
    pilih = r3.multiselect("Model yang ditampilkan (maksimal 5)", list(RAM), default=bawaan, max_selections=5)
    i0 = int(np.searchsorted(TGL.values, np.datetime64(pd.Timestamp(mulai)))); i1 = min(len(TGL), i0 + n_hari)
    waktu = pd.date_range(TGL[i0], periods=(i1 - i0) * 24, freq="h")
    selang = st.checkbox("Tampilkan selang ramalan 80% (model pertama yang punya kuantil)", value=True)
    fig = go.Figure()
    ps = next((m for m in pilih if m in LO), None)
    if selang and ps:
        fig.add_scatter(x=waktu, y=HI[ps][i0:i1].reshape(-1), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=waktu, y=LO[ps][i0:i1].reshape(-1), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(42,120,214,0.18)", name=f"selang 80% {ps}", hoverinfo="skip")
    fig.add_scatter(x=waktu, y=AKT[i0:i1].reshape(-1), mode="lines", name="aktual", line=dict(color="#0b0b0b", width=2.6), hovertemplate="aktual: %{y:.0f} MWh<extra></extra>")
    for j, m in enumerate(pilih):
        fig.add_scatter(x=waktu, y=RAM[m][i0:i1].reshape(-1), mode="lines", name=m, line=dict(color=SERI[j % len(SERI)], width=1.7), hovertemplate=m + ": %{y:.0f} MWh<extra></extra>")
    st.plotly_chart(tata(fig, 420, judul_y="MWh", legenda="x"), width="stretch")
    libur_set = set(BEBAN.loc[BEBAN.libur == 1, "waktu"].dt.normalize())
    lib = [str(t.date()) for t in TGL[i0:i1] if t in libur_set]
    if lib: st.caption("Hari libur dalam rentang ini: " + ", ".join(lib) + ". Model fondasi biasanya meleset lebih jauh di hari libur.")
    if pilih:
        gw = galat(i0, i1); gw = gw[gw.Model.isin(pilih)].set_index("Model").loc[pilih, ["MAPE (%)", "MAE (MWh)", "RMSE (MWh)", "Bias (MWh)"]].reset_index()
        st.dataframe(gw, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in gw.columns[1:]})

        g1, g2 = st.columns(2)
        with g1:
            st.subheader("MAPE menurut jam target")
            fig = go.Figure()
            for j, m in enumerate(pilih):
                e = (np.abs(RAM[m][sl] - AKT[sl]) / AKT[sl] * 100).mean(0)
                fig.add_scatter(x=list(range(24)), y=e, mode="lines", name=m, line=dict(color=SERI[j % len(SERI)], width=2.2))
            st.plotly_chart(tata(fig, 340, judul_y="MAPE (%)", judul_x="jam", legenda="x"), width="stretch")
            st.caption("Dihitung pada periode yang dipilih di sidebar. Galat membesar pada jam kerja, saat beban paling tinggi.")
        with g2:
            st.subheader("MAPE per bulan")
            fig = go.Figure(); kunci = TGL.to_period("M")
            for j, m in enumerate(pilih):
                e = pd.Series((np.abs(RAM[m] - AKT) / AKT * 100).mean(1)).groupby(kunci).mean()
                fig.add_scatter(x=[f"{BULAN[p.month - 1]} {p.year}" for p in e.index], y=e.values, mode="lines+markers", name=m, line=dict(color=SERI[j % len(SERI)], width=2.2), marker=dict(size=5))
            st.plotly_chart(tata(fig, 340, judul_y="MAPE (%)", legenda="x"), width="stretch")
            st.caption("Seluruh data uji. Model yang dilatih pada data sebelum pandemi melonjak pada Maret–April 2020.")

# =====================================================================================
with t_input:
    st.subheader("Input mana yang paling menurunkan galat?")
    st.caption("Satu model (LightGBM) dilatih berulang sambil menambah satu kelompok input setiap kali, mengikuti peta jenis input dari kuliah.")
    si = STUDI[STUDI.studi == "input"].sort_values("nama").copy()
    si["label"] = si.nama.str.replace(r"^(\d+) ", lambda m: f"{int(m.group(1))}. ", regex=True)
    kol = {"Seluruh": "mape_semua", "Uji periode normal": "mape_normal", "Uji periode COVID": "mape_covid"}
    kp = next((v for k, v in kol.items() if periode.startswith(k)), "mape_semua")
    i1_, i2_ = st.columns([1.3, 1])
    with i1_:
        warna = ["#898781" if "diketahui" in n else SERI[0] for n in si.nama]
        fig = go.Figure(go.Bar(y=si.label, x=si[kp], orientation="h", marker_color=warna, text=[f(v) for v in si[kp]], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>MAPE: %{x:.2f}%<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(tata(fig, 400, judul_x=f"MAPE (%), {periode.split(' (')[0].lower()}", legenda=False), width="stretch")
        st.caption("Batang abu-abu memakai cuaca pada jam target, yang di dunia nyata belum diketahui saat prediksi dibuat (skenario ramalan cuaca sempurna).")
    with i2_:
        gk = pd.Series(INFO["gain"]["kelompok"]).sort_values()
        fig = go.Figure(go.Bar(y=[k[2:].strip() for k in gk.index], x=gk.values * 100, orientation="h", marker_color=SERI[2], text=[f(v * 100, 0) + "%" for v in gk.values], textposition="outside", cliponaxis=False))
        st.plotly_chart(tata(fig, 400, judul_x="sumbangan menurut gain LightGBM (%)", legenda=False), width="stretch")
    a, b_ = si[kp].iloc[0], si[kp].iloc[1]
    st.markdown(f"- **Lag musiman** memberi lompatan terbesar: {f(a)}% menjadi {f(b_)}%.\n"
                f"- **Hari libur** memberi lompatan terbesar kedua: {f(si[kp].iloc[5])}% menjadi {f(si[kp].iloc[6])}%.\n"
                f"- **Cuaca kemarin** hampir tidak mengubah apa-apa ({f(si[kp].iloc[6])}% menjadi {f(si[kp].iloc[7])}%); informasinya sudah terkandung di beban kemarin.")
    with st.expander("Peta 12 jenis input dan pemakaiannya di data ini"):
        st.dataframe(pd.DataFrame([
            ("1. Lag target", "jam yang sama kemarin dan 2 hari lalu, nilai teramati terakhir", "Dipakai"),
            ("2. Lag musiman", "jam yang sama 7, 14, 21, 28 hari lalu", "Dipakai"),
            ("3. Statistik bergulir", "rata-rata, maks, min, simpangan baku kemarin; rata-rata 7 hari; EWMA", "Dipakai"),
            ("4. Selisih", "kemarin dikurangi minggu lalu; perubahan rata-rata harian dan mingguan", "Dipakai"),
            ("5. Kalender", "jam, hari, bulan, akhir pekan, tren waktu", "Dipakai"),
            ("6. Fourier", "sinus dan kosinus siklus harian, mingguan, tahunan", "Dipakai"),
            ("7. Libur & variasi kalender", "libur, jenis libur, libur kemarin/minggu lalu/besok, hari kejepit, sekolah", "Dipakai"),
            ("8. Eksogen tertinda", "cuaca kemarin di tiga kota", "Dipakai"),
            ("9. Eksogen diketahui", "cuaca pada jam target", "Hanya sebagai skenario ramalan sempurna"),
            ("10. Komponen dekomposisi", "mode VMD, IMF dari EMD dan CEEMDAN", "Diuji di tab Dekomposisi"),
            ("11. Intervensi & kejadian", "penanda karantina, penanda pemadaman", "Diuji di tab Studi lain"),
            ("12. Lintas-deret & hierarki", "tidak ada (satu deret nasional)", "Tidak relevan"),
        ], columns=["Jenis input", "Wujudnya di data ini", "Status"]), hide_index=True, width="stretch")

# =====================================================================================
with t_dek:
    st.subheader("Dekomposisi sinyal: VMD, EMD, CEEMDAN")
    st.caption("Kurva beban dipecah menjadi 6 komponen, lalu komponen itu dipakai sebagai input tambahan. Dekomposisi dihitung ulang di tiap titik prediksi dari 504 jam terakhir, supaya tidak memakai data masa depan.")
    sd = STUDI[STUDI.studi == "dekomp"].copy()
    urut = ["tanpa", "VMD", "EMD", "CEEMDAN", "VMD-bocor"]; wd = {"tanpa": "#898781", "VMD": SERI[0], "EMD": SERI[1], "CEEMDAN": SERI[2], "VMD-bocor": "#d03b3b"}
    fig = go.Figure()
    for v in urut:
        s = sd[sd.varian == v]
        if len(s): fig.add_bar(x=s.model, y=s[kp], name=v, marker_color=wd[v], text=[f(x) for x in s[kp]], textposition="outside", cliponaxis=False, hovertemplate="%{x} · " + v + "<br>MAPE: %{y:.2f}%<extra></extra>")
    fig.update_layout(barmode="group")
    st.plotly_chart(tata(fig, 420, judul_y=f"MAPE (%), {periode.split(' (')[0].lower()}"), width="stretch")
    lg = sd[sd.model == "LightGBM"].set_index("varian")[kp]
    d1, d2, d3 = st.columns(3)
    d1.metric("LightGBM tanpa dekomposisi", f(lg["tanpa"]) + "%")
    d2.metric("LightGBM + VMD yang sah", f(lg["VMD"]) + "%", ("+" if lg["VMD"] >= lg["tanpa"] else "") + f(lg["VMD"] - lg["tanpa"]) + " poin", delta_color="inverse")
    d3.metric("LightGBM + VMD bocor", f(lg["VMD-bocor"]) + "%", f(lg["VMD-bocor"] - lg["tanpa"]) + " poin", delta_color="inverse")
    st.warning("**Versi bocor** menghitung VMD sekali atas seluruh deret sebelum data dibagi, cara yang umum di makalah. Nilai komponen hari ini jadi ikut dihitung dari data besok, "
               "sehingga akurasinya tampak jauh lebih baik. Angka itu tidak bisa dicapai dalam pemakaian nyata.")
    st.markdown("- Pada model terlatih (LightGBM, N-BEATSx, CNN-LSTM), dekomposisi yang sah hampir tidak mengubah galat.\n"
                "- Pada model fondasi, dekomposisi merugikan: galat tiap komponen menumpuk saat dijumlahkan kembali.")

# =====================================================================================
with t_lain:
    st.subheader("Uji kontaminasi: kenapa TimesFM dikeluarkan dari peringkat")
    l1, l2 = st.container(), st.container()
    with l1:
        fig = go.Figure()
        for j, m in enumerate(KONT.model.unique()):
            s = KONT[KONT.model == m].sort_values("jam"); tf = "TimesFM" in m
            fig.add_scatter(x=s.jam, y=s.mape, mode="lines", name=m, line=dict(color="#d03b3b" if tf else SERI[j], width=3 if tf else 2))
        st.plotly_chart(tata(fig, 330, judul_y="MAPE (%)", judul_x="jam horizon ke-", legenda="x"), width="stretch")
    with l2:
        kt = KONT.groupby("model", sort=False).agg(asli=("mape_asli", "first"), derau=("mape_berderau", "first")).reset_index()
        kt["Kenaikan"] = kt.derau / kt.asli
        kt.columns = ["Model", "MAPE riwayat asli (%)", "MAPE riwayat berderau 3% (%)", "Kenaikan (kali)"]
        st.dataframe(kt, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in kt.columns[1:]})
        st.caption("Peramal yang sah makin tidak pasti makin jauh ke depan. Galat TimesFM rata di semua jam dan melonjak saat riwayat diberi derau: tanda kuat data ini ada di data pralatihnya.")

    st.subheader("Transfer learning: zero-shot melawan fine-tuning")
    tl = G_SEMUA.merge(G_NORMAL[["Model", "MAPE (%)"]], on="Model", suffixes=("", " normal")).merge(G_COVID[["Model", "MAPE (%)"]], on="Model", suffixes=("", " COVID"))
    tl = tl[tl.Model.isin(["Chronos-2", "Chronos-2 FT-LoRA", "Chronos-2 FT-penuh", "Chronos-2 FT-LoRA+"])][["Model", "MAPE (%) normal", "MAPE (%) COVID", "MAPE (%)"]]
    tl = tl.rename(columns={"MAPE (%)": "MAPE (%) semua"})
    tl.insert(1, "Cara", tl.Model.map({"Chronos-2": "tanpa pelatihan (zero-shot)", "Chronos-2 FT-LoRA": "LoRA, 100 langkah, konteks 512 jam", "Chronos-2 FT-penuh": "semua bobot, 100 langkah, konteks 512 jam", "Chronos-2 FT-LoRA+": "LoRA, 200 langkah, konteks 1.008 jam"}))
    st.dataframe(tl, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in tl.columns[2:]})
    st.caption("Fine-tuning dijalankan di CPU dengan anggaran sangat kecil dan belum mengalahkan zero-shot. Menaikkan anggaran memperbaiki hasil, jadi ini batas bawah.")

    s1, s2 = st.columns(2)
    with s1:
        st.subheader("Adaptasi saat pola berubah")
        ad = STUDI[STUDI.studi == "adaptasi"].assign(nama=lambda d_: d_.nama.map({"statis": "dilatih sekali", "latih ulang mingguan": "latih ulang mingguan", "latih ulang mingguan +penanda intervensi": "mingguan + penanda"}))
        fig = go.Figure()
        for lab, c, w in (("uji normal", "mape_normal", SERI[0]), ("uji COVID", "mape_covid", SERI[1])):
            fig.add_bar(x=ad.nama, y=ad[c], name=lab, marker_color=w, text=[f(v) for v in ad[c]], textposition="outside", cliponaxis=False)
        fig.update_layout(barmode="group"); fig.update_xaxes(categoryorder="array", categoryarray=["dilatih sekali", "latih ulang mingguan", "mingguan + penanda"], tickangle=0)
        st.plotly_chart(tata(fig, 360, judul_y="MAPE LightGBM (%)"), width="stretch")
        st.caption("Melatih ulang LightGBM tiap minggu memangkas galat saat COVID. Chronos-2 zero-shot mendapat " + f(G_COVID.set_index("Model").loc["Chronos-2", "MAPE (%)"]) + "% tanpa pelatihan ulang.")
    with s2:
        st.subheader("Panjang riwayat untuk model fondasi")
        kx = STUDI[STUDI.studi == "konteks"].assign(jam=lambda d_: d_.nama.astype(int))
        kx = pd.concat([kx[["jam", "mape_semua"]], pd.DataFrame({"jam": [1008], "mape_semua": [G_SEMUA.set_index("Model").loc["Chronos-Bolt-small", "MAPE (%)"]]})]).sort_values("jam")
        fig = go.Figure(go.Scatter(x=[f"{j} jam" for j in kx.jam], y=kx.mape_semua, mode="lines+markers+text", text=[f(v) for v in kx.mape_semua], textposition="top center", line=dict(color=SERI[0], width=2.5), marker=dict(size=8)))
        st.plotly_chart(tata(fig, 360, judul_y="MAPE Chronos-Bolt-small (%)", legenda=False), width="stretch")
        st.caption("Satu minggu riwayat (168 jam) belum cukup. Sekitar 4 sampai 6 minggu sudah optimal.")

    s3, s4 = st.columns(2)
    with s3:
        st.subheader("Efisiensi data")
        ef = STUDI[STUDI.studi == "efisiensi"].assign(hari=lambda d_: d_.varian.astype(int)).sort_values("hari")
        gs = G_NORMAL.set_index("Model")["MAPE (%)"]
        et = pd.DataFrame({"Model": ["Chronos-2 (zero-shot)"] + [f"N-BEATSx, {h_} hari data latih" for h_ in ef.hari] + ["N-BEATSx, seluruh data latih"],
                           "MAPE uji normal (%)": [gs["Chronos-2"]] + list(ef.mape_normal) + [gs["N-BEATSx"]]})
        st.dataframe(et, hide_index=True, width="stretch", column_config={"MAPE uji normal (%)": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Model fondasi tidak butuh data latih Panama sama sekali. N-BEATSx yang dilatih dari nol butuh lebih dari 180 hari data untuk menyamainya.")
    with s4:
        st.subheader("AIC dan BIC (hanya model statistik)")
        stt = pd.DataFrame(INFO["stat"]).T.reset_index(); stt.columns = ["Model", "Spesifikasi", "AIC", "BIC", "Parameter"]
        stt["AIC"] = stt.AIC.astype(float).round(0); stt["BIC"] = stt.BIC.astype(float).round(0)
        st.dataframe(stt, hide_index=True, width="stretch", column_config={"AIC": st.column_config.NumberColumn(format="%d"), "BIC": st.column_config.NumberColumn(format="%d")})
        st.caption("AIC/BIC butuh likelihood dan parameter yang ditaksir dari data ini, jadi tidak terdefinisi untuk model fondasi zero-shot. Hanya SARIMA dan SARIMAX yang sebanding satu sama lain.")

# =====================================================================================
with t_coba:
    st.subheader("Coba ramal sendiri")
    st.caption("Pilih data, tanggal, dan model. Ramalan dibuat untuk 24 jam mulai pukul 00.00 tanggal tersebut, hanya memakai data sebelum tanggal itu.")
    sumber = st.radio("Sumber data", ["Data Panama (bawaan)", "Unggah file sendiri (CSV atau Excel)"], horizontal=True)
    deret = None
    if sumber.startswith("Data"):
        deret = BEBAN.set_index("waktu")["beban"]
    else:
        up = st.file_uploader("File berisi satu kolom waktu per jam dan satu kolom nilai", type=["csv", "xlsx"])
        if up is not None:
            df = pd.read_excel(up) if up.name.endswith("xlsx") else pd.read_csv(up)
            u1, u2 = st.columns(2)
            kw = u1.selectbox("Kolom waktu", df.columns); kn = u2.selectbox("Kolom nilai", [c for c in df.columns if c != kw and pd.api.types.is_numeric_dtype(df[c])])
            try:
                deret = pd.Series(df[kn].values, index=pd.to_datetime(df[kw])).sort_index().asfreq("h").interpolate()
            except Exception as e:
                st.error(f"Kolom waktu tidak bisa dibaca sebagai waktu per jam: {e}")
    if deret is not None and len(deret) > 24 * 15:
        hari_ada = pd.DatetimeIndex(sorted(set(deret.index.normalize())))
        terakhir = hari_ada[-1] + pd.Timedelta(days=1)
        o1, o2, o3 = st.columns(3)
        tgl = o1.date_input("Tanggal yang diramal", value=min(pd.Timestamp("2020-04-06"), terakhir).date() if sumber.startswith("Data") else terakhir.date(),
                            min_value=(hari_ada[0] + pd.Timedelta(days=14)).date(), max_value=terakhir.date())
        mdl = o2.selectbox("Model", ["Chronos-Bolt-small (model fondasi, ringan)", "Chronos-2 (model fondasi, lebih lambat)", "Naif musiman (jam yang sama minggu lalu)"])
        konteks = o3.selectbox("Panjang riwayat", [336, 672, 1008], index=2, format_func=lambda v: f"{v} jam ({v // 168} minggu)")
        if st.button("Ramal 24 jam", type="primary"):
            t0 = pd.Timestamp(tgl); riw = deret[deret.index < t0].dropna()
            waktu = pd.date_range(t0, periods=24, freq="h"); lo_ = hi_ = None
            if len(riw) < 168:
                st.error("Riwayat sebelum tanggal itu kurang dari 1 minggu.")
            else:
                if mdl.startswith("Naif"):
                    p = riw.values[-168:-144]
                else:
                    try:
                        import torch
                        from chronos import BaseChronosPipeline

                        @st.cache_resource(show_spinner="Memuat model (pertama kali butuh unduhan)...")
                        def pipa(nama):
                            return BaseChronosPipeline.from_pretrained(nama, device_map="cpu")
                        nm = "amazon/chronos-bolt-small" if "Bolt" in mdl else "amazon/chronos-2"
                        x = riw.values[-konteks:].astype("float32")
                        with st.spinner("Meramal..."):
                            q, _ = pipa(nm).predict_quantiles([torch.tensor(x)] if "Bolt" in mdl else [x], prediction_length=24, quantile_levels=[0.1, 0.5, 0.9])
                        q = np.asarray(q[0]); q = q[0] if q.ndim == 3 else q
                        lo_, p, hi_ = q[:, 0], q[:, 1], q[:, 2]
                    except ImportError:
                        st.error("Pustaka Chronos belum terpasang. Jalankan `pip install chronos-forecasting torch`, lalu mulai ulang aplikasi. Sementara itu, pilih model Naif musiman.")
                        p = None
                if p is not None:
                    akt = deret.reindex(waktu)
                    fig = go.Figure()
                    lalu = riw.iloc[-72:]
                    fig.add_scatter(x=lalu.index, y=lalu.values, mode="lines", name="riwayat (3 hari terakhir)", line=dict(color="#898781", width=1.6))
                    if lo_ is not None:
                        fig.add_scatter(x=waktu, y=hi_, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
                        fig.add_scatter(x=waktu, y=lo_, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(42,120,214,0.18)", name="selang ramalan 80%", hoverinfo="skip")
                    if akt.notna().any(): fig.add_scatter(x=waktu, y=akt.values, mode="lines", name="aktual", line=dict(color="#0b0b0b", width=2.6))
                    fig.add_scatter(x=waktu, y=p, mode="lines", name="ramalan", line=dict(color=SERI[0], width=2.4))
                    st.plotly_chart(tata(fig, 400, judul_y="nilai", legenda="x"), width="stretch")
                    hasil = pd.DataFrame({"waktu": waktu, "ramalan": np.round(p.astype(float), 1)})
                    if lo_ is not None: hasil["batas_bawah"], hasil["batas_atas"] = np.round(lo_.astype(float), 1), np.round(hi_.astype(float), 1)
                    if akt.notna().all():
                        hasil["aktual"] = akt.values
                        m1, m2 = st.columns(2)
                        m1.metric("MAPE hari ini", f(float(np.mean(np.abs(p - akt.values) / akt.values) * 100)) + "%")
                        m2.metric("MAE hari ini", f(float(np.mean(np.abs(p - akt.values))), 1))
                    else:
                        st.caption("Belum ada nilai aktual untuk tanggal ini, jadi galat tidak bisa dihitung.")
                    st.dataframe(hasil, hide_index=True, width="stretch")
                    st.download_button("Unduh ramalan (CSV)", hasil.to_csv(index=False).encode("utf-8"), f"ramalan_{tgl}.csv", "text/csv")
    elif deret is not None:
        st.warning("Data terlalu pendek; butuh minimal 15 hari data per jam.")
