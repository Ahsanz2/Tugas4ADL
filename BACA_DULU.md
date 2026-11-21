# Dashboard Peramalan Beban Listrik Panama (Streamlit)

## Menjalankan di komputer sendiri
1. Pasang Python 3.10 atau lebih baru.
2. Buka terminal di folder ini, lalu pasang pustaka:
   `pip install -r requirements.txt`
3. Jalankan:
   `streamlit run app.py`
4. Browser terbuka otomatis di `http://localhost:8501`.

## Tab "Coba ramal sendiri"
Model "Naif musiman" selalu bisa dipakai. Untuk model Chronos, pasang juga:
`pip install chronos-forecasting torch`
Saat pertama dipakai, bobot model terunduh otomatis (Chronos-Bolt-small sekitar 190 MB, Chronos-2 sekitar 480 MB).

## Menaruh di internet (gratis)
1. Unggah folder ini ke repositori GitHub.
2. Buka share.streamlit.io, login dengan GitHub, pilih repositori dan `app.py`.
3. Untuk Chronos di sana, hapus tanda `#` pada dua baris terakhir `requirements.txt`.

## Isi folder
| Berkas | Isi |
|---|---|
| `app.py` | seluruh kode dashboard |
| `data/model.csv` | tabel perbandingan 29 model |
| `data/ramalan.npz` | beban aktual dan ramalan tiap model untuk 362 hari uji |
| `data/beban.parquet` | data per jam (beban, suhu tiga kota, libur, sekolah) |
| `data/studi.csv`, `data/kontaminasi.csv`, `data/info.json` | hasil studi tambahan, AIC/BIC |
| `siapkan_data.py` | pembuat folder `data/` dari hasil eksperimen (tidak perlu dijalankan lagi) |
| `.streamlit/config.toml` | tema warna |
