"""Kumpulkan hasil eksperimen menjadi paket data kecil untuk dashboard (jalankan dari folder panama-forecast)."""
import sys, json, shutil, numpy as np, pandas as pd
sys.path.insert(0, ".")
from s00_setup import HOURLY, DATES, Y, TE, TE_N, PRED
OUT = "dashboard/data"
m = pd.read_csv("hasil/dash/model.csv"); m.to_csv(f"{OUT}/model.csv", index=False)
isi = {"tanggal": np.array([str(d.date()) for d in DATES[TE]]), "aktual": Y[TE].astype(np.float32), "n_normal": np.array(len(TE_N))}
for n in m.model:
    isi["m::" + n] = np.load(f"{PRED}/{n}.npz")["test"].astype(np.float32)
    try:
        q = np.load(f"{PRED}/_q_{n}.npz"); isi["lo::" + n] = q["test10"].astype(np.float32); isi["hi::" + n] = q["test90"].astype(np.float32)
    except FileNotFoundError: pass
np.savez_compressed(f"{OUT}/ramalan.npz", **isi)
h = HOURLY.reset_index()
pd.DataFrame({"waktu": h["datetime"], "beban": h["y"].round(2), "beban_asli": h["nat_demand"].round(2), "suhu_toc": h["T2M_toc"].round(2), "suhu_san": h["T2M_san"].round(2),
              "suhu_dav": h["T2M_dav"].round(2), "libur": h["holiday"], "jenis_libur": h["Holiday_ID"], "sekolah": h["school"]}).to_parquet(f"{OUT}/beban.parquet", index=False)
for f in ("studi.csv", "kontaminasi.csv"): shutil.copy(f"hasil/dash/{f}", f"{OUT}/{f}")
json.dump({"stat": json.load(open(f"{PRED}/_stat_info.json")), "gain": json.load(open(f"{PRED}/_gain.json"))}, open(f"{OUT}/info.json", "w"), indent=1)
import os
for f in sorted(os.listdir(OUT)): print(f, os.path.getsize(f"{OUT}/{f}") // 1024, "KB")
print("kunci selang:", [k[4:] for k in isi if k.startswith("lo::")])
