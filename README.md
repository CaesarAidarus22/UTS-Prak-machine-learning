# UTS Praktikum Machine Learning

## Anggota Kelompok
- Muhammad Reky
- Riyan Hadi Samudra
- M. Caesar Aidarus

## Ringkasan Project
Project ini membangun sistem klasifikasi untuk memprediksi apakah kondisi agro-environmental tertentu termasuk `Suitable` atau `Not Suitable` berdasarkan beberapa fitur tanah:
- `bulk_density`
- `organic_matter_pct`
- `cation_exchange_capacity`
- `salinity_ec`

Model dilatih di notebook, disimpan ke folder `model`, lalu diintegrasikan ke aplikasi web dengan:
- FastAPI sebagai backend
- Streamlit sebagai frontend

## Struktur Folder
- `application/backend`: API FastAPI dan loader model
- `application/frontend`: UI Streamlit
- `dataset`: dataset training
- `model`: artefak model hasil training
- `notebook`: notebook eksplorasi, training, tuning, evaluasi, dan export model

## Menjalankan Notebook
Install dependency notebook:

```powershell
pip install -r notebook/requirements.txt
```

Lalu buka `notebook/notebook.ipynb` dan jalankan seluruh cell sampai model tersimpan ke `model/pipeline.pkl`.

## Menjalankan Backend
Install dependency backend:

```powershell
pip install -r application/backend/requirements.txt
```

Masuk ke folder backend lalu jalankan:

```powershell
uvicorn main:app --reload
```

Backend default berjalan di `http://127.0.0.1:8000`.

Smoke test backend:

```powershell
python -m unittest application.backend.test_smoke
```

## Menjalankan Frontend
Install dependency frontend:

```powershell
pip install -r application/frontend/requirements.txt
```

Masuk ke folder frontend lalu jalankan:

```powershell
streamlit run main.py
```

## Endpoint Penting
- `GET /`: status API
- `GET /health`: status model dan path model
- `POST /predict`: prediksi kelayakan kondisi tanah

## Catatan
- Pastikan backend dan frontend berjalan bersamaan.
- Pastikan file model `model/pipeline.pkl` tersedia sebelum demo.
- Folder virtual environment tidak perlu disertakan saat pengumpulan.
- Jika backend dijalankan di alamat lain, set environment variable `BACKEND_URL` untuk frontend.
