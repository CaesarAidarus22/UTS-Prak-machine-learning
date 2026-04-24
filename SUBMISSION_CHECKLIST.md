# Submission Checklist

## Wajib Dicek Sebelum Submit
- Jalankan notebook `notebook/notebook.ipynb` sampai selesai
- Pastikan file `model/pipeline.pkl` sudah terbentuk
- Pastikan backend bisa dijalankan dengan `uvicorn main:app --reload`
- Pastikan frontend bisa dijalankan dengan `streamlit run main.py`
- Pastikan prediksi berhasil dari frontend ke backend
- Pastikan folder virtual environment tidak ikut disubmit

## File yang Perlu Ada
- source code project lengkap
- `dataset/agro_environmental_dataset.csv`
- `model/pipeline.pkl`
- notebook yang berisi EDA, tuning, cross-validation, evaluasi, dan export model
- `application/backend/requirements.txt`
- `application/frontend/requirements.txt`
- `README.md`

## Lampiran Pengumpulan
- laporan proyek dalam format PDF
- video demo aplikasi

## Isi Minimal Laporan PDF
- ringkasan tujuan project
- proses EDA dan preprocessing
- metode model dan alasan pemilihan model
- hyperparameter tuning dan cross-validation
- hasil evaluasi model
- integrasi model ke backend dan frontend
- hasil akhir aplikasi
