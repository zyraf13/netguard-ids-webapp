# NetGuard — Web Deteksi Intrusi Jaringan

Web app ini dibangun dari notebook riset `pipeline_final.ipynb`, yang membandingkan
model **KNN**, **Random Forest**, dan **XGBoost** pada tiga dataset: **UNSW-NB15**,
**NSL-KDD**, dan **CICIDS2019**.

## Struktur folder

```
ids_webapp/
├── data/                     ⟵ taruh file dataset mentah di sini (belum ada, harus diisi manual)
│   ├── unsw_nb15/
│   ├── nsl_kdd/
│   └── cicids2019/
├── training/                 ⟵ script untuk melatih & menyimpan model
│   ├── common.py
│   ├── train_unsw_nb15.py
│   ├── train_nsl_kdd.py
│   └── train_cicids2019.py
├── artifacts/                ⟵ akan dibuat otomatis setelah training (model, scaler, encoder, metadata)
├── backend/                  ⟵ REST API (FastAPI)
│   ├── app.py
│   └── requirements.txt
└── frontend/                 ⟵ dashboard web (HTML/JS statis, tidak perlu build tool)
    └── index.html
```

## Langkah 1 — Siapkan dataset

Unduh dan taruh file berikut (nama file harus persis sama):

| Dataset | File | Taruh di |
|---|---|---|
| UNSW-NB15 | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` | `data/unsw_nb15/` |
| NSL-KDD | `Train_data.csv`, `Test_data.csv` (delimiter `;`) | `data/nsl_kdd/` |
| CICIDS2019 | `syn_data.csv` (dari Kaggle: `tarundhamor/cicids-2019-dataset`) | `data/cicids2019/` |

Kamu tidak perlu menyiapkan ketiganya sekaligus — dataset mana pun yang belum
ada file-nya akan otomatis dilewati / muncul errornya sendiri saat training,
dan dashboard hanya akan menampilkan dataset yang berhasil dilatih.

## Langkah 2 — Install dependency

```bash
cd ids_webapp
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

## Langkah 3 — Latih model & simpan artifacts

Jalankan salah satu atau semua, sesuai dataset yang sudah kamu siapkan:

```bash
cd training
python train_unsw_nb15.py
python train_nsl_kdd.py
python train_cicids2019.py
```

Setiap script akan:
1. Load & preprocessing data (sama seperti notebook riset asli).
2. Melatih KNN, Random Forest, XGBoost.
3. Memilih model dengan **F1-score** tertinggi sebagai model utama.
4. Menyimpan `model.joblib`, `scaler.joblib`, `encoders.joblib`, dan
   `metadata.json` (berisi daftar fitur, hasil evaluasi semua model, arti label)
   ke `artifacts/<nama_dataset>/`.

## Langkah 4 — Jalankan backend API

```bash
cd backend
uvicorn app:app --reload --port 8000
```

Cek di browser: `http://localhost:8000/api/datasets` — harus menampilkan
dataset yang tadi berhasil dilatih.

## Langkah 5 — Buka frontend

Buka file `frontend/index.html` langsung di browser (double click, atau lewat
extension "Live Server" di VS Code). Dashboard akan otomatis:
- Menampilkan tombol pilihan dataset yang tersedia.
- Menampilkan tabel perbandingan performa (accuracy, precision, recall, F1) semua model.
- Membuat form input sesuai fitur dataset yang dipilih.
- Mengirim data ke `/predict` dan menampilkan hasil (Normal / Attack).

> Jika ingin frontend & backend berjalan di domain/port berbeda saat deploy
> nanti, ubah nilai `API_BASE` di bagian atas tag `<script>` pada `index.html`.

## Deploy ke internet (opsional, langkah lanjutan)

- **Backend**: bisa dideploy ke Railway, Render, atau VPS biasa (jalankan
  `uvicorn` di baliknya, atau bungkus dengan Docker).
- **Frontend**: karena hanya HTML statis, bisa langsung di-host di Netlify,
  Vercel, atau GitHub Pages.
- Ingat: folder `artifacts/` (berisi model hasil training) harus ikut
  di-deploy bersama backend, karena backend membacanya saat start.

## Catatan penting

- Struktur fitur UNSW-NB15, NSL-KDD, dan CICIDS2019 **berbeda satu sama lain**,
  sehingga form input di dashboard akan otomatis berubah mengikuti dataset
  yang dipilih (dibaca dari `metadata.json` masing-masing).
- Kolom kategorikal (misalnya `proto`, `service`, `protocol_type`) harus diisi
  dengan nilai yang sama seperti pada data training aslinya (contoh: `tcp`,
  `http`), karena encoder tidak bisa mengenali kategori baru yang belum pernah
  dilihat saat training.
