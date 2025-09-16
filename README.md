# 🌱 MLOps Iris Demo  

Proyek ini merupakan implementasi sederhana **MLOps workflow** menggunakan:  
- **DVC (Data Version Control)** → untuk versioning dataset  
- **MLflow** → untuk tracking eksperimen dan model  
- **GitHub Actions** → untuk CI/CD otomatis  

Dataset yang digunakan adalah **Iris Dataset** (150 sampel, 4 fitur, 3 kelas).  

---

## 📂 Struktur Direktori  

```

mlops-iris/
│
├── data/                    # Folder dataset
│   └── iris.csv             # Dataset publik
│
├── src/                     # Kode training
│   ├── **init**.py
│   └── train.py
│
├── dvc.yaml                 # Pipeline DVC
├── data.dvc                 # Metadata DVC
├── requirements.txt         # Dependensi Python
├── README.md                # Dokumentasi proyek
│
├── .github/
│   └── workflows/
│       └── mlops.yml        # GitHub Actions workflow
│
├── .dvc/                    # Folder internal DVC (auto)
└── mlruns/                  # Hasil tracking MLflow (auto)

````

---

## ⚙️ Instalasi  

1. Clone repo:  
   ```bash
   git clone https://github.com/ShotZ9/mlops-iris.git
   cd mlops-iris
    ```

2. Buat virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependensi:

   ```bash
   pip install -r requirements.txt
   ```

4. Tarik dataset dengan DVC:

   ```bash
   dvc pull
   ```

---

## 🚀 Menjalankan Training

```bash
python src/train.py
```

* Model akan dilatih dengan **RandomForestClassifier**
* Metrik & parameter otomatis dicatat oleh **MLflow**

---

## 📊 Menjalankan MLflow UI

```bash
mlflow ui
```

Buka di browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)
→ Semua eksperimen, parameter, metrik, dan model akan muncul.

---

## 🔄 Pipeline DVC

Proyek ini menggunakan DVC untuk merekam workflow:

```bash
dvc repro
```

Akan menjalankan ulang training sesuai `dvc.yaml`.

---

## 🤖 CI/CD dengan GitHub Actions

Setiap kali ada perubahan kode yang dipush ke branch `main`, workflow otomatis akan:

1. Install dependensi
2. Pull dataset via DVC
3. Menjalankan training (`train.py`)
4. Menyimpan hasil eksperimen

File workflow: `.github/workflows/mlops.yml`

---

## 📌 Hasil Eksperimen

* Model: **RandomForestClassifier**
* Akurasi rata-rata: \~96% pada Iris Dataset
* Semua versi model tersimpan di MLflow registry

---

## 👨‍💻 Author

* **Yoel Amadeo Pratomo** ([@ShotZ9](https://github.com/ShotZ9))
