# Otomatik E-Commerce ML Data Pipeline — Sistem Mimarisi Raporu

## 1) Kullanılan Teknolojiler ve Araçlar

### 1.1 Dil / Çalışma Ortamı

- **Python**
- Veri işleme: **pandas**, **numpy**
- Dosya/JSON: `os`, `json`

### 1.2 Makine Öğrenmesi (Modelleme)

- **scikit-learn**
  - Model: `RandomForestClassifier`
  - Özellik kodlama: `LabelEncoder`
  - Eğitim/Değerlendirme: `train_test_split`, `accuracy_score`, `classification_report`
- Model/encoder serileştirme:
  - `joblib` (encoders)
  - Model diskte paketli halde tutulur ve API tarafında yüklenir

### 1.3 Experiment Tracking / Model Versiyonlama

- **MLflow**
  - Tracking backend: `sqlite:///mlflow.db`
  - Deney adı: `sales_pipeline`
  - Train sırasında:
    - `mlflow.log_params`, `mlflow.log_metric`
    - `mlflow.sklearn.log_model(..., "random_forest_model")`
  - Servis sırasında:
    - `mlflow.sklearn.load_model(MODEL_DIR)` ile model yükleme

### 1.4 Drift Monitoring (Dağılım Kayması)

- **SciPy**
  - Drift testi: `scipy.stats.ks_2samp` (Kolmogorov–Smirnov)

### 1.5 Veri Dönüşümü / Warehouse Mantığı

- **dbt**
  - `dbt run` ile dönüşüm
  - `dbt test` ile veri kalite kontrolleri
  - Model SQL’leri:
    - `dbt_project/models/staging/stg_retail.sql`
    - `dbt_project/models/marts/retail_clean.sql`
  - Testler: `dbt_project/models/*/schema.yml` (ör. `not_null`, `accepted_values`)
- **DuckDB**
  - dbt çıktısını okumak için `data/warehouse.duckdb` kullanımı

### 1.6 Orkestrasyon / Scheduler

- **Apache Airflow**
  - Docker Compose ile koşturulur
  - DAG: `dags/pipeline_dag.py`
  - Zamanlama: günlük `02:00`
- Local scheduler:
  - **schedule** kütüphanesi (`scheduler.py`)

### 1.7 Model Serving (REST API)

- **FastAPI** + **Uvicorn**
  - Endpointler: `GET /`, `GET /health`, `POST /predict`

### 1.8 LLM (Raporlama) ve RAG

- **Groq**
  - `main_pipeline.py` içinde CEO odaklı executive rapor
- **Ollama**
  - `alert_monitor.py` içinde “investigator agent” teşhis üretimi
- **RAG / semantik arama**
  - `rag_utils.py` ile rapor embedding saklama ve benzer rapor arama
  - Arama için DuckDB tabanlı yardımcı katman

### 1.9 Konteyner / Deployment

- **Docker**
  - `Dockerfile`
  - `Dockerfile.airflow`
- **Docker Compose**
  - `docker-compose.yml`: Airflow + PostgreSQL
- **Procfile**
  - Render start komutu: `uvicorn api:app --host 0.0.0.0 --port $PORT`

---

## 2) Sistem Çalışma Mantığı ve Akışı (Data Pipeline + Döngüler)

Bu proje; **(1) eğitim**, **(2) model sunumu**, **(3) drift/alert izlemesi**, **(4) gerektiğinde otomatik retrain** döngüsünü bir arada yürütür.

### 2.1 Veri Kaynağı ve Hazırlık

- Ham veri: `data/sales_data.csv` (proje içinde CSV tabanlı pipeline da var)
- Ek dönüşüm katmanı: **dbt** ile feature-ready tablo üretimi
- dbt çıktısı: DuckDB içindeki `retail_clean` tablosu

### 2.2 Feature Üretimi (dbt ile)

`retail_clean` modeli iade/iş hedefi için etiket ve tahmin için özellikleri üretir.

- **Etiket (target):**
  - `returned = 1` (invoice_no `C%` ise iade kabul)
- **Gelir:**
  - `revenue = quantity * price`
- **Zaman feature’ları:**
  - `month`
  - `day_of_week`
- **Müşteri (history/RFM-benzeri) feature’ları:**
  - `customer_prior_orders`
  - `customer_avg_order_value`
  - `customer_return_rate`
  - `customer_recency_days` (lag + gün farkı)

Ayrıca `schema.yml` içinde `not_null` / `accepted_values` gibi kalite testleri tanımlıdır.

### 2.3 Eğitim ve Deney Kaydı (main_pipeline.py)

`main_pipeline.py` her çalıştırıldığında:

1. **dbt run/test** çalıştırır
2. DuckDB’den `retail_clean` tablosunu yükler
3. Kategorik alanları `LabelEncoder` ile encode eder
4. Modeli eğitir:
   - `RandomForestClassifier`
   - train/test ayrımı: `stratify=y`
5. **MLflow** kaydı yapar:
   - parametre/metric loglama
   - modelin `random_forest_model` artifact olarak loglanması
6. **Encoder’ları diske yazar** (`encoders/`)
7. **LLM raporu üretir**:
   - Groq ile executive report
   - geçmiş raporlardan trend içgörüsü (RAG + MLflow accuracy geçmişi)
8. Raporu dosyaya yazar:
   - `reports/report_YYYYMMDD_HHMMSS.txt`
9. Son accuracy’yi izleme için kaydeder:
   - `data/last_accuracy.json`

### 2.4 Drift Monitoring (drift_monitor.py)

`drift_monitor.py`:

1. `data/sales_data.csv` okur
2. `revenue`, `month`, `day_of_week` gibi yardımcı alanları türetir
3. KS-test uygular:
   - reference vs current ayrımıyla (random split)
   - seçilen feature’lar üzerinde p-value hesaplar
4. Sonuçları raporlar:
   - `reports/drift_report_YYYYMMDD_HHMMSS.json`

Drift tespiti “p-value < threshold” mantığıyla yapılır ve genel karar, drifted feature oranına göre verilir.

### 2.5 Auto Retrain Trigger (auto_retrain.py)

`auto_retrain.py` son durumu okur:

- en güncel drift raporu (`drift_detected`?)
- en güncel accuracy (`report_*.txt` içindeki `Model Accuracy:` satırından parse)

Karar:

- `drift_detected == True` **veya** `latest_accuracy < 0.85`

Şart sağlanırsa:

- `python main_pipeline.py` yeniden çağrılır.

### 2.6 Alert Monitoring ve Investigator Agent (alert_monitor.py)

`alert_monitor.py`:

1. `data/pipeline.db` içinden `sales_data` tablo metriklerini okur
2. Eşik kontrolü yapar:
   - accuracy < 0.85 ⇒ `MODEL_DEGRADATION`
   - return_rate > 25 ⇒ `HIGH_RETURN_RATE`
3. Alert varsa:
   - `ollama.chat` ile investigator agent teşhis üretir
   - teşhisi `logs/alerts.log` içine yazar
4. İstersen e-posta gönderimi de yapılabilir (Gmail SMTP + App Password gerektirir).

---

## 3) Döngülerin Birlikte Çalıştığı Orkestrasyon (Airflow DAG)

`dags/pipeline_dag.py` içinde DAG şu sıralamayı kurar:

1. `main_pipeline.py`
2. `drift_monitor.py`
3. `auto_retrain.py`
4. `alert_monitor.py`

DAG günde bir kez (02:00) koşturulur.

Ek olarak `scheduler.py` aynı mantığı local ortamda da (startup + her gün 02:00) tetikler.

---

## 4) Ana Dosyaların Görevleri

- **main_pipeline.py**:
  - dbt dönüşüm/test
  - veri yükleme ve feature seti
  - RandomForest eğitimi + MLflow loglama
  - encoder kaydı
  - Groq executive rapor + RAG trend içgörüsü
  - `reports/` ve `data/last_accuracy.json` üretimi

- **api.py**:
  - model + encoder’ları startup’ta diskten yükler
  - `POST /predict` ile encode → feature vector → predict/predict_proba
  - response: `prediction`, `probability`, `label`

- **dags/pipeline_dag.py**:
  - Airflow DAG tanımı ve günlük job akışı

- **drift_monitor.py**:
  - KS-test drift analizi
  - drift raporu üretimi

- **auto_retrain.py**:
  - en güncel drift ve accuracy bilgisini okuyup retrain tetikleme

- **alert_monitor.py**:
  - SQLite metriklerinden eşik kontrolü
  - alert varsa Ollama ile investigator agent diagnosis

- **scheduler.py**:
  - local schedule ile pipeline’ı 02:00’da tetikleme

- **db_setup.py**:
  - `data/pipeline.db` SQLite şemasını ve `sales_data` yüklemesini sağlar

- **docker-compose.yml / Dockerfile / Dockerfile.airflow / Procfile**:
  - konteyner ortamı, Airflow servisi ve API deployment ayarları

---

## 5) Kısa Sonuç (Sistem Özeti)

Bu proje; **dbt ile feature üretir → RandomForest eğitir ve MLflow’a loglar → Groq ile rapor üretir → drift izler (KS-test) → drift/accuracy eşiğine göre otomatik retrain yapar → alert olursa investigator agent ile teşhis üretir → FastAPI ile tahmini servise sunar → Airflow ile günlük olarak tüm süreci orkestre eder.**
