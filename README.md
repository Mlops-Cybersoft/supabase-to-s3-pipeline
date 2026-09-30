# Supabase Transaction Pooler → S3 CSV Export

Project độc lập thay cho AWS DMS. Notebook kết nối Supabase qua **Transaction Pooler port 6543**, đọc từng bảng theo batch, ghi CSV tạm trên máy và upload CSV cùng manifest lên S3. Không dùng replication slot, CDC hay AWS DMS.

```text
Supabase Transaction Pooler :6543
       → Python notebook (batch SELECT)
       → CSV snapshot per table
       → S3 raw/supabase-pooler-export/run_id=<UTC timestamp>/
       → Airflow: ingest → validate → features → train → MLflow publish
```

## Giới hạn quan trọng

- Đây là **snapshot export**, không phải CDC. Mỗi lần chạy tạo một run mới trên S3.
- Transaction Pooler phù hợp với các query batch ngắn. Pipeline mở một kết nối qua pooler cho mỗi bảng và dùng `autocommit=True`.
- Không có database-wide consistent snapshot giữa nhiều bảng. Khi cần snapshot nghiêm ngặt, dừng writes trong lúc export hoặc chuyển sang direct connection/CDC architecture.
- `SUPABASE_TRANSACTION_POOLER_URL` phải dùng port `6543`. Lấy URL tại Supabase Dashboard → **Database → Database Access → Transaction pooler**. Với shared pooler và custom role, username thường có dạng `<role>.<project-ref>`.

## Cấu trúc

```text
notebooks/supabase_pooler_to_s3.ipynb  # entry point để chạy
src/export_pipeline.py                 # export code có thể test
tests/test_export_pipeline.py          # unit tests không cần AWS/Supabase
.env.example                           # biến môi trường mẫu
docker-compose.yml                     # Airflow + PostgreSQL + MLflow local
dags/supabase_s3_training_pipeline.py  # DAG S3-to-training
app/pipeline.py                         # ingest, validate, feature, train stages
```

## 1. Chuẩn bị Supabase và S3

User trong URL pooler cần quyền `SELECT` trên các bảng export. S3 IAM identity chạy notebook cần quyền tối thiểu:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:AbortMultipartUpload"],
      "Resource": "arn:aws:s3:::<BUCKET>/raw/supabase-pooler-export/*"
    }
  ]
}
```

Khi chạy trên laptop, chọn **một** trong hai cách sau:

1. Khuyến nghị: cấu hình AWS CLI profile (`aws configure` hoặc AWS SSO). Boto3 tự đọc profile mặc định; không cần thêm key vào `.env`.
2. Dùng credentials trong `.env` (file này đã bị `.gitignore`): điền `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` và, nếu dùng AWS SSO/STS, `AWS_SESSION_TOKEN`.

Không đưa access key vào notebook hoặc commit file `.env`. Khi chạy trên AWS (EC2, ECS, MWAA...), gắn IAM role vào workload thay vì tạo access key.

## 2. Cấu hình và chạy notebook

```bash
cd /Users/vietannguyen/TechX_projects/giai_doan_6_new/supabase-pooler-to-s3-pipeline
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-dev.txt
cp .env.example .env
```

Sửa `.env`:

- Vào **Supabase Dashboard → Database → Database Access → Transaction pooler**, chọn/copy connection string dạng URI rồi dán vào `SUPABASE_TRANSACTION_POOLER_URL`.
- Đặt `S3_EXPORT_BUCKET` và tùy chọn đổi `S3_EXPORT_PREFIX`.
- Đặt `EXPORT_TABLES_JSON`; mỗi bảng phải có `schema`, `table`, `primary_key` hợp lệ.

Sau đó mở notebook:

```bash
jupyter lab notebooks/supabase_pooler_to_s3.ipynb
```

Chọn **Run All**. Kết quả trả về có số dòng, URI CSV, SHA-256 của từng bảng và URI `manifest.json`.

Ví dụ output:

```text
s3://<bucket>/raw/supabase-pooler-export/run_id=20260930T010203Z/public/employment/data.csv
s3://<bucket>/raw/supabase-pooler-export/run_id=20260930T010203Z/manifest.json
```

## 3. Kiểm tra

```bash
python3 -m pytest -q
aws s3 ls s3://<bucket>/raw/supabase-pooler-export/ --recursive
```

Mỗi manifest ghi run ID, thời điểm export, row count, URI và checksum CSV. Chỉ sử dụng downstream training sau khi chọn cụ thể một `run_id`, không đọc lẫn CSV giữa các run.

## 4. Airflow: ingest từ S3 và train model

Sau khi notebook đã export thành công tối thiểu một run lên S3, Airflow sẽ dùng đúng sáu CSV sau trong **cùng một** `run_id`:

```text
public/enrollees/data.csv
public/enrollee_education/data.csv
public/work_experience/data.csv
public/training_hours/data.csv
public/employment/data.csv
public/city_dev_index/data.csv
```

Trong `.env`, giữ nguyên `S3_EXPORT_BUCKET`, `S3_EXPORT_PREFIX` và AWS credentials đã dùng cho notebook. Thêm tùy chọn:

```env
# Để trống: chọn run_id mới nhất. Khuyến nghị khi training production: pin một run cụ thể.
S3_TRAINING_RUN_ID=20260930T010203Z
MIN_ROC_AUC=0.70
```

Khởi động stack từ thư mục project:

```bash
docker compose up --build -d
```

Mở Airflow tại `http://localhost:8083` bằng `admin` / `admin`, bật và Trigger DAG `supabase_s3_employment_training`. MLflow ở `http://localhost:5002`.

DAG chạy tuần tự:

```text
ingest_selected_s3_snapshot
  → validate_source_data
  → prepare_feature_dataset
  → create_train_test_split
  → train_and_track_mlflow
  → quality_gate_and_publish
```

Task ingest download CSV về `artifacts/raw/run_id=<id>/` và ghi `artifacts/source_run.json`. Các task sau chỉ đọc run đó. Model chỉ được publish khi ROC-AUC đạt `MIN_ROC_AUC`; manifest phát hành nằm tại `artifacts/production/model_manifest.json`.

Để dừng stack khi xong:

```bash
docker compose down
```

### Nếu Airflow không đọc được task log (HTTP 403)

Scheduler tạo URL log ký số, còn webserver xác minh URL đó. Vì vậy mọi Airflow container phải dùng cùng `AIRFLOW__WEBSERVER__SECRET_KEY`. Project đã cấu hình một local development key dùng chung trong `docker-compose.yml`. Nếu bạn tự đặt key trong `.env`, nó phải là **một giá trị duy nhất** cho webserver và scheduler.

Sau khi cập nhật project, recreate các container Airflow để chúng nhận cùng cấu hình:

```bash
docker compose up -d --force-recreate airflow-webserver airflow-scheduler airflow-init
```

Sau đó trigger một DAG run mới. Lỗi 403 chỉ chặn giao diện đọc log; nó không tự nói rằng `validate_source_data` đã fail. Xem log runtime nếu task mới fail:

```bash
docker compose logs --tail=200 airflow-scheduler
```
