# Pic2Word — Zero-shot Composed Image Retrieval

Dự án nghiệm thu lại phương pháp trong bài báo **Pic2Word: Mapping Pictures to Words
for Zero-shot Composed Image Retrieval**. Mô hình nhận một ảnh tham chiếu và một câu
mô tả thay đổi, sau đó tìm ảnh phù hợp nhất trong một kho ảnh ứng viên.

> Pic2Word là mô hình **tìm kiếm ảnh**, không phải mô hình sinh hoặc chỉnh sửa ảnh.

## Dành cho giảng viên

Repo đã kèm source code, unit test, báo cáo thực nghiệm, log loss và checkpoint Mapping
Network local 500 bước. Repo không kèm ảnh CC3M/CIRR, CLIP ViT-L/14 và candidate index
vì các tệp này lớn hoặc có điều kiện phân phối riêng.

Có ba mức kiểm tra độc lập:

1. **Kiểm tra code:** cài môi trường và chạy `pytest`; không cần dataset.
2. **Demo ảnh tự chọn:** tải CLIP, thêm một ảnh truy vấn và một thư mục ảnh ứng viên.
3. **Tính lại CIRR Recall:** cần chép dataset CIRR vào đúng cấu trúc ở mục 4.3.

## 1. Chạy nhanh để nghiệm thu

Các lệnh dưới đây dành cho Windows PowerShell. Tải repo và cài môi trường:

```powershell
git clone https://github.com/huylenhat1701-stack/Pic2word.git
cd Pic2word
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

### 1.1. Kiểm tra môi trường và GPU

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu máy có GPU NVIDIA và PyTorch CUDA phù hợp, kết quả sẽ có `CUDA: True`. Nếu không,
thay `--device cuda` trong các lệnh bên dưới bằng `--device cpu`.

### 1.2. Chạy bộ kiểm thử code

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Kết quả hiện tại của dự án là `18 passed`.

### 1.3. Tải CLIP cho demo hoặc đánh giá

```powershell
.\.venv\Scripts\python.exe scripts\download_clip.py --device auto
```

### 1.4. Kiểm tra kết quả thực nghiệm đã lưu

Không cần dataset để đọc hai tệp bằng chứng đã commit:

- `reports/pic2word_local_acceptance.md`: báo cáo nghiệm thu và bảng Recall.
- `logs/local_final/pic2word/training_metrics.csv`: loss của toàn bộ quá trình train.

### 1.5. Tính lại Recall bằng checkpoint đã train

Không cần train lại. Lệnh sau nạp checkpoint 500 bước và tính Recall trên phần dữ liệu
CIRR hiện có:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_cirr.py `
  --allow-partial `
  --device cuda `
  --checkpoint checkpoints\local_final\pic2word\last.pt `
  --index indexes\cirr_val_partial.pt `
  --index-batch-size 1 `
  --output reports\cirr_val_metrics_local_final_partial.json
```

Lệnh này chỉ chạy sau khi đã đặt annotation và ảnh CIRR vào `data/cirr` theo mục 4.3.
Dataset không nằm trong GitHub. Nếu chưa có dataset, cô vẫn có thể kiểm tra code bằng
`pytest`, đọc báo cáo đã lưu và chạy demo ảnh tự chọn ở mục 2.

Kết quả tham chiếu của lần chạy đã hoàn thành:

| Chỉ số | Model local 500 bước |
|---|---:|
| Recall@1 | 33,33% |
| Recall@5 | 69,44% |
| Recall@10 | 83,33% |
| Recall@50 | 86,11% |
| Group Recall@1 | 66,67% |
| Group Recall@2 | 77,78% |
| Group Recall@3 | 83,33% |

Tạo lại báo cáo nghiệm thu sau khi đánh giá:

```powershell
.\.venv\Scripts\python.exe scripts\build_acceptance_report.py
```

Báo cáo được lưu tại `reports/pic2word_local_acceptance.md`.

## 2. Thử mô hình bằng ảnh tự chọn

### 2.1. Chuẩn bị ảnh

Tạo cấu trúc sau bằng File Explorer hoặc VS Code:

```text
data/demo/
|-- query.jpg
`-- candidates/
    |-- target.jpg
    |-- image_01.jpg
    |-- image_02.jpg
    `-- ...
```

- `query.jpg`: ảnh tham chiếu ban đầu.
- `candidates`: kho ảnh để mô hình tìm kiếm.
- `target.jpg`: ảnh được kỳ vọng phù hợp với câu mô tả thay đổi.
- Không đặt `query.jpg` bên trong thư mục `candidates`.
- Hỗ trợ các định dạng `.jpg`, `.jpeg`, `.png` và `.webp`.

Ví dụ: `query.jpg` là giày trắng, `target.jpg` là giày đỏ và câu thay đổi là
`make the shoes red`.

### 2.2. Chạy tìm kiếm Top-5

```powershell
.\.venv\Scripts\python.exe scripts\retrieve.py `
  --query-image data\demo\query.jpg `
  --text "make the shoes red" `
  --candidate-dir data\demo\candidates `
  --checkpoint checkpoints\local_final\pic2word\last.pt `
  --config configs\train_final_local.yaml `
  --index indexes\demo_candidates.pt `
  --top-k 5 `
  --max-candidates 1000 `
  --index-batch-size 1 `
  --rebuild-index `
  --device cuda
```

Chương trình in ra đường dẫn và cosine similarity score của năm ảnh phù hợp nhất.
Nếu `target.jpg` đứng thứ nhất thì truy vấn đạt Recall@1; nếu nằm trong năm ảnh đầu thì
đạt Recall@5.

Chỉ dùng `--rebuild-index` ở lần chạy đầu hoặc khi thêm, xoá, thay đổi ảnh trong thư mục
`candidates`. Nếu chỉ đổi `query.jpg` hoặc câu `--text`, có thể bỏ tham số này để dùng lại
index đã tạo.

## 3. Train lại Mapping Network

### 3.1. Smoke test 10 bước

Cấu hình này dùng để xác minh nhanh rằng forward, loss, backward và checkpoint đều chạy:

```powershell
.\.venv\Scripts\python.exe scripts\train.py `
  --config configs\train_local.yaml `
  --device cuda
```

Checkpoint smoke test nằm tại `checkpoints/pic2word/last.pt`.

### 3.2. Train local đủ 500 bước

Lệnh sau train từ đầu trên toàn bộ 1.000 ảnh CC3M đã chuẩn bị. Nó sẽ ghi lại checkpoint
và log của lần chạy cũ trong thư mục `local_final`:

```powershell
.\.venv\Scripts\python.exe scripts\train.py `
  --config configs\train_final_local.yaml `
  --device cuda
```

Trên RTX 3050 Laptop 4 GB, lần chạy đã đo mất khoảng 200 giây. Đầu ra:

```text
checkpoints/local_final/pic2word/last.pt
logs/local_final/pic2word/training_metrics.csv
logs/local_final/pic2word/training_summary.json
```

Trong lần nghiệm thu hiện tại, loss trung bình 50 dòng đầu là `1,280234`, còn 50 dòng
cuối là `0,549226`, giảm `57,10%`.

## 4. Chuẩn bị dữ liệu

### 4.1. Cài gói dữ liệu ZIP từ Google Drive

**Link dữ liệu Google Drive:** sinh viên điền liên kết chia sẻ tại đây trước khi nộp.

Tệp chia sẻ nên được đặt tên là `pic2word_data.zip` và phải chứa nguyên thư mục
`data`. Sau khi giải nén, cấu trúc tối thiểu để train phải giống như sau:

```text
Pic2word/
|-- data/
|   |-- cc/
|   |   `-- Train_GCC-training_output.csv
|   `-- cc_data/
|       `-- train/
|           |-- 000000000.jpg
|           |-- 000000001.jpg
|           `-- ...
|-- configs/
|-- scripts/
`-- README.md
```



1. Trên Google Drive, tải tệp `pic2word_data.zip` về thư mục `Downloads`.
2. Mở Windows PowerShell tại thư mục gốc `Pic2word` (nơi có `README.md`).
3. Giải nén gói dữ liệu trực tiếp vào thư mục dự án:

```powershell
$zip = "$env:USERPROFILE\Downloads\pic2word_data.zip"
Expand-Archive -LiteralPath $zip -DestinationPath . -Force
```

4. Kiểm tra đường dẫn và số lượng dữ liệu:

```powershell
Test-Path data\cc\Train_GCC-training_output.csv
(Get-ChildItem data\cc_data\train -File | Where-Object Extension -In '.jpg','.jpeg','.png','.webp' | Measure-Object).Count
(Import-Csv data\cc\Train_GCC-training_output.csv | Measure-Object).Count
```

Với gói dữ liệu dùng trong báo cáo, ba kết quả dự kiến lần lượt là `True`, `1000` và
`1000`. Nếu xuất hiện đường dẫn `data\data\cc_data`, ZIP đã bị giải nén thừa một tầng;
cần đặt thư mục `data` cùng cấp với `configs`, `scripts` và `README.md` như cây thư mục
phía trên.

5. Bắt đầu train:

```powershell
.\.venv\Scripts\python.exe scripts\train.py `
  --config configs\train_final_local.yaml `
  --device cuda
```

Sau khi train xong, kiểm tra hai kết quả chính:

```text
checkpoints/local_final/pic2word/last.pt
logs/local_final/pic2word/training_metrics.csv
```

Nếu máy không có GPU NVIDIA, đổi `--device cuda` thành `--device cpu`; quá trình sẽ
chậm hơn. Tệp CIRR không bắt buộc để train: CIRR chỉ cần khi tính Recall ở mục 4.3.

### 4.2. Dữ liệu CC3M dùng để train

Pic2Word chỉ cần ảnh CC3M để train Mapping Network; caption và URL không được đưa vào
loss. Manifest phải chứa một cột đường dẫn ảnh, ví dụ:

```csv
image
000000000.jpg
000000001.jpg
```

Các đường dẫn hiện dùng:

```text
data/cc/Train_GCC-training_output.csv
data/cc_data/train/*.jpg
```

Nếu có một shard WebDataset, chuẩn bị tối đa 1.000 ảnh bằng:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_cc3m_shard.py `
  "C:\duong-dan\toi\cc3m-train-0000.tar" `
  --max-images 1000
```

Script giữ nguyên TAR gốc, giải nén ảnh hợp lệ và tạo lại manifest.

### 4.3. Dữ liệu CIRR dùng để đánh giá

Cấu trúc yêu cầu:

```text
data/cirr/
|-- captions/cap.rc2.val.json
|-- image_splits/split.rc2.val.json
`-- img_raw/dev/...
```

Nếu có ảnh NLVR2 dạng ZIP hoặc thư mục đã giải nén:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_cirr_images.py `
  "C:\duong-dan\toi\nlvr2.zip"
```

Có thể thử lấy những ảnh còn truy cập được từ danh sách URL công khai:

```powershell
.\.venv\Scripts\python.exe scripts\download_cirr_from_nlvr2_urls.py
```

Dữ liệu hiện tại có 980/2.297 ảnh và chỉ 36/4.181 truy vấn có đủ nhóm ảnh. Vì vậy báo
cáo ghi `partial: true` và `officially_comparable: false`. Kết quả này chứng minh pipeline
chạy được từ đầu đến cuối, nhưng không được trình bày như kết quả tái lập chính thức của
bài báo.

Khi có đủ 2.297 ảnh, chạy nghiệm thu đầy đủ mà không dùng `--allow-partial`:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_cirr.py `
  --device cuda `
  --checkpoint checkpoints\local_final\pic2word\last.pt `
  --index indexes\cirr_val_full_local.pt `
  --index-batch-size 1 `
  --rebuild-index `
  --output reports\cirr_val_metrics_local_final.json
```

## 5. Nội dung có và không có trên GitHub

| Thành phần | Có trong repo? | Ghi chú |
|---|---:|---|
| Source, cấu hình và unit test | Có | Chạy được ngay sau khi cài dependencies |
| Mapping Network local 500 bước | Có | `checkpoints/local_final/pic2word/last.pt` |
| Báo cáo Recall và log loss | Có | Dùng để kiểm tra kết quả thực nghiệm đã chạy |
| OpenAI CLIP ViT-L/14 | Không | Tải bằng `scripts/download_clip.py` |
| Ảnh CC3M và CIRR | Không | Dữ liệu lớn, đặt thủ công trong `data/` |
| Candidate index | Không | Tự sinh ở lần retrieval/evaluation đầu tiên |

Python hỗ trợ: 3.11 hoặc 3.12. Nếu muốn dùng GPU NVIDIA, cài PyTorch CUDA tương thích
với driver của máy. CPU vẫn chạy được nhưng chậm hơn.

## 6. Kiến trúc và phạm vi

Luồng xử lý chính:

```text
Ảnh tham chiếu
      |
      v
CLIP Image Encoder (đóng băng)
      |
      v
Mapping Network: 768 -> 512 -> 512 -> 768
      |
      v
Pseudo-word token + câu mô tả thay đổi
      |
      v
CLIP Text Encoder (đóng băng)
      |
      v
Cosine similarity với kho ảnh -> Top-K
```

Chỉ Mapping Network được cập nhật bằng symmetric contrastive loss. CLIP Image Encoder
và Text Encoder luôn đóng băng.

Các file quan trọng:

| File/thư mục | Vai trò |
|---|---|
| `src/pic2word/models/` | CLIP backbone, Mapping Network và mô hình Pic2Word |
| `src/pic2word/training/` | Contrastive loss và training loop |
| `src/pic2word/retrieval/` | Candidate index và tìm kiếm cosine Top-K |
| `src/pic2word/evaluation/` | Recall@K và Group Recall@K cho CIRR |
| `scripts/train.py` | Train và lưu checkpoint/log |
| `scripts/retrieve.py` | Thử ảnh tự chọn |
| `scripts/evaluate_cirr.py` | Đánh giá định lượng trên CIRR |
| `scripts/build_acceptance_report.py` | Tạo báo cáo nghiệm thu Markdown |
| `configs/train_final_local.yaml` | Cấu hình train phù hợp GPU 4 GB |
| `reports/pic2word_local_acceptance.md` | Báo cáo kết quả gần nhất |

Checkpoint dùng để chạy demo là `checkpoints/local_final/pic2word/last.pt`. Checkpoint
`checkpoints/pic2word/last.pt` chỉ là smoke test 10 bước; checkpoint
`checkpoints/pic2word/pic2word_official_converted.pt` là baseline Pic2Word epoch 30 đã
được chuyển đổi an toàn về đúng sáu tensor của MLP.
