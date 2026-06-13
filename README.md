# Basketball Action Recognition (Project I)

  Hệ thống nhận diện hành động bóng rổ theo thời gian thực (dribbling, shooting, defense, idle) từ webcam, sử dụng MediaPipe Pose để trích xuất keypoints và SVM để phân loại hành động.

## Demo

![demo](docs/demo/demo.gif)

*(Video demo đầy đủ: `docs/demo/`)*

## Pipeline

```
Video gốc (data/raw_videos/)
        │
        ▼  src/data_collection/auto_dataset.py
Ảnh frame theo từng class (data/dataset/)
        │
        ▼  src/features/extract_keypoints.py
Pose keypoints -> data/data.csv
        │
        ▼  src/training/train_svm.py
Model SVM + Scaler (models/)
        │
        ▼  src/inference/app.py | demo_webcam.py
Nhận diện hành động theo thời gian thực
```

## Tech stack

- **Python**
- **OpenCV** - xử lý video/ảnh
- **MediaPipe Pose** - trích xuất 33 pose landmarks (x, y, z, visibility)
- **scikit-learn (SVM)** - phân loại hành động
- **Tkinter** - GUI desktop (app.py)

## Cấu trúc thư mục

```
Project-I/
├── data/
│   ├── raw_videos/      # video gốc (không push lên git)
│   ├── dataset/         # ảnh đã cắt frame theo class
│   ├── data.csv          # v1: 132 features (position) + label
│   └── data_v2.csv        # v2: 264 features (position + velocity) + label
├── models/
│   ├── svm_model.pkl     / scaler.pkl       # v1
│   └── svm_model_v2.pkl  / scaler_v2.pkl    # v2
├── docs/
│   └── demo/             # video/gif demo
├── notebooks/
│   └── eda_and_evaluation.ipynb
├── src/
│   ├── data_collection/  # cut_frame.py, auto_dataset.py
│   ├── features/         # extract_keypoints.py, extract_keypoints_v2.py
│   ├── training/          # train_svm.py, train_svm_v2.py
│   ├── inference/          # app.py, demo_webcam.py, demo_webcam_v2.py
│   └── utils/             # hàm dùng chung
└── tests/
```

## Chạy notebook EDA & Evaluation

Notebook `notebooks/eda_and_evaluation.ipynb` dùng để phân tích dữ liệu và đánh giá model bằng biểu đồ trực quan (confusion matrix, so sánh accuracy giữa các model/phiên bản).

**Yêu cầu:**
+ Đã có `data/data.csv` (và `data/data_v2.csv` nếu muốn so sánh v1/v2)
+ Cài thêm `ipykernel` (để chạy notebook trong VSCode/Jupyter):
```bash
pip install ipykernel jupyter
```
+ Cài đủ thư viện: pandas, numpy, matplotlib, seaborn, scikit-learn
```bash
pip install pandas numpy matplotlib seaborn scikit-learn --break-system-packages
```

**Cách chạy (VSCode):**
1. Mở file `notebooks/eda_and_evaluation.ipynb`
2. Chọn kernel: bấm vào tên Python ở góc trên phải -> chọn đúng môi trường Python đã cài `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

3. Bấm **Run All** để chạy toàn bộ notebook

**Cách chạy (Jupyter Notebook qua terminal):**
```bash
jupyter notebook notebooks/eda_and_evaluation.ipynb
```
Sau đó vào menu **Run -> Run All Cells**.

Kết quả mong đợi: các cell hiển thị số liệu (accuracy, classification report) và biểu đồ (phân phối class, confusion matrix, so sánh accuracy giữa các model) ngay dưới mỗi cell, không có lỗi màu đỏ.

## Cài đặt

```bash
pip install -r requirements.txt
```

### Git LFS (cho video raw)

Video gốc trong `data/raw_videos/` có dung lượng lớn, được quản lý bằng [Git LFS](https://git-lfs.com/) thay vì commit trực tiếp. Để clone đầy đủ
project (bao gồm video):

```bash
git lfs install
git clone https://github.com/<username>/Project-I.git
```

Nếu chỉ cần code (không cần video gốc), clone bình thường — Git LFS sẽ chỉ tải các file `.pkl` model và bỏ qua/placeholder các file video.

## Sử dụng

**1. Cắt frame từ video để tạo dataset**
```bash
python -m src.data_collection.auto_dataset
```

**2. Trích xuất pose keypoints thành CSV**
```bash
python -m src.features.extract_keypoints
```

**3. Train model SVM**
```bash
python -m src.training.train_svm
```

**4. Chạy demo nhận diện qua webcam**
```bash
# Phiên bản OpenCV đơn giản
python -m src.inference.demo_webcam

# Phiên bản GUI đầy đủ (Tkinter, có record video)
python -m src.inference.app
```

## Kết quả

Dataset: 2475 mẫu (sau khi trích keypoints, bỏ qua 205 ảnh không phát hiện được pose), chia train/test theo tỷ lệ 80/20 (1980 / 495 mẫu).

| Metric | Giá trị |
|---|---|
| Accuracy | **0.986** |
| Số class | 4 (dribbling, shooting, defense, idle) |

**Classification Report:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| defense | 0.99 | 0.99 | 0.99 | 141 |
| dribbling | 0.99 | 0.97 | 0.98 | 129 |
| idle | 1.00 | 0.99 | 1.00 | 109 |
| shooting | 0.96 | 1.00 | 0.98 | 116 |

**Confusion Matrix:**

```
              defense  dribbling  idle  shooting
defense           139          1     0         1
dribbling           1        125     0         3
idle                0          0   108         1
shooting            0          0     0       116
```

Chi tiết và biểu đồ trực quan: xem `notebooks/eda_and_evaluation.ipynb`.

## Cải tiến: Position + Velocity Features (v2)

Phiên bản v1 chỉ dùng vị trí tĩnh (x, y, z, visibility) của 33 landmarks trong từng frame riêng lẻ (132 features), nên model không "nhìn thấy được tốc độ/hướng chuyển động — điều giúp phân biệt rõ các động tác như `dribbling` (tay di chuyển nhanh, lặp lại) và `defense` (tư thế tĩnh hơn).

**Cải tiến v2:** thêm **velocity** = chênh lệch pose giữa frame hiện tại và frame trước đó (cùng video), nâng tổng số feature từ 132 lên **264** (132 position + 132 velocity).

### Pipeline v2

```
data/dataset/                                  (ảnh đã có từ v1)
        │
        ▼  src/features/extract_keypoints_v2.py
data/data_v2.csv  (264 features + label)
        │
        ▼  src/training/train_svm_v2.py
models/svm_model_v2.pkl, models/scaler_v2.pkl
        │
        ▼  src/inference/demo_webcam_v2.py
```

### So sánh v1 vs v2

| Phiên bản | Features | Accuracy |
|---|---|---|
| v1 (position only) | 132 | **0.986** |
| v2 (position + velocity) | 264 | 0.944 |

**Nhận xét:** v2 cho kết quả thấp hơn v1. Nguyên nhân có thể do:
- Tăng gấp đôi số chiều đặc trưng (132 -> 264) trong khi dữ liệu không tăng, khiến model dễ bị ảnh hưởng bởi nhiễu hơn (đặc biệt với SVM-RBF).
- Các ảnh trong `data/dataset/` được lấy mẫu không đều theo thời gian (chỉ giữ frame có pose thay đổi đủ lớn so với frame trước, theo `auto_dataset.py`), nên "velocity" tính giữa 2 ảnh liên tiếp trong dataset không phản ánh đúng vận tốc chuyển động thực tế trong video — trở thành nhiễu hơn là tín hiệu hữu ích.

**Kết luận:** model v1 (position only) vẫn được dùng làm bản chính. 
Thí nghiệm v2 cho thấy hướng thêm temporal information là hợp lý về lý thuyết, nhưng cần lấy mẫu dataset theo *frame liên tiếp thực sự* (không qua bước lọc threshold) để velocity có ý nghĩa — đây là hướng cải tiến tiếp theo (xem mục Hạn chế & hướng phát triển).

### Chạy v2

```bash
python -m src.features.extract_keypoints_v2
python -m src.training.train_svm_v2
python -m src.inference.demo_webcam_v2
```

## Hạn chế & hướng phát triển

- Thí nghiệm v2 (position + velocity) cho thấy việc thêm thông tin thời gian là cần thiết nhưng cách lấy mẫu velocity hiện tại chưa phù hợp (do dataset lấy theo threshold thay đổi pose, không phải frame liên tiếp). Hướng tiếp theo: trích keypoints từ video gốc theo frame liên tiếp thực sự (fps cố định), rồi thử lại velocity hoặc sequence model(LSTM/Transformer).
- Dataset còn nhỏ, cần thu thập thêm nhiều người chơi/góc camera khác nhau để tăng khả năng tổng quát hoá.
- Hiện chỉ xử lý 1 người trong frame; có thể mở rộng multi-person tracking.

## License

[MIT](LICENSE)