# Basketball Action Recognition (Project I)

Hệ thống nhận diện hành động bóng rổ theo thời gian thực (dribbling, shooting,
defense, idle) từ webcam, sử dụng MediaPipe Pose để trích xuất keypoints và
SVM để phân loại hành động.

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
│   └── data.csv          # bảng pose keypoints + label
├── models/
│   ├── svm_model.pkl
│   └── scaler.pkl
├── docs/
│   └── demo/             # video/gif demo
├── notebooks/
│   └── eda_and_evaluation.ipynb
├── src/
│   ├── data_collection/  # cut_frame.py, auto_dataset.py
│   ├── features/         # extract_keypoints.py
│   ├── training/          # train_svm.py
│   ├── inference/          # app.py, demo_webcam.py
│   └── utils/             # hàm dùng chung
└── tests/
```

## Cài đặt

```bash
pip install -r requirements.txt
```

### Git LFS (cho video raw)

Video gốc trong `data/raw_videos/` có dung lượng lớn, được quản lý bằng
[Git LFS](https://git-lfs.com/) thay vì commit trực tiếp. Để clone đầy đủ
project (bao gồm video):

```bash
git lfs install
git clone https://github.com/<username>/Project-I.git
```

Nếu chỉ cần code (không cần video gốc), clone bình thường — Git LFS sẽ
chỉ tải các file `.pkl` model và bỏ qua/placeholder các file video.

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

Dataset: 2475 mẫu (sau khi trích keypoints, bỏ qua 205 ảnh không phát hiện
được pose), chia train/test theo tỷ lệ 80/20 (1980 / 495 mẫu).

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

## Hạn chế & hướng phát triển

- Model SVM dựa trên pose tĩnh từng frame, chưa khai thác thông tin theo
  thời gian (sequence) -> có thể cải thiện bằng LSTM/Transformer trên
  sequence pose.
- Dataset còn nhỏ, cần thu thập thêm nhiều người chơi/góc camera khác nhau
  để tăng khả năng tổng quát hoá.

## License

[MIT](LICENSE)
