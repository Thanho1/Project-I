"""
Train model SVM để phân loại hành động bóng rổ (dribbling, shooting,
defense, idle) dựa trên pose keypoints trong data/data.csv.

Sau khi train, lưu model và scaler vào thư mục models/.
"""

import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.utils.path_utils import get_base_dir

BASE_DIR = get_base_dir(__file__)

DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")


def main():
    # 1. Đọc dữ liệu
    print("Đang load dữ liệu...")
    data = pd.read_csv(DATA_PATH)

    X = data.drop("label", axis=1)
    y = data["label"]

    print("Số mẫu:", len(X))

    # 2. Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Chia train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Train:", len(X_train))
    print("Test:", len(X_test))

    # 4. Tạo model SVM
    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True
    )

    # 5. Train model
    print("Đang train model...")
    model.fit(X_train, y_train)

    # 6. Predict
    y_pred = model.predict(X_test)

    # 7. Đánh giá
    print("\nKẾT QUẢ ĐÁNH GIÁ:\n")

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # 8. Lưu model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nĐã lưu model vào {MODEL_PATH} và {SCALER_PATH}")


if __name__ == "__main__":
    main()
