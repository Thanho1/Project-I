import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle

# 1. Đọc dữ liệu
print("Đang load dữ liệu...")

data = pd.read_csv("data.csv")

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
    kernel='rbf',
    C=10,          
    gamma='scale',
    probability=True  
)

# 5. Train model
print("Đang train model...")
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Đánh giá
print("\n KẾT QUẢ ĐÁNH GIÁ:\n")

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# 8. Lưu model
with open("svm_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nĐã lưu model vào svm_model.pkl và scaler.pkl")