import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Paksa MLflow pakai folder lokal 'mlruns' di project
mlruns_path = os.path.abspath("./mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_path}")

# Pastikan folder artifact ada
artifact_dir = os.path.join(mlruns_path, "artifacts")
os.makedirs(artifact_dir, exist_ok=True)

# Load dataset dari CSV (versi DVC)
df = pd.read_csv("data/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Aktifkan autolog MLflow
mlflow.sklearn.autolog()

with mlflow.start_run():
    # Inisialisasi model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung berbagai metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred).tolist()  # convert ke list biar bisa JSON

    # Log metrik manual
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Log confusion matrix sebagai artifact JSON
    cm_path = os.path.join(artifact_dir, "confusion_matrix.json")
    mlflow.log_dict({"confusion_matrix": cm}, cm_path)

    # Print metrik ke console
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion matrix saved to:", cm_path)
