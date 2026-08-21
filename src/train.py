import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

F1_THRESHOLD = 0.65
REFERENCE_POSITIVE_RATIO = 0.248
LABEL_MAP = {0: "thu_nhap_thap", 1: "thu_nhap_cao"}


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():
        mlflow.log_params(params)
        model = GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_eval)
        f1 = f1_score(y_eval, preds)
        acc = accuracy_score(y_eval, preds)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        prec_pos = precision_score(y_eval, preds, pos_label=1, zero_division=0)
        rec_pos = recall_score(y_eval, preds, pos_label=1, zero_division=0)
        prec_neg = precision_score(y_eval, preds, pos_label=0, zero_division=0)
        rec_neg = recall_score(y_eval, preds, pos_label=0, zero_division=0)
        cm = confusion_matrix(y_eval, preds)
        detail_lines = []
        detail_lines.append("=== Detail Report (Adult Income) ===")
        detail_lines.append(f"F1 (lop duong): {f1:.4f} | Accuracy: {acc:.4f}")
        detail_lines.append(f"Precision (thu_nhap_cao): {prec_pos:.4f} | Recall (thu_nhap_cao): {rec_pos:.4f}")
        detail_lines.append(f"Precision (thu_nhap_thap): {prec_neg:.4f} | Recall (thu_nhap_thap): {rec_neg:.4f}")
        detail_lines.append("")
        detail_lines.append("Confusion Matrix (rows=true 0,1 / cols=pred 0,1):")
        detail_lines.append(str(cm))
        detail_text = "\n".join(detail_lines)
        print(detail_text)

        proba = model.predict_proba(X_eval)[:, 1]
        best_t, best_t_f1 = 0.5, f1
        for t in np.arange(0.1, 0.91, 0.05):
            t = round(float(t), 2)
            preds_t = (proba >= t).astype(int)
            f1_t = f1_score(y_eval, preds_t)
            if f1_t > best_t_f1:
                best_t, best_t_f1 = t, f1_t
        mlflow.log_metric("best_threshold_f1", best_t_f1)
        print(f"Best threshold: {best_t:.2f} -> F1 {best_t_f1:.4f}")

        pos_ratio = float(np.mean(y_train == 1))
        drift = abs(pos_ratio - REFERENCE_POSITIVE_RATIO)
        if drift > 0.05:
            print(f"[DRIFT WARNING] Ty le lop duong {pos_ratio:.3f} lech {drift*100:.2f}pp so voi tham chieu 24.8%")
        mlflow.log_metric("positive_ratio", pos_ratio)

        os.makedirs("outputs", exist_ok=True)
        report = {
            "f1_score": f1,
            "accuracy": acc,
            "best_threshold": best_t,
            "best_threshold_f1": best_t_f1,
            "precision_positive": prec_pos,
            "recall_positive": rec_pos,
            "positive_ratio": pos_ratio,
        }
        with open("outputs/report.json", "w") as f:
            json.dump(report, f, indent=2)

        with open("outputs/detail.txt", "w") as f:
            f.write(detail_text)
        mlflow.log_artifact("outputs/detail.txt")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
