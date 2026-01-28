import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    }

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
