import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import uuid


def save_confusion_matrix(y_true, y_pred, save_path):
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()


def save_roc_curve(y_true, y_proba, save_path):
    from sklearn.preprocessing import label_binarize
    import numpy as np

    # Support binary and multiclass
    if len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(save_path)
        plt.close()
    else:
        # Multiclass ROC: plot only for first class
        y_true_bin = label_binarize(y_true, classes=list(set(y_true)))
        fpr, tpr, _ = roc_curve(y_true_bin[:, 0], y_proba[:, 0])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (class 0)")
        plt.legend()
        plt.savefig(save_path)
        plt.close()


def save_feature_importance(model, feature_names, save_path):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(
            range(len(indices)), [feature_names[i] for i in indices], rotation=45
        )
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def train_model(df, params):
    target_column = params.get("target_column")
    task_type = params.get("task_type")
    model_name = params.get("model_name")
    test_size = params.get("test_size", 0.2)
    loss_function = params.get("loss_function")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    if task_type == "classification":
        if model_name == "LogisticRegression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "RandomForest":
            model = RandomForestClassifier()
        elif model_name == "DecisionTree":
            model = DecisionTreeClassifier()
        else:
            return {"error": "Invalid model for classification"}
    elif task_type == "regression":
        if model_name == "LinearRegression":
            model = LinearRegression()
        elif model_name == "RandomForest":
            model = RandomForestRegressor()
        elif model_name == "DecisionTree":
            model = DecisionTreeRegressor()
        else:
            return {"error": "Invalid model for regression"}
    else:
        return {"error": "Invalid task type"}
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = {}
    # Supported metrics
    classification_metrics = {
        "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
        "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
        "precision": lambda y_true, y_pred: precision_score(
            y_true, y_pred, average="weighted"
        ),
        "recall": lambda y_true, y_pred: recall_score(
            y_true, y_pred, average="weighted"
        ),
    }
    regression_metrics = {
        "mse": lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
        "rmse": lambda y_true, y_pred: mean_squared_error(
            y_true, y_pred, squared=False
        ),
        "mae": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
        "r2": lambda y_true, y_pred: r2_score(y_true, y_pred),
    }
    eval_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "outputs",
        "evaluation",
    )
    os.makedirs(eval_dir, exist_ok=True)
    if task_type == "classification":
        result["accuracy"] = accuracy_score(y_test, y_pred)
        result["report"] = classification_report(y_test, y_pred, output_dict=True)
        # Confusion Matrix
        cm_path = os.path.join(eval_dir, "confusion_matrix.png")
        save_confusion_matrix(y_test, y_pred, cm_path)
        result["confusion_matrix"] = "/outputs/evaluation/confusion_matrix.png"
        # ROC Curve
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
                if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                    y_proba = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba
                save_roc_curve(y_test, y_proba, os.path.join(eval_dir, "roc_curve.png"))
                result["roc_curve"] = "/outputs/evaluation/roc_curve.png"
            except Exception:
                pass
        # Feature Importance
        if hasattr(model, "feature_importances_"):
            save_feature_importance(
                model, X.columns, os.path.join(eval_dir, "feature_importance.png")
            )
            result["feature_importance"] = "/outputs/evaluation/feature_importance.png"
        # Custom metric
        if loss_function:
            func = classification_metrics.get(loss_function)
            if func:
                result[loss_function] = func(y_test, y_pred)
            else:
                result["error"] = (
                    f"Unsupported loss_function for classification. Allowed: {list(classification_metrics.keys())}"
                )
    elif task_type == "regression":
        result["mse"] = mean_squared_error(y_test, y_pred)
        result["rmse"] = mean_squared_error(y_test, y_pred, squared=False)
        # Feature Importance
        if hasattr(model, "feature_importances_"):
            save_feature_importance(
                model, X.columns, os.path.join(eval_dir, "feature_importance.png")
            )
            result["feature_importance"] = "/outputs/evaluation/feature_importance.png"
        # Custom metric
        if loss_function:
            func = regression_metrics.get(loss_function)
            if func:
                result[loss_function] = func(y_test, y_pred)
            else:
                result["error"] = (
                    f"Unsupported loss_function for regression. Allowed: {list(regression_metrics.keys())}"
                )
        result["mae"] = mean_absolute_error(y_test, y_pred)
        result["r2"] = r2_score(y_test, y_pred)
    # Save model
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
    )
    os.makedirs(models_dir, exist_ok=True)
    model_id = str(uuid.uuid4())[:8]
    model_filename = f"model_{task_type}_{model_name}_{model_id}.pkl"
    model_path = os.path.join(models_dir, model_filename)
    joblib.dump(model, model_path)
    result["model_file"] = model_filename
    return result
