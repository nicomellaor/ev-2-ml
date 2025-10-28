import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from pathlib import Path

def plot_confusion_matrix(y_true, y_pred, output_path: Path):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap='Blues')
    plt.savefig(output_path)
    plt.close(fig)

def plot_roc_curve(estimator, X_test, y_test, output_path: Path):
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=ax)
    plt.savefig(output_path)
    plt.close(fig)

def plot_precision_recall_curve(estimator, X_test, y_test, output_path: Path):
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_estimator(estimator, X_test, y_test, ax=ax)
    plt.savefig(output_path)
    plt.close(fig)

def plot_elbow_curve(): pass

def plot_kmeans_metrics(): pass

def plot_kmeans_pca_2d(): pass

def save_metrics_table(): pass

def save_centroids_table(): pass

def generate_html_report(): pass # jinja2