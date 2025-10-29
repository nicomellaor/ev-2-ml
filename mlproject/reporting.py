import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from jinja2 import Environment, FileSystemLoader
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

def generate_html_report(context: dict, template_name: str, template_dir: Path, output_path: Path):
    loader = FileSystemLoader(searchpath=template_dir)
    env = Environment(loader=loader)
    
    template = env.get_template(template_name)
    
    html_content = template.render(context)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)