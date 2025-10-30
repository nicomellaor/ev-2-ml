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

def plot_elbow_curve(inertia_values: list, k_range: list, output_path: Path):
    fig, ax = plt.subplots()
    ax.plot(k_range, inertia_values, marker='o')
    ax.set_xlabel('Número de clusters (k)')
    ax.set_ylabel('Inercia')
    ax.set_title('Curva Elbow para KMeans')
    plt.savefig(output_path)
    plt.close(fig)

def plot_kmeans_metrics(metrics_df: pd.DataFrame, path_prefix: Path):
    sns.set_theme(style="whitegrid")
    metric_names = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    
    for metric in metric_names:
        fig, ax = plt.subplots()
        sns.lineplot(data=metrics_df, x='k', y=metric, marker='o', ax=ax)
        ax.set_title(f'Métrica {metric} vs Número de clusters (k)')
        ax.set_xlabel('Número de clusters (k)')
        ax.set_ylabel(metric)
        plt.savefig(path_prefix / f'kmeans_{metric}_curve.png')
        plt.close(fig)

def plot_kmeans_pca_2d(pca_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots()
    sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='cluster', palette='Set2', ax=ax)
    ax.set_title('Proyección PCA 2D de Clusters KMeans')
    plt.savefig(path)
    plt.close(fig)

def save_metrics_table(metrics: dict, path: Path):
    df = pd.DataFrame(metrics)
    df.to_csv(path, index=False)

def save_centroids_table(centroids_df: pd.DataFrame, path: Path):
    centroids_df.to_csv(path, index=False)

def generate_html_report(context: dict, template_name: str, template_dir: Path, output_path: Path):
    loader = FileSystemLoader(searchpath=template_dir)
    env = Environment(loader=loader)
    
    template = env.get_template(template_name)
    
    html_content = template.render(context)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)