import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans

def evaluate_k_values(X_processed: pd.DataFrame, k_range: list[int]) -> dict[str, list]:
    metrics = {
        'k': [],
        'inertia': [],
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': []
    }
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_processed)
        
        metrics['k'].append(k)
        metrics['inertia'].append(kmeans.inertia_)
        metrics['silhouette'].append(silhouette_score(X_processed, labels))
        metrics['calinski_harabasz'].append(calinski_harabasz_score(X_processed, labels))
        metrics['davies_bouldin'].append(davies_bouldin_score(X_processed, labels))
        
    return metrics

def get_cluster_centroids(kmeans_pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    # Extraer K-Means y preprocesador del pipeline
    try:
        kmeans_model = kmeans_pipeline.named_steps['kmeans']
        preprocessor = kmeans_pipeline.named_steps['preprocessor']
    except KeyError as e:
        raise KeyError(f"El pipeline no contiene el paso esperado: {e}")
    
    # Obtener los centroides en el espacio transformado
    centroids_scaled = kmeans_model.cluster_centers_
    
    # Desescalar los centroides a las características originales
    try:
        # Extraer el ColumnTransformer del preprocesador
        column_transformer = preprocessor.named_steps['preprocessor']

        # Desescalar númericas y descodificar categóricas
        centroids_original = column_transformer.inverse_transform(centroids_scaled)

        # Crear DataFrame con nombres de columnas originales
        df_centroids = pd.DataFrame(centroids_original, columns=feature_names)

    except (ValueError, AttributeError, KeyError) as e:
        print(f"No se pudo invertir la transformación de los centroides: {e}")
        try:
            # Obtener nombres de columnas transformadas
            transformed_feature_names = preprocessor.get_feature_names_out()
            df_centroids = pd.DataFrame(centroids_scaled, columns=transformed_feature_names)

        except Exception:
            # Devolver sin nombres de columnas
            df_centroids = pd.DataFrame(centroids_scaled)
    
    df_centroids.index.name = 'Cluster'
    return df_centroids

def get_cluster_sizes(labels: np.ndarray) -> pd.DataFrame:
    # Contar apariciones de cada etiqueta de clúster
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    
    # Calcular porcentaje
    cluster_percentages = (cluster_counts / cluster_counts.sum())
    
    # Crear un DataFrame
    size_df = pd.DataFrame({
        'Count': cluster_counts,
        'Percentage': cluster_percentages
    })
    
    size_df.index.name = 'Cluster'
    return size_df

def get_pca_2d_projection(X_processed: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    pca = PCA(n_components=2)
    X_pca_2d = pca.fit_transform(X_processed)
    
    pca_df = pd.DataFrame(
        data=X_pca_2d, 
        columns=['PCA1', 'PCA2']
    )
    pca_df['cluster'] = labels
    return pca_df