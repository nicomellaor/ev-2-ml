from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator

def create_preprocessor(numerical_features: list[str], categorical_features: list[str], use_pca: bool = False, pca_variance_target: float = 0.95) -> Pipeline: 
    '''Crea el ColumnTransformer y el pipeline de preprocesamiento.'''
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough'
    )

    steps = [('preprocessor', preprocessor)]

    if use_pca:
        steps.append(('pca', PCA(n_components=pca_variance_target)))

    return Pipeline(steps=steps)

def get_pipeline(preprocessor: Pipeline, model: BaseEstimator) -> Pipeline: 
    '''Devuelve el pipeline encadenando el preprocesador con el modelo.'''
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
