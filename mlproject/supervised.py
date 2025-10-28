import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

def get_cv_strategy(config: dict) -> KFold | StratifiedKFold:
    task_type = config['task_type']
    n_splits = config['supervised']['cv_folds']
    random_state = config['random_seed']
    
    if task_type == 'classification':
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
def run_tuning(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, model_config: dict, cv: KFold | StratifiedKFold, scoring: str, n_iter: int, random_state: int = 42) -> BaseEstimator:
    param_distributions = model_config['params']['random_search_space']

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=random_state
    )

    random_search.fit(X, y)

    print(f"Mejor score de validación (Randomized): {random_search.best_score_.round(4)}")
    print(f"Mejores hiperparámetros encontrados (Randomized): {random_search.best_params_}")

    param_grid = model_config['params']['grid_search_space']

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )

    grid_search.fit(X, y)

    print(f"Mejor score de validación (Grid): {grid_search.best_score_.round(4)}")
    print(f"Mejores hiperparámetros encontrados (Grid): {grid_search.best_params_}")

    return grid_search.best_estimator_
    # return random_search.best_estimator_

def get_cv_metrics(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: KFold | StratifiedKFold, scoring: list[str]) -> dict:
    scores = cross_validate(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    metrics = {}
    for metric_name in scoring:
        key = f"test_{metric_name}"
        mean_score = scores[key].mean()
        std_score = scores[key].std()
        
        metrics[f"cv_{metric_name}_mean"] = mean_score
        metrics[f"cv_{metric_name}_std"] = std_score
        
        print(f"Métrica {metric_name}: {mean_score:.4f} ± {std_score:.4f}")
        
    return metrics

def get_model(model_name: str, config: dict) -> BaseEstimator:
    random_state = config['random_seed']

    if model_name == 'LogisticRegression':
        return LogisticRegression(random_state=random_state)
    elif model_name == 'RandomForestClassifier':
        return RandomForestClassifier(random_state=random_state)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
    
def evaluate_model_on_test(best_model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, config: dict) -> dict:
    '''To-Do: Retornar métricas como classification_report, roc_auc_score y otros.'''
    pass