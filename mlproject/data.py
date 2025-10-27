import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"El archivo {path} no existe.")
    return pd.read_csv(path)

def validate_schema(df: pd.DataFrame, config: dict) -> bool:
    try:
        target = config['data']['target_col']
        schema = config['data']['schema']
        numerical_cols = schema['numerical_cols']
        categorical_cols = schema['categorical_cols']
        max_null_percentage = schema['max_null_percentage']

    except KeyError as e:
        raise KeyError(f"Falta la clave en la configuración: {e}")
    
    # Validación de columnas esperadas y tipos de datos
    expected_columns = numerical_cols + categorical_cols + [target]

    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"La columna esperada '{col}' no está en el DataFrame.")
    for col in numerical_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"La columna '{col}' no es numérica.")
    for col in categorical_cols:
        if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
            raise TypeError(f"La columna '{col}' no es categórica.")
        
    # Validación de valores nulos
    null_percentage = df[expected_columns].isnull().mean()
    for col, perc in null_percentage.items():
        if perc > max_null_percentage:
            raise ValueError(f"La columna '{col}' tiene {perc*100:.2f}% de valores nulos. Máximo permitido es {max_null_percentage*100:.2f}%.")


def generate_data_report(df: pd.DataFrame, report_path: Path):
    report = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "columns": df.dtypes.apply(lambda x: x.name).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_statistics": df.describe().to_dict()
    }
    with open(report_path, 'w') as f:
        f.write(str(report))

def split_data(df: pd.DataFrame, target_col: str, test_size: float, stratify: bool, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify_col = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify_col, random_state=random_state)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return train_df, test_df