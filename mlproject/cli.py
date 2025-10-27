import typer
from pathlib import Path
from typing_extensions import Annotated
from . import utils, data, supervised, unsupervised, reporting, pipelines

app = typer.Typer()

@app.command()
def data_summary(config_path: Annotated[Path, typer.Option("--config", help="Ruta al archivo de configuración YAML")]):
    config = utils.load_config(config_path)
    typer.echo("Cargando y validando el conjunto de datos...")

    dataset_path = Path(config['data']['path'])
    df = data.load_dataset(dataset_path)
    typer.echo("Conjunto de datos cargado exitosamente.")

    data.validate_schema(df, config)
    typer.echo("Esquema de datos validado exitosamente.")
    
    report_path = Path(config['data']['report_path'])
    data.generate_data_report(df, report_path)
    typer.echo(f"Reporte de datos generado en {report_path}")

@app.command()
def train_supervised(): pass

@app.command()
def tune_supervised(): pass

@app.command()
def cluster_kmeans(): pass

@app.command()
def report(): pass

if __name__ == "__main__":
    app()