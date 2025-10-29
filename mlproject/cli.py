import typer
import json
from datetime import datetime
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
def tune_supervised(config_path: Annotated[Path, typer.Option("--config", help="Ruta al archivo de configuración YAML")]):
    typer.echo("Iniciando entrenamiento supervisado...")
    config = utils.load_config(config_path)
    random_seed = config['random_seed']
    target_col = config['data']['target_col']

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    typer.echo(f"ID de la ejecución: {run_id}")

    utils.setup_logging(Path(f"outputs/runs/{run_id}/run.log"))

    typer.echo("Cargando y dividiendo el conjunto de datos...")
    dataset_path = Path(config['data']['path'])
    df = data.load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = data.split_data(df=df, target_col=target_col, test_size=config['data']['test_size'], stratify=(config['task_type']=='classification'), random_state=random_seed)
    typer.echo("Conjunto de datos dividido exitosamente.")

    numeric_features = config['data']['schema']['numerical_cols']
    categorical_features = config['data']['schema']['categorical_cols']

    preprocessor = pipelines.create_preprocessor(numeric_features, categorical_features, use_pca=config['preprocessing']['use_pca'], pca_variance_target=config['preprocessing']['pca_variance_target'])
    
    cv_strategy = supervised.get_cv_strategy(config)

    run_metadata = {"run_id": run_id, "config": config, "models": {}}

    for model_key, model_config in config['supervised']['models'].items():
        model_name = model_config['model_name']
        typer.echo(f"Entrenando modelo: {model_key} {model_name}...")

        model = supervised.get_model(model_name, config)
        pipeline = pipelines.get_pipeline(preprocessor, model)

        best_model, best_params = supervised.run_tuning(
            pipeline=pipeline,
            X=X_train,
            y=y_train,
            model_config=model_config,
            config=config,
            cv=cv_strategy,
            scoring=config['supervised']['cv_metric'],
            n_iter=config['supervised']['random_search_iterations'],
            random_state=random_seed
        )

        main_metric = config['supervised']['cv_metric']

        if config['task_type'] == 'classification':
            report_metrics = [main_metric, 'balanced_accuracy', 'f1_weighted']
        else:
            report_metrics = [main_metric, 'neg_root_mean_squared_error', 'r2']
        
        cv_metrics = supervised.get_cv_metrics(
            pipeline=best_model,
            X=X_train,
            y=y_train,
            cv=cv_strategy,
            scoring=report_metrics
        )

        test_metrics = supervised.evaluate_model_on_test(
            best_model=best_model,
            X_test=X_test,
            y_test=y_test,
            config=config
        )

        output_dir = Path(f"outputs/runs/{run_id}/{model_key}")
        output_dir.mkdir(parents=True, exist_ok=True)

        utils.save_artifact(best_model, output_dir / "best_model.joblib")

        if config['task_type'] == 'classification':
            y_pred_test = best_model.predict(X_test)
            reporting.plot_confusion_matrix(y_test, y_pred_test, output_dir / "confusion_matrix.png")
            reporting.plot_roc_curve(best_model, X_test, y_test, output_dir / "roc_curve.png")
            reporting.plot_precision_recall_curve(best_model, X_test, y_test, output_dir / "precision_recall_curve.png")

        typer.echo(f"Modelo {model_key} entrenado y evaluado exitosamente.")
        typer.echo(f"Outputs guardados en {output_dir}")

        run_metadata['models'][model_key] = {
            "model_name": model_name,
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
            "best_params" : best_params,
        }

    utils.save_run_metadata(run_metadata, Path(f"outputs/runs/{run_id}/run_metadata.json"))
    typer.echo("Metadatos de la ejecución guardados exitosamente.")
    typer.echo(f"Ejecución {run_id} completada.")

@app.command()
def cluster_kmeans(config_path: Annotated[Path, typer.Option("--config", help="Ruta al archivo de configuración YAML")]): pass

@app.command()
def report(run_id: Annotated[Path, typer.Option("--run-id", help="ID de la ejecución para generar el reporte")]):
    typer.echo("Generando reporte HTML de la ejecución...")

    run_dir = Path(f"outputs/runs/{run_id}")
    metadata_path = run_dir / "run_metadata.json"
    template_dir = Path("templates/")
    template_name = "report.html"
    output_path = run_dir / "final_report.html"

    if not metadata_path.exists():
        typer.echo(f"No se encontró el archivo de metadatos en {metadata_path}")
        raise typer.Exit(code=1)
    
    typer.echo(f"Cargando metadatos desde {metadata_path}...")
    with open(metadata_path, 'r') as f:
        context = json.load(f)

    context['run_id'] = run_id
    context['generation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for model_name in context['models'].keys():
        context['models'][model_name]['image_paths'] = {
            "confusion_matrix": f"{model_name}/confusion_matrix.png",
            "roc_curve": f"{model_name}/roc_curve.png",
            "precision_recall_curve": f"{model_name}/precision_recall_curve.png",
            # Añadir K-means
        }

    try:
        reporting.generate_html_report(
            context=context,
            template_name=template_name,
            template_dir=template_dir,
            output_path=output_path
        )
        typer.echo(f"Reporte HTML generado en {output_path}")
    except Exception as e:
        typer.echo(f"Error al generar el reporte HTML: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()