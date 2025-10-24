import typer
from pathlib import Path
from . import utils, data, supervised, unsupervised, reporting, pipelines

app = typer.Typer()

@app.command()
def data_summary(): pass

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