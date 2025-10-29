### Crear entorno virtual e instalar dependencias
1. `python -m venv .venv` o `python3 -m venv .venv`
2. `.venv\Scripts\activate` o `source .venv/bin/activate`
3. `pip install -e .`

### Scripts por CLI
- `python3 -m mlproject.cli data-summary --config configs/default.yaml`
- `python3 -m mlproject.cli tune-supervised --config configs/default.yaml`
- `python3 -m mlproject.cli cluster-kmeans --config configs/default.yaml`
- `python3 -m mlproject.cli report --run-id [run_id]`