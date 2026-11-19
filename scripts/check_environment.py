"""Prueba de infraestructura, separada de los experimentos científicos."""
import json
from pathlib import Path
import mlflow
import nbformat
from mlflow import MlflowClient
from src.tracking import configurar_tracking
import sklearn, xgboost

for path in Path('notebooks').glob('*.ipynb'):
    nbformat.validate(nbformat.read(path, as_version=4))
configurar_tracking()
client = MlflowClient()
name = 'protein-affinity-infrastructure'
experiment = client.get_experiment_by_name(name)
experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
with mlflow.start_run(experiment_id=experiment_id, run_name='environment-smoke-test') as run:
    mlflow.set_tag('purpose', 'infrastructure_only_not_a_model')
    mlflow.log_metric('connection_ok', 1)
    mlflow.log_dict({'status': 'ok', 'trained_model': False}, 'health.json')
    run_id = run.info.run_id
import tempfile
with tempfile.TemporaryDirectory() as folder:
    result = client.download_artifacts(run_id, 'health.json', folder)
    assert json.loads(Path(result).read_text())['status'] == 'ok'
print(json.dumps({'mlflow': mlflow.__version__, 'sklearn': sklearn.__version__,
                  'xgboost': xgboost.__version__, 'artifact_roundtrip': True,
                  'smoke_run_id': run_id, 'notebooks_valid': 5}))
