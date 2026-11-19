import io
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from src.pipeline import MODELOS, construir_pipeline
from src.splitting import folds_por_grupos

@pytest.mark.parametrize('modelo', MODELOS)
def test_pipeline_imputa_solo_train_y_serializa(modelo):
    x = pd.DataFrame({'a': [1., 2., np.nan, 4., 5., 6.], 'b': [0., 1., 2., 3., 4., 5.]})
    y = np.arange(6.)
    params = {'n_estimators': 3} if modelo in ('RandomForestRegressor', 'XGBRegressor') else {}
    if modelo == 'MLPRegressor': params = {'hidden_layer_sizes': (3,), 'max_iter': 100, 'solver': 'lbfgs'}
    pipe = clone(construir_pipeline(modelo, list(x), n_jobs=1, parametros_modelo=params))
    pipe.fit(x, y)
    imp = pipe.named_steps['preparacion'].named_transformers_['originales'].named_steps['imputacion']
    assert imp.statistics_[0] == 4.
    pred = pipe.predict(pd.DataFrame({'b': [100.], 'a': [np.nan], 'PKD': [999.]}))
    assert imp.statistics_[0] == 4.
    stream = io.BytesIO(); joblib.dump(pipe, stream); stream.seek(0)
    np.testing.assert_allclose(pred, joblib.load(stream).predict(pd.DataFrame({'a':[np.nan], 'b':[100.]})))


def test_transformaciones_y_guardas():
    x = pd.DataFrame({'a':[0., 1., 2., 8., np.nan], 'b':[-2., -1., 0., 1., 3.]})
    for modo, cols in [('log1p', ['a']), ('yeo-johnson', ['b'])]:
        p = construir_pipeline('Ridge', list(x), modo, cols)
        assert np.isfinite(p.fit(x, np.arange(5.)).predict(x)).all()
    with pytest.raises(ValueError):
        construir_pipeline('Ridge', list(x), 'log1p', ['b']).fit(x, np.arange(5.))
    with pytest.raises(ValueError): construir_pipeline('Ridge', ['PKD'])


def test_folds_no_comparten_grupos_ni_validan_obligatorios():
    groups = pd.Series(['a','a','b','c','d','e','z','z'])
    fixed = pd.Series([False]*6 + [True, False])
    folds = folds_por_grupos(groups, fixed, 3)
    validated = []
    for tr, va in folds:
        assert {6,7} <= set(tr)
        assert not set(groups.iloc[tr]) & set(groups.iloc[va])
        validated.extend(va)
    assert sorted(validated) == list(range(6))


def test_hiperparametros_por_defecto():
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from xgboost import XGBRegressor
    clases = [Ridge, RandomForestRegressor, XGBRegressor, MLPRegressor, SVR]
    for clase in clases:
        esperado = clase().get_params()
        real = construir_pipeline(clase.__name__, ["a"]).named_steps["modelo"].get_params()
        assert {k:v for k,v in real.items() if k not in {"random_state", "n_jobs"}} == {k:v for k,v in esperado.items() if k not in {"random_state", "n_jobs"}}
