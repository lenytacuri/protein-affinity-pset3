import pandas as pd
import pytest
from src import datos
from src.validacion import comprobar_particiones

def test_rechaza_grupos_compartidos():
    with pytest.raises(ValueError, match="Grupos compartidos"):
        comprobar_particiones(["a"], ["b"], ["familia1"], ["familia1"])

def test_rechaza_grupos_ausentes():
    with pytest.raises(ValueError):
        comprobar_particiones(["a"], ["b"], [None], ["familia2"])

def test_particiones_separadas():
    comprobar_particiones(["a"], ["b"], ["familia1"], ["familia2"])

def test_rechaza_ids_duplicados():
    with pytest.raises(ValueError):
        datos.validar_obt(pd.DataFrame({"OBSERVATION_ID":["a","a"], "PKD":[6.,7.]}))

def test_snapshot_detecta_cambio(tmp_path, monkeypatch):
    monkeypatch.setattr(datos, "ROOT", tmp_path)
    df = pd.DataFrame({"OBSERVATION_ID":["b","a"], "PKD":[6.,7.]})
    folder = datos.guardar_snapshot(df, {"source":"test_sintetico"})
    result, _ = datos.cargar_snapshot(folder)
    assert result.OBSERVATION_ID.tolist() == ["a", "b"]
    with (folder / "obt.parquet").open("ab") as f:
        f.write(b"modificacion")
    with pytest.raises(ValueError, match="modificado"):
        datos.cargar_snapshot(folder)
