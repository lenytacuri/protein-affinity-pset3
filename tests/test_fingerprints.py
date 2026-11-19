import pandas as pd
import pytest
from src.fingerprints import fingerprint_minimos, crear_fingerprints, unir_fingerprints

def test_invariancias_y_multiplicidad():
    f = fingerprint_minimos
    assert f([3, 8, 5], [4, 9]) == f([9, 4], [5, 3, 8])
    assert f([3, 8, 5], [4, 9]) != f([3, 8, 5, 5], [4, 9])
    assert f([3, 8, 5], [4, 9]) != f([3, 8, 5], [4, 9.00001])
    assert f([3, 8, 5], [4, 9], 3) == f([3, 8, 5], [4, 9.00001], 3)

@pytest.mark.parametrize("valores", [[], [float("nan")], [float("inf")], [-1]])
def test_rechaza_minimos_invalidos(valores):
    with pytest.raises(ValueError):
        fingerprint_minimos(valores, [3])

def test_union_preserva_filas_y_afinidades():
    base = pd.DataFrame({"OBSERVATION_ID": ["a", "b", "c"], "GEOMETRY_ID": ["g", "g", "fallo"], "PKD": [6., 9., 7.]}, index=[9, 3, 5])
    minima = pd.DataFrame({"GEOMETRY_ID": ["g"], "DISTANCES_L": [[3, 4]], "DISTANCES_R": [[3]]})
    fp = crear_fingerprints(minima)
    resultado = unir_fingerprints(base, fp)
    pd.testing.assert_frame_equal(resultado[base.columns], base)
    assert resultado.FP_MINIMA_EXACT.notna().tolist() == [True, True, False]
    pd.testing.assert_frame_equal(resultado, unir_fingerprints(resultado, fp))


def test_vectores_solo_lectura_de_parquet():
    import numpy as np
    vector = np.array([0., 3., 8.])
    vector.setflags(write=False)
    assert fingerprint_minimos(vector, [3.]) == fingerprint_minimos([0., 3., 8.], [3.])
