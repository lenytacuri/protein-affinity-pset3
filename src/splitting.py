"""Holdout por grupos con observaciones reservadas exclusivamente a entrenamiento."""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .validacion import comprobar_particiones


def separar_por_grupos(datos, grupo, solo_train, test_size=0.2, seed=42):
    if not datos.index.is_unique or not solo_train.index.equals(datos.index):
        raise ValueError("Índices únicos y alineados requeridos")
    if datos[grupo].isna().any() or solo_train.isna().any():
        raise ValueError("Grupos y máscara no pueden tener ausentes")
    # Si una fila es obligatoria, todo su grupo queda en entrenamiento.
    reservados = set(datos.loc[solo_train.astype(bool), grupo])
    obligatorias = datos[grupo].isin(reservados)
    elegibles = datos.loc[~obligatorias].sort_index()
    if elegibles[grupo].nunique() < 2:
        raise ValueError("Se necesitan al menos dos grupos elegibles")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, test_pos = next(splitter.split(elegibles, groups=elegibles[grupo]))
    asignacion = pd.Series("train", index=datos.index, name="PARTICION")
    asignacion.loc[elegibles.iloc[test_pos].index] = "test"
    train = datos.loc[asignacion.eq("train")].copy()
    test = datos.loc[asignacion.eq("test")].copy()
    comprobar_particiones(train.index, test.index, train[grupo], test[grupo])
    assert asignacion.loc[obligatorias].eq("train").all()
    assert len(train) + len(test) == len(datos)
    return train, test, asignacion


def folds_por_grupos(grupos, solo_train, n_splits=5):
    """Índices posicionales de CV: los grupos reservados nunca van a validación."""
    import numpy as np
    from sklearn.model_selection import GroupKFold
    if not grupos.index.is_unique or not grupos.index.equals(solo_train.index):
        raise ValueError("Índices únicos y alineados requeridos")
    if grupos.isna().any() or solo_train.isna().any():
        raise ValueError("Grupos y máscara no pueden tener ausentes")
    reservados = set(grupos.loc[solo_train.astype(bool)])
    obligatorias = grupos.isin(reservados).to_numpy()
    posiciones = np.flatnonzero(~obligatorias)
    fijos = np.flatnonzero(obligatorias)
    if grupos.iloc[posiciones].nunique() < n_splits:
        raise ValueError("No hay suficientes grupos para los folds solicitados")
    folds = []
    for train, val in GroupKFold(n_splits=n_splits).split(posiciones, groups=grupos.iloc[posiciones]):
        train_idx = np.sort(np.concatenate([posiciones[train], fijos]))
        val_idx = posiciones[val]
        assert not set(grupos.iloc[train_idx]) & set(grupos.iloc[val_idx])
        folds.append((train_idx, val_idx))
    return folds
