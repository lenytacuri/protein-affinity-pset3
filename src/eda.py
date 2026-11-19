"""Resumen descriptivo sin imputar ni modificar observaciones."""
import numpy as np
import pandas as pd


def resumen_numerico(frame):
    """Porcentajes sobre valores finitos; Z-score con desviación muestral (ddof=1)."""
    rows = []
    for nombre in frame:
        raw = pd.to_numeric(frame[nombre], errors="raise")
        finitos = raw.notna() & np.isfinite(raw)
        x = raw.loc[finitos].astype(float)
        n = len(x)
        q1, mediana, q3 = x.quantile([0.25, 0.5, 0.75])
        media, std = x.mean(), x.std(ddof=1)
        iqr = q3 - q1
        inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        fuera = int(((x < inferior) | (x > superior)).sum())
        z_valido = n > 1 and pd.notna(std) and std > 0
        z_count = int(((x - media).abs() / std > 3).sum()) if z_valido else np.nan
        rows.append({
            "feature": nombre, "validos": n, "ausentes": int(raw.isna().sum()),
            "no_finitos": int((raw.notna() & ~np.isfinite(raw)).sum()),
            "valores_distintos": x.nunique(), "media": media, "mediana": mediana,
            "desviacion_estandar": std, "minimo": x.min(), "Q1": q1,
            "Q3": q3, "maximo": x.max(),
            "asimetria": x.skew() if n >= 3 and x.nunique() > 1 else np.nan,
            "IQR": iqr, "limite_IQR_inferior": inferior, "limite_IQR_superior": superior,
            "fuera_IQR_count": fuera if n else np.nan,
            "fuera_IQR_pct": 100 * fuera / n if n else np.nan,
            "z_mayor_3_count": z_count,
            "z_mayor_3_pct": 100 * z_count / n if n else np.nan,
            "IQR_cero": bool(n and iqr == 0),
        })
    return pd.DataFrame(rows).set_index("feature")
