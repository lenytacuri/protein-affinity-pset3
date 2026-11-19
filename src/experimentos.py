"""Evaluación de candidatos sin consultar test ni guardar modelos finales."""
import json
import time
import warnings
import numpy as np
import pandas as pd
import mlflow
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .pipeline import construir_pipeline
from .tracking import configurar_tracking


def comparar_candidatos(X, y, folds, modelos, carpeta, contexto, seed=42):
    configurar_tracking()
    carpeta.mkdir(parents=True, exist_ok=True)
    resultados = []
    advertencias = []
    candidatos = {"DummyMean": DummyRegressor(strategy="mean")}
    candidatos.update({m: construir_pipeline(m, list(X.columns), seed=seed, n_jobs=-1) for m in modelos})
    for nombre, pipeline in candidatos.items():
        print(f"Evaluando {nombre}...", flush=True)
        predicciones = []
        inicio = time.monotonic()
        with mlflow.start_run(run_name=f"baseline_{nombre}") as run:
            mlflow.set_tags({"phase":"baseline_group_cv", "test_used":"false", "protocol":"defaults_v2", "family":nombre,
                             "git_commit":contexto.get("git_commit", "unknown")})
            mlflow.log_params({"candidate":nombre, "seed":seed, "cv_folds":len(folds),
                               "train_rows":len(X), "predictors":X.shape[1], "transformation":"none", "estimator_defaults":True, "blas_threads":1})
            mlflow.log_dict(contexto, "contexto.json")
            mlflow.log_dict({k:str(v) for k,v in pipeline.get_params(deep=True).items()}, "parametros.json")
            for fold, (tr, va) in enumerate(folds, 1):
                fitted = clone(pipeline)
                with warnings.catch_warnings(record=True) as capturadas:
                    warnings.simplefilter("always")
                    # Limitar BLAS evita sobrecarga en matrices pequeñas; no cambia hiperparámetros.
                    with threadpool_limits(limits=1, user_api="blas"):
                        t_fit = time.monotonic()
                        fitted.fit(X.iloc[tr], y.iloc[tr])
                        fit_seconds = time.monotonic() - t_fit
                        train_pred = fitted.predict(X.iloc[tr])
                        t_pred = time.monotonic()
                        pred = fitted.predict(X.iloc[va])
                        predict_seconds = time.monotonic() - t_pred
                for w in capturadas:
                    advertencias.append({"modelo":nombre,"fold":fold,"categoria":w.category.__name__,"mensaje":str(w.message)})
                metrics = {"rmse":float(np.sqrt(mean_squared_error(y.iloc[va],pred))),
                           "mae":float(mean_absolute_error(y.iloc[va],pred)),
                           "r2":float(r2_score(y.iloc[va],pred))}
                metrics.update({
                    "train_rmse":float(np.sqrt(mean_squared_error(y.iloc[tr],train_pred))),
                    "train_mae":float(mean_absolute_error(y.iloc[tr],train_pred)),
                    "train_r2":float(r2_score(y.iloc[tr],train_pred)),
                    "fit_seconds":fit_seconds, "validation_predict_seconds":predict_seconds,
                })
                metrics["gap_rmse"] = metrics["rmse"] - metrics["train_rmse"]
                metrics["gap_mae"] = metrics["mae"] - metrics["train_mae"]
                metrics["gap_r2"] = metrics["train_r2"] - metrics["r2"]
                mlflow.log_metrics(metrics, step=fold)
                resultados.append({"modelo":nombre,"fold":fold,"n_train":len(tr),"n_validacion":len(va),
                                   "run_id":run.info.run_id,**metrics})
                predicciones.append(pd.DataFrame({"fila_modelo":X.index[va],"fold":fold,"pkd_real":y.iloc[va].to_numpy(),"pkd_predicho":pred}))
                print(f"  fold {fold}: RMSE={metrics['rmse']:.3f}", flush=True)
            tabla = pd.DataFrame([r for r in resultados if r["modelo"]==nombre])
            for metrica in ["rmse","mae","r2","train_rmse","train_mae","train_r2","gap_rmse","gap_mae","gap_r2","fit_seconds","validation_predict_seconds"]:
                mlflow.log_metric(metrica+"_mean",float(tabla[metrica].mean()))
                mlflow.log_metric(metrica+"_std",float(tabla[metrica].std()))
            mlflow.log_metric("fit_seconds_total",float(tabla["fit_seconds"].sum()))
            mlflow.log_metric("duration_seconds",time.monotonic()-inicio)
            oof = pd.concat(predicciones,ignore_index=True)
            assert oof.fila_modelo.is_unique
            archivo = carpeta / f"oof_{nombre}.csv"
            oof.to_csv(archivo,index=False)
            mlflow.log_artifact(str(archivo))
            mlflow.log_dict([w for w in advertencias if w['modelo']==nombre],"advertencias.json")
        pd.DataFrame(resultados).to_csv(carpeta/"metricas_folds.csv",index=False)
        (carpeta/"advertencias.json").write_text(json.dumps(advertencias,indent=2))
    detalle = pd.DataFrame(resultados)
    resumen = detalle.groupby("modelo")[["rmse","mae","r2","train_rmse","train_mae","train_r2","gap_rmse","gap_mae","gap_r2","fit_seconds","validation_predict_seconds"]].agg(["mean","std"])
    resumen.columns = ["_".join(c) for c in resumen.columns]
    resumen["fit_seconds_total"] = detalle.groupby("modelo")["fit_seconds"].sum()
    resumen = resumen.sort_values("rmse_mean")
    resumen.to_csv(carpeta/"comparacion.csv")
    (carpeta/"contexto.json").write_text(json.dumps(contexto,indent=2))
    return resumen, detalle


def ajustar_candidatos(X, y, folds, grids, carpeta, contexto, baseline, seed=42, n_jobs=4):
    """GridSearchCV sobre entrenamiento; una corrida padre por familia y una hija por combinación."""
    from sklearn.model_selection import GridSearchCV, ParameterGrid
    from sklearn.metrics import make_scorer
    from joblib import parallel_config
    from mlflow.models import infer_signature
    configurar_tracking()
    carpeta.mkdir(parents=True, exist_ok=True)
    resumen = []
    scoring = {"rmse":"neg_root_mean_squared_error", "mae":"neg_mean_absolute_error", "r2":"r2"}
    for nombre, grid in grids.items():
        cantidad = len(ParameterGrid(grid))
        print(f"{nombre}: {cantidad} combinaciones × {len(folds)} folds", flush=True)
        pipeline = construir_pipeline(nombre, list(X.columns), seed=seed, n_jobs=1)
        with mlflow.start_run(run_name=f"tuning_{nombre}") as padre:
            mlflow.set_tags({"phase":"tuning", "family":nombre, "test_used":"false",
                             "git_commit":contexto.get("git_commit", "unknown")})
            mlflow.log_params({"search":"GridSearchCV", "configurations":cantidad,
                               "folds":len(folds), "seed":seed, "parallel_jobs":n_jobs})
            mlflow.log_dict(contexto,"contexto.json")
            mlflow.log_dict(grid,"grid.json")
            inicio = time.monotonic()
            search = GridSearchCV(pipeline, grid, scoring=scoring, refit="rmse", cv=folds,
                                  return_train_score=True, n_jobs=n_jobs, pre_dispatch=n_jobs,
                                  error_score="raise", verbose=1)
            with parallel_config(backend="loky", inner_max_num_threads=1):
                with threadpool_limits(limits=1, user_api="blas"):
                    search.fit(X, y)
            total_seconds = time.monotonic()-inicio
            cv = search.cv_results_
            records = []
            best_child = None
            for i, params in enumerate(cv["params"]):
                with mlflow.start_run(run_name=f"config_{i+1:02}", nested=True) as hija:
                    mlflow.set_tags({"phase":"tuning", "family":nombre, "test_used":"false",
                                     "git_commit":contexto.get("git_commit", "unknown")})
                    mlflow.log_params({**params, "seed":seed, "cv_folds":len(folds)})
                    mlflow.log_dict(contexto,"contexto.json")
                    record = {"modelo":nombre,"configuracion":i+1,"run_id":hija.info.run_id,
                              "parent_run_id":padre.info.run_id,"params":json.dumps(params),
                              "rank_rmse":int(cv["rank_test_rmse"][i]),
                              "fit_seconds_mean":float(cv["mean_fit_time"][i]),
                              "score_seconds_mean":float(cv["mean_score_time"][i])}
                    for metrica in scoring:
                        signo = -1 if metrica != "r2" else 1
                        # ddof=1: misma convención que la comparación baseline.
                        train_scores = np.array([signo*cv[f"split{k}_train_{metrica}"][i] for k in range(len(folds))])
                        val_scores = np.array([signo*cv[f"split{k}_test_{metrica}"][i] for k in range(len(folds))])
                        record[f"train_{metrica}_mean"] = float(train_scores.mean())
                        record[f"{metrica}_mean"] = float(val_scores.mean())
                        record[f"{metrica}_std"] = float(val_scores.std(ddof=1))
                        record[f"train_{metrica}_std"] = float(train_scores.std(ddof=1))
                        record[f"gap_{metrica}"] = float((val_scores-train_scores).mean() if metrica != "r2" else (train_scores-val_scores).mean())
                        for fold, (tr, va) in enumerate(zip(train_scores,val_scores),1):
                            mlflow.log_metrics({f"train_{metrica}":float(tr),f"cv_{metrica}":float(va)},step=fold)
                    mlflow.log_metrics({k:float(v) for k,v in record.items() if isinstance(v,(float,int))})
                    if i == search.best_index_:
                        best_child = hija.info.run_id
                        mlflow.set_tag("best_in_family", "true")
                        signature = infer_signature(X.head(5),search.best_estimator_.predict(X.head(5)))
                        mlflow.sklearn.log_model(search.best_estimator_,artifact_path="pipeline",
                                                signature=signature,input_example=X.head(5))
                    records.append(record)
            resultados = pd.DataFrame(records)
            archivo = carpeta / f"grid_{nombre}.csv"
            resultados.to_csv(archivo,index=False)
            mlflow.log_artifact(str(archivo))
            best = records[search.best_index_]
            previo = float(baseline.loc[nombre,"rmse_mean"])
            improvement = previo-best["rmse_mean"]
            # Regla del PSet: no llamar mejora si no supera variación entre folds.
            referencia_std = max(float(baseline.loc[nombre,"rmse_std"]),best["rmse_std"])
            fila = {"modelo":nombre,"rmse_baseline":previo,"rmse_cv":best["rmse_mean"],
                    "rmse_cv_std":best["rmse_std"],"rmse_train":best["train_rmse_mean"],
                    "gap_rmse":best["gap_rmse"],"mejora_rmse":improvement,
                    "std_referencia":referencia_std,"supera_variacion_folds":bool(improvement>referencia_std),
                    "mae_cv":best["mae_mean"],"r2_cv":best["r2_mean"],
                    "best_params":json.dumps(search.best_params_),"best_run_id":best_child,
                    "parent_run_id":padre.info.run_id,"pipeline_uri":f"runs:/{best_child}/pipeline",
                    "search_seconds":total_seconds,"refit_seconds":float(search.refit_time_),
                    "n_combinaciones":cantidad}
            mlflow.log_metrics({"best_rmse_cv":fila["rmse_cv"], "improvement_rmse":improvement,
                                "search_seconds":total_seconds,"refit_seconds":float(search.refit_time_)})
            mlflow.set_tag("best_child_run_id",best_child)
            mlflow.log_dict(fila,"resultado.json")
            resumen.append(fila)
            pd.DataFrame(resumen).sort_values("rmse_cv").to_csv(carpeta/"comparacion_tuning.csv",index=False)
            print(f"{nombre} terminado: RMSE CV={fila['rmse_cv']:.4f}",flush=True)
    (carpeta/"contexto.json").write_text(json.dumps(contexto,indent=2))
    return pd.DataFrame(resumen).sort_values("rmse_cv").reset_index(drop=True)
