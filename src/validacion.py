"""Guardas de integridad; no asigna grupos biológicos ni genera splits todavía."""
def comprobar_particiones(train_ids, test_ids, train_groups, test_groups):
    for label, values in [("train IDs", train_ids), ("test IDs", test_ids),
                          ("train grupos", train_groups), ("test grupos", test_groups)]:
        if not len(values) or any(x is None or str(x).strip() == "" or x != x for x in values):
            raise ValueError(f"{label}: vacío o valores ausentes")
    if len(train_ids) != len(train_groups) or len(test_ids) != len(test_groups):
        raise ValueError("Cada observación debe tener un grupo")
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("IDs repetidos")
    if set(train_ids) & set(test_ids):
        raise ValueError("Observaciones compartidas")
    if set(train_groups) & set(test_groups):
        raise ValueError("Grupos compartidos: fuga de datos")
