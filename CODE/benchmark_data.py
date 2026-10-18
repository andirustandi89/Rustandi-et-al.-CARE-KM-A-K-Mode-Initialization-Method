from pathlib import Path
import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile

import numpy as np
import pandas as pd

# Generalization datasets physically stored in CARE-KM-PAPER/DATA.
# The registry is intentionally explicit so notebook 09/10 cannot silently
# pick up unrelated files.
DATASETS = {
    "car": {
        "file": "dataset_car+evaluation.zip",
        "display": "Car Evaluation",
        "reference_rows": 1728,
        "reference_features": 6,
        "reference_k": 4,
    },
    "lymphography": {
        "file": "dataset_lymphography.zip",
        "display": "Lymphography",
        "reference_rows": 148,
        "reference_features": 18,
        "reference_k": 4,
    },
    "hayes_roth": {
        "file": "hayes+roth.zip",
        "display": "Hayes-Roth",
        "reference_rows": 160,
        "reference_features": 3,
        "reference_k": 3,
    },
    "australian_credit": {
        "file": "statlog+australian+credit+approval.zip",
        "display": "Statlog Australian Credit Approval",
        "reference_rows": 690,
        "reference_features": 14,
        "reference_k": 2,
    },
    "chronic_kidney": {
        "file": "chronic+kidney+disease.zip",
        "display": "Chronic Kidney Disease",
        "reference_rows": 400,
        "reference_features": 24,
        "reference_k": 2,
    },
    "annealing": {
        "file": "annealing.zip",
        "display": "Annealing",
        # The UCI archive used here contains train + test = 898 rows and
        # five observed class labels.
        "reference_rows": 898,
        "reference_features": 38,
        "reference_k": 5,
    },
    "statlog_heart": {
        "file": "statlog+heart.zip",
        "display": "Statlog Heart",
        "reference_rows": 270,
        "reference_features": 13,
        "reference_k": 2,
    },
    "heart_disease": {
        "file": "heart+disease.zip",
        "display": "Heart Disease (Cleveland)",
        "reference_rows": 303,
        "reference_features": 13,
        "reference_k": 2,
    },
    "thyroid_disease": {
        "file": "thyroid+disease.zip",
        "display": "Thyroid Disease (New Thyroid subset)",
        "reference_rows": 215,
        "reference_features": 5,
        "reference_k": 3,
    },
    "nursery": {
        "file": "dataset_nursery.zip",
        "display": "Nursery",
        "reference_rows": 12960,
        "reference_features": 8,
        "reference_k": 5,
    },
}

DATASET_ORDER = list(DATASETS.keys())


def _find_member(zf, suffixes):
    names = [n for n in zf.namelist() if not n.endswith("/")]
    lowered = [(n, n.lower()) for n in names]
    for suffix in suffixes:
        suffix = suffix.lower()
        for original, low in lowered:
            if low.endswith(suffix):
                return original
    raise FileNotFoundError(
        "Data file not found in ZIP. Available files: " + str(names[:25])
    )


def _read_member_bytes(zf, member):
    raw = zf.read(member)
    if member.lower().endswith(".z"):
        try:
            proc = subprocess.run(
                ["gzip", "-cd"],
                input=raw,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("gzip is required to read this .Z archive member.") from exc
        if proc.returncode != 0:
            message = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Failed to decompress {member!r}: {message}")
        raw = proc.stdout
    return raw


def _read_csv_member(zf, member, **kwargs):
    return pd.read_csv(io.BytesIO(_read_member_bytes(zf, member)), **kwargs)


def _strip_frame(df):
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip()
    return out


def _clean_missing_strings(df):
    out = df.copy()
    for c in out.columns:
        s = out[c].astype(str).str.strip()
        out[c] = s.replace({
            "?": "missing", "": "missing", "NA": "missing",
            "N/A": "missing", "nan": "missing", "NaN": "missing",
        })
    return out


def _auto_categorize_numeric(df, max_native_cardinality=10, q=5):
    """Keep low-cardinality values categorical and quantile-bin high-cardinality numeric."""
    out = df.copy()
    for col in out.columns:
        s = out[col].astype(str).str.strip()
        missing = s.isin(["?", "", "nan", "NaN", "NA", "N/A"])
        numeric = pd.to_numeric(s.where(~missing), errors="coerce")
        nonmissing = int((~missing).sum())
        if (
            nonmissing
            and int(numeric.notna().sum()) == nonmissing
            and int(numeric.nunique(dropna=True)) > max_native_cardinality
        ):
            bins = min(q, int(numeric.nunique(dropna=True)))
            try:
                codes = pd.qcut(numeric, q=bins, labels=False, duplicates="drop")
                s = codes.map(lambda v: "missing" if pd.isna(v) else f"Q{int(v)+1}")
            except ValueError:
                s = numeric.astype(str)
                s[missing] = "missing"
        else:
            s = s.mask(missing, "missing")
        out[col] = s.astype(str)
    return out


def _binary_presence_target(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(0).gt(0).map({False: "absence", True: "presence"})


def load_car(path):
    cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    with zipfile.ZipFile(path) as zf:
        member = _find_member(zf, ["car.data", "/car.data"])
        df = _read_csv_member(zf, member, header=None, names=cols, dtype=str)
    return df.drop(columns="class"), df["class"]


def load_lymphography(path):
    cols = ["class"] + [f"a{i}" for i in range(1, 19)]
    with zipfile.ZipFile(path) as zf:
        member = _find_member(zf, ["lymphography.data", ".data"])
        df = _read_csv_member(
            zf, member, header=None, names=cols, dtype=str, keep_default_na=False
        )
    df = _strip_frame(df)
    return df.drop(columns="class"), df["class"]


def load_hayes_roth(path):
    with zipfile.ZipFile(path) as zf:
        tr = _read_csv_member(
            zf, _find_member(zf, ["hayes-roth.data"]),
            header=None, dtype=str, keep_default_na=False
        )
        te = _read_csv_member(
            zf, _find_member(zf, ["hayes-roth.test"]),
            header=None, dtype=str, keep_default_na=False
        )
    # The test file omits the instance-name distractor.
    Xtr = tr.iloc[:, [2, 3, 4]].copy()
    Xtr.columns = ["age", "education", "marital"]
    Xte = te.iloc[:, [1, 2, 3]].copy()
    Xte.columns = ["age", "education", "marital"]
    ytr = tr.iloc[:, 5].copy()
    yte = te.iloc[:, 4].copy()
    return (
        pd.concat([Xtr, Xte], ignore_index=True),
        pd.concat([ytr, yte], ignore_index=True),
    )


def load_australian_credit(path):
    with zipfile.ZipFile(path) as zf:
        member = _find_member(zf, ["australian.dat"])
        df = pd.read_csv(
            io.BytesIO(_read_member_bytes(zf, member)),
            sep=r"\s+", header=None, dtype=str, engine="python"
        )
    df.columns = [f"a{i}" for i in range(1, 15)] + ["class"]
    return _auto_categorize_numeric(df.drop(columns="class")), df["class"]


def _extract_ckd_arff_from_outer_zip(path, data_dir=None):
    # Optional manually-extracted fallback. If present, it avoids any RAR tool.
    if data_dir is not None:
        data_dir = Path(data_dir)
        for name in [
            "chronic_kidney_disease_full.arff",
            "chronic_kidney_disease.arff",
        ]:
            candidate = data_dir / name
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate.read_bytes()

    with zipfile.ZipFile(path) as zf:
        # Some repackaged mirrors may already contain the ARFF directly.
        direct = [
            n for n in zf.namelist()
            if n.lower().endswith("chronic_kidney_disease_full.arff")
            or n.lower().endswith("chronic_kidney_disease.arff")
        ]
        if direct:
            direct.sort(key=lambda n: ("full" not in n.lower(), len(n)))
            return zf.read(direct[0])

        rar_member = _find_member(zf, ["Chronic_Kidney_Disease.rar", ".rar"])
        rar_bytes = zf.read(rar_member)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rar_path = td / "dataset.rar"
        rar_path.write_bytes(rar_bytes)

        def available_commands():
            commands = []
            if shutil.which("unrar"):
                commands.append(["unrar", "x", "-o+", str(rar_path), str(td)])
            if shutil.which("unrar-free"):
                commands.append(["unrar-free", "x", str(rar_path), str(td)])
            if shutil.which("7z"):
                commands.append(["7z", "x", "-y", f"-o{td}", str(rar_path)])
            if shutil.which("unar"):
                commands.append(["unar", "-f", "-o", str(td), str(rar_path)])
            if shutil.which("bsdtar"):
                commands.append(["bsdtar", "-xf", str(rar_path), "-C", str(td)])
            return commands

        commands = available_commands()

        # Colab commonly permits installing unrar-free. Only attempt this when
        # the CKD dataset is actually loaded and no extractor is available.
        if not commands and os.environ.get("COLAB_RELEASE_TAG"):
            try:
                subprocess.run(
                    ["apt-get", "update", "-qq"],
                    check=False, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, timeout=180,
                )
                subprocess.run(
                    ["apt-get", "install", "-y", "-qq", "unrar-free"],
                    check=False, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, timeout=180,
                )
            except Exception:
                pass
            commands = available_commands()

        if not commands:
            raise RuntimeError(
                "Chronic Kidney Disease source contains a RAR archive and no RAR "
                "extractor is available. In Colab install `unrar-free`, or place "
                "`chronic_kidney_disease_full.arff` directly in CARE-KM-PAPER/DATA."
            )

        last_error = None
        for cmd in commands:
            try:
                proc = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=False, timeout=180
                )
                if proc.returncode == 0:
                    matches = list(td.rglob("chronic_kidney_disease_full.arff"))
                    if not matches:
                        matches = list(td.rglob("*.arff"))
                    if matches:
                        matches.sort(key=lambda p: ("full" not in p.name.lower(), len(str(p))))
                        return matches[0].read_bytes()
                last_error = proc.stderr.decode("utf-8", errors="replace")[-500:]
            except Exception as exc:
                last_error = repr(exc)

        raise RuntimeError(
            "Gagal mengekstrak RAR Chronic Kidney Disease. "
            f"Last extractor error: {last_error}"
        )


def load_chronic_kidney(path, data_dir=None):
    raw = _extract_ckd_arff_from_outer_zip(path, data_dir=data_dir)
    text = raw.decode("utf-8", errors="replace")
    rows = []
    in_data = False

    for source_line_no, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if s.lower() == "@data":
            in_data = True
            continue
        if not in_data or s.startswith("@"):
            continue

        fields = [v.strip() for v in s.split(",")]
        if len(fields) > 25:
            cleaned = [v for v in fields if v != ""]
            if len(cleaned) == 25:
                fields = cleaned
            else:
                while len(fields) > 25 and "" in fields:
                    fields.remove("")

        if len(fields) != 25:
            raise ValueError(
                "Chronic Kidney Disease: malformed ARFF record at source "
                f"line {source_line_no}; expected 25 fields, found {len(fields)}."
            )
        rows.append(fields)

    if not rows:
        raise ValueError("Chronic Kidney Disease: no data rows found in ARFF")

    df = pd.DataFrame(rows, columns=[f"a{i}" for i in range(1, 25)] + ["class"])
    df = _strip_frame(df)

    for col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace("\\t", "", regex=False)
            .str.replace("\t", "", regex=False)
            .str.strip()
        )

    df["class"] = df["class"].str.lower().str.replace(" ", "", regex=False)
    bad_class = ~df["class"].isin(["ckd", "notckd"])
    if bad_class.any():
        bad = sorted(df.loc[bad_class, "class"].unique().tolist())
        raise ValueError(f"Chronic Kidney Disease: unexpected class label(s): {bad}")

    return _auto_categorize_numeric(df.drop(columns="class")), df["class"]


def load_annealing(path):
    parts = []
    with zipfile.ZipFile(path) as zf:
        for suffix in ["anneal.data", "anneal.test"]:
            member = _find_member(zf, [suffix])
            d = _read_csv_member(
                zf, member, header=None, dtype=str, keep_default_na=False
            )
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    if df.shape[1] != 39:
        raise ValueError(f"Annealing: expected 39 columns, found {df.shape[1]}")
    df.columns = [f"a{i}" for i in range(1, 39)] + ["class"]
    return (
        _auto_categorize_numeric(_clean_missing_strings(df.drop(columns="class"))),
        df["class"],
    )


def load_statlog_heart(path):
    with zipfile.ZipFile(path) as zf:
        member = _find_member(zf, ["heart.dat"])
        df = pd.read_csv(
            io.BytesIO(_read_member_bytes(zf, member)),
            sep=r"\s+", header=None, dtype=str, engine="python"
        )
    if df.shape[1] != 14:
        raise ValueError(f"Statlog Heart: expected 14 columns, found {df.shape[1]}")
    df.columns = [f"a{i}" for i in range(1, 14)] + ["class"]
    return _auto_categorize_numeric(df.drop(columns="class")), df["class"]


def load_heart_disease(path):
    with zipfile.ZipFile(path) as zf:
        member = _find_member(zf, ["processed.cleveland.data"])
        df = _read_csv_member(
            zf, member, header=None, dtype=str, keep_default_na=False
        )
    if df.shape[1] != 14:
        raise ValueError(f"Heart Disease: expected 14 columns, found {df.shape[1]}")
    df.columns = [f"a{i}" for i in range(1, 14)] + ["class_raw"]
    y = _binary_presence_target(df["class_raw"])
    X = _auto_categorize_numeric(
        _clean_missing_strings(df.drop(columns="class_raw"))
    )
    return X, y


def load_thyroid_disease(path):
    with zipfile.ZipFile(path) as zf:
        member = _find_member(zf, ["new-thyroid.data", "new-thyroid.data.txt"])
        df = _read_csv_member(
            zf, member, header=None, dtype=str, keep_default_na=False
        )
    if df.shape[1] != 6:
        raise ValueError(
            f"Thyroid Disease/new-thyroid: expected 6 columns, found {df.shape[1]}"
        )
    df.columns = ["class"] + [f"a{i}" for i in range(1, 6)]
    return _auto_categorize_numeric(df.drop(columns="class")), df["class"]


def load_nursery(path):
    cols = [
        "parents", "has_nurs", "form", "children",
        "housing", "finance", "social", "health", "class",
    ]
    with zipfile.ZipFile(path) as zf:
        member = _find_member(zf, ["nursery.data", "/nursery.data"])
        df = _read_csv_member(
            zf, member, header=None, names=cols,
            dtype=str, keep_default_na=False
        )
    return df.drop(columns="class"), df["class"]


LOADERS = {
    "car": load_car,
    "lymphography": load_lymphography,
    "hayes_roth": load_hayes_roth,
    "australian_credit": load_australian_credit,
    "chronic_kidney": load_chronic_kidney,
    "annealing": load_annealing,
    "statlog_heart": load_statlog_heart,
    "heart_disease": load_heart_disease,
    "thyroid_disease": load_thyroid_disease,
    "nursery": load_nursery,
}


def resolve_dataset_keys(requested=None):
    if requested is None:
        env = os.environ.get("CARE_KM_GEN_DATASETS", "").strip()
        requested = env or None
    if requested is None:
        return DATASET_ORDER.copy()
    if isinstance(requested, str):
        keys = [x.strip() for x in requested.split(",") if x.strip()]
    else:
        keys = [str(x).strip() for x in requested if str(x).strip()]
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        raise KeyError(f"Unknown generalization dataset key(s): {unknown}")
    return keys


def load_dataset(data_dir, key):
    data_dir = Path(data_dir)
    if key not in DATASETS:
        raise KeyError(key)
    spec = DATASETS[key]
    path = data_dir / spec["file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset generalization tidak ditemukan: {path}"
        )
    loader = LOADERS[key]
    if key == "chronic_kidney":
        X, y = loader(path, data_dir=data_dir)
    else:
        X, y = loader(path)
    X = X.copy()
    y = pd.Series(y).copy()

    X.columns = [str(c).strip() for c in X.columns]
    for c in X.columns:
        X[c] = X[c].astype(str).str.strip()
    y = y.astype(str).str.strip()

    if len(X) != len(y):
        raise ValueError(f"{key}: X/y size mismatch")
    if len(X) == 0 or X.shape[1] == 0:
        raise ValueError(f"{key}: empty dataset")
    if y.nunique() < 2:
        raise ValueError(f"{key}: class count < 2")

    return X, y


def audit_dataset(data_dir, key):
    X, y = load_dataset(data_dir, key)
    spec = DATASETS[key]
    rows = int(len(X))
    features = int(X.shape[1])
    classes = int(y.nunique())
    row = {
        "dataset": key,
        "display": spec["display"],
        "source_file": spec["file"],
        "rows": rows,
        "features": features,
        "classes": classes,
        "reference_rows": int(spec["reference_rows"]),
        "reference_features": int(spec["reference_features"]),
        "reference_k": int(spec["reference_k"]),
        "missing_cells": int((X == "missing").sum().sum()),
        "duplicate_feature_rows": int(X.duplicated().sum()),
        "min_cardinality": int(min(X[c].nunique() for c in X.columns)),
        "max_cardinality": int(max(X[c].nunique() for c in X.columns)),
    }
    row["rows_match_reference"] = rows == row["reference_rows"]
    row["features_match_reference"] = features == row["reference_features"]
    row["k_match_reference"] = classes == row["reference_k"]
    row["metadata_match"] = bool(
        row["rows_match_reference"]
        and row["features_match_reference"]
        and row["k_match_reference"]
    )
    # Structural validity is what gates execution. Metadata reference fields
    # are reported separately and never silently alter the data.
    row["valid"] = bool(rows > 0 and features > 0 and classes >= 2)
    return row
