"""Build MIMIC-CXR-JPG and CheXpert Step 0 manifests.

The script expects the standard public layouts, but discovers metadata files
recursively so the downloaded root can be configured without hard-coded paths.
Raw images remain outside the repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CLASS_NAMES, normalise_label, write_class_frequencies, resolve_label_columns


def _find_file(root: Path, names: list[str]) -> Path:
    for name in names:
        if (root / name).is_file():
            return root / name
    candidates: list[Path] = []
    for name in names:
        candidates.extend(root.rglob(name))
    if not candidates:
        raise FileNotFoundError(f"Could not find any of {names} below {root}")
    return sorted(set(candidates), key=lambda p: (len(p.parts), str(p).lower()))[0]


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer", low_memory=False)


def _mimic_image_path(root: Path, subject_id: int, study_id: int, dicom_id: str) -> Path:
    subject = f"p{int(subject_id)}"
    study = f"s{int(study_id)}"
    for base in (root / "files", root / "file", root):
        directory = base / subject[:3] / subject / study
        exact = directory / f"{dicom_id}.jpg"
        if exact.exists():
            return exact
    return root / "files" / subject[:3] / subject / study / f"{dicom_id}.jpg"


def _resolve_chexpert_path(root: Path, raw_path: object) -> Path:
    value = str(raw_path).replace("\\", "/")
    parts = Path(value).parts
    candidates = [root / value]
    if parts and parts[0].lower().startswith("chexpert"):
        candidates.append(root / Path(*parts[1:]))
    candidates.append(root.parent / value)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _label_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = resolve_label_columns(frame)
    for class_name in CLASS_NAMES:
        raw = frame[mapping[class_name]]
        mapped = raw.map(normalise_label)
        frame[f"raw_{class_name}"] = raw
        frame[f"label_{class_name}"] = mapped.map(lambda pair: pair[0])
        frame[f"mask_{class_name}"] = mapped.map(lambda pair: pair[1])
    return frame


def _finalise(frame: pd.DataFrame, *, require_images: bool = True) -> pd.DataFrame:
    columns = [
        "domain", "source", "patient_id", "study_id", "image_id",
        "image_path", "view_position", "split",
    ]
    columns += [f"raw_{name}" for name in CLASS_NAMES]
    columns += [f"label_{name}" for name in CLASS_NAMES]
    columns += [f"mask_{name}" for name in CLASS_NAMES]
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    result = frame[columns].sort_values(["patient_id", "study_id", "image_id"]).reset_index(drop=True)
    result["image_exists"] = result["image_path"].map(lambda path: Path(path).is_file())
    missing_images = result.loc[~result["image_exists"], "image_path"].tolist()
    if require_images and missing_images:
        raise FileNotFoundError(
            "The selected JPG paths do not exist. Check the configured data root "
            "and extract the selected images before building a manifest; "
            f"first missing paths: {missing_images[:3]}"
        )
    return result


def _write_domain_outputs(
    frame: pd.DataFrame, data_dir: Path, domain: str, *, selection_only: bool = False
) -> None:
    """Write one domain's manifest and class-frequency table."""

    data_dir.mkdir(parents=True, exist_ok=True)
    suffix = "selection" if selection_only else "manifest"
    frame.to_csv(data_dir / f"{domain}_{suffix}.csv", index=False)
    frequency_suffix = f"{domain}_selection" if selection_only else domain
    write_class_frequencies(frame, data_dir / f"class_frequencies_{frequency_suffix}.csv")


def build_mimic(
    root: Path,
    n_patients: int,
    selection_seed: int,
    split_seed: int,
    *,
    require_images: bool = True,
    subject_buckets: list[str] | None = None,
) -> pd.DataFrame:
    metadata_path = _find_file(
        root,
        [
            "mimic-cxr-metadata.csv.gz",
            "mimic-cxr-metadata.csv",
            "mimic-cxr-2.0.0-metadata.csv.gz",
            "mimic-cxr-2.0.0-metadata.csv",
        ],
    )
    labels_path = _find_file(root, ["mimic-cxr-2.0.0-chexpert.csv.gz", "mimic-cxr-2.0.0-chexpert.csv"])
    metadata = _read_table(metadata_path)
    labels = _read_table(labels_path)
    required = {"subject_id", "study_id", "dicom_id", "ViewPosition"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"MIMIC metadata is missing columns: {missing}")
    metadata = metadata[metadata["ViewPosition"].astype(str).str.upper().isin({"AP", "PA"})].copy()
    metadata["subject_id"] = pd.to_numeric(metadata["subject_id"], errors="coerce")
    metadata["study_id"] = pd.to_numeric(metadata["study_id"], errors="coerce")
    metadata = metadata.dropna(subset=["subject_id", "study_id", "dicom_id"])
    metadata["subject_id"] = metadata["subject_id"].astype(int)
    metadata["study_id"] = metadata["study_id"].astype(int)
    if subject_buckets:
        buckets = {value.lower() for value in subject_buckets}
        metadata = metadata[
            metadata["subject_id"].map(lambda value: f"p{int(value)}"[:3].lower()).isin(buckets)
        ].copy()
        if metadata.empty:
            raise ValueError(f"No AP/PA MIMIC records found for patient buckets: {sorted(buckets)}")
    mapping = resolve_label_columns(labels)
    labels = labels.rename(columns={value: key for key, value in mapping.items()})
    label_keys = ["subject_id", "study_id"] + CLASS_NAMES
    metadata = metadata.merge(labels[label_keys], on=["subject_id", "study_id"], how="left")
    metadata = metadata.sort_values(["subject_id", "study_id", "dicom_id"]).groupby("subject_id", as_index=False).head(1).copy()
    eligible = np.array(sorted(metadata["subject_id"].unique()))
    if len(eligible) < n_patients:
        raise ValueError(f"MIMIC has {len(eligible)} eligible patients, fewer than {n_patients}")
    rng = np.random.default_rng(selection_seed)
    selected_patients = np.sort(rng.choice(eligible, size=n_patients, replace=False))
    frame = metadata[metadata["subject_id"].isin(selected_patients)].copy()
    split_rng = np.random.default_rng(split_seed)
    shuffled = selected_patients.copy()
    split_rng.shuffle(shuffled)
    n_train = int(round(n_patients * 0.8))
    n_val = int(round(n_patients * 0.1))
    split_map = {int(value): "train" for value in shuffled[:n_train]}
    split_map.update({int(value): "val" for value in shuffled[n_train:n_train + n_val]})
    split_map.update({int(value): "test" for value in shuffled[n_train + n_val:]})
    frame["domain"] = "mimic"
    frame["source"] = "MIMIC-CXR-JPG-2.1.0"
    frame["patient_id"] = frame["subject_id"].map(lambda value: f"mimic:p{int(value)}")
    frame["study_id"] = frame["study_id"].map(lambda value: f"s{int(value)}")
    frame["image_id"] = frame["dicom_id"].astype(str)
    frame["image_path"] = [
        str(_mimic_image_path(root, row.subject_id, int(str(row.study_id).lstrip("s")), row.dicom_id))
        for row in frame.itertuples()
    ]
    frame["view_position"] = frame["ViewPosition"].astype(str).str.upper()
    frame["split"] = frame["subject_id"].map(split_map)
    return _finalise(_label_columns(frame), require_images=require_images)


def build_chexpert(
    root: Path,
    n_patients: int,
    selection_seed: int,
    *,
    require_images: bool = True,
) -> pd.DataFrame:
    train_path = _find_file(root, ["train.csv"])
    frame = _read_table(train_path)
    required = {"Path", "Frontal/Lateral", "AP/PA"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CheXpert train.csv is missing: {sorted(required.difference(frame.columns))}")
    if "Frontal/Lateral" in frame.columns:
        frame = frame[frame["Frontal/Lateral"].astype(str).str.lower().eq("frontal")].copy()
    if "AP/PA" in frame.columns:
        frame = frame[frame["AP/PA"].astype(str).str.upper().isin({"AP", "PA"})].copy()
    if "Path" not in frame.columns:
        raise ValueError("CheXpert train.csv must contain a Path column")
    frame["patient_num"] = frame["Path"].astype(str).str.extract(r"patient(\d+)", expand=False)
    frame["study_num"] = frame["Path"].astype(str).str.extract(r"study(\d+)", expand=False).fillna("0")
    frame = frame.dropna(subset=["patient_num"])
    frame["patient_num"] = frame["patient_num"].astype(int)
    frame["study_num"] = pd.to_numeric(frame["study_num"], errors="coerce").fillna(0).astype(int)
    frame = frame.sort_values(["patient_num", "study_num", "Path"]).groupby("patient_num", as_index=False).head(1).copy()
    eligible = np.array(sorted(frame["patient_num"].unique()))
    if len(eligible) < n_patients:
        raise ValueError(f"CheXpert has {len(eligible)} eligible patients, fewer than {n_patients}")
    rng = np.random.default_rng(selection_seed)
    selected = np.sort(rng.choice(eligible, size=n_patients, replace=False))
    frame = frame[frame["patient_num"].isin(selected)].copy()
    frame["domain"] = "chexpert"
    frame["source"] = "Stanford-CheXpert"
    frame["patient_id"] = frame["patient_num"].map(lambda value: f"chexpert:patient{int(value):05d}")
    frame["study_id"] = frame["study_num"].map(lambda value: f"study{int(value)}")
    frame["image_id"] = frame["Path"].astype(str).map(lambda value: Path(value).stem)
    frame["image_path"] = frame["Path"].map(lambda value: str(_resolve_chexpert_path(root, value)))
    frame["view_position"] = frame["AP/PA"].astype(str).str.upper() if "AP/PA" in frame.columns else ""
    frame["split"] = "test"
    return _finalise(_label_columns(frame), require_images=require_images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Step 0 manifests")
    parser.add_argument("--config", default="config/step0.yaml")
    parser.add_argument(
        "--domain",
        choices=("mimic", "chexpert", "both"),
        default="both",
        help="Build one domain independently, or both when both roots are available.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help=(
            "Write <domain>_selection.csv with expected paths while JPGs are unavailable. "
            "This does not write a runnable manifest or overwrite an existing one."
        ),
    )
    parser.add_argument(
        "--mimic-buckets",
        nargs="+",
        choices=tuple(f"p1{digit}" for digit in range(10)),
        default=None,
        help=(
            "Restrict MIMIC selection to patient-ID buckets represented by the available image shards, "
            "for example p10. This defines a shard subcohort; it is not a full-cohort sample."
        ),
    )
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_cfg = config["data"]
    split_cfg = config["split"]
    if [split_cfg.get(k) for k in ("train_ratio", "val_ratio", "test_ratio")] != [0.8, 0.1, 0.1]:
        raise ValueError("This assignment fixes the source split to 8:1:1")
    mimic_root = Path(data_cfg["mimic_root"]) if data_cfg.get("mimic_root") else None
    chexpert_root = Path(data_cfg["chexpert_root"]) if data_cfg.get("chexpert_root") else None
    if args.domain in ("mimic", "both") and (mimic_root is None or not mimic_root.exists()):
        raise FileNotFoundError(f"MIMIC root does not exist: {mimic_root}")
    if args.domain in ("chexpert", "both") and (chexpert_root is None or not chexpert_root.exists()):
        raise FileNotFoundError(f"CheXpert root does not exist: {chexpert_root}")
    data_dir = Path(data_cfg.get("data_dir", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    if args.domain in ("mimic", "both"):
        mimic = build_mimic(
            mimic_root,
            int(data_cfg["source_patients"]),
            int(data_cfg["source_selection_seed"]),
            int(data_cfg["split_seed"]),
            require_images=not args.allow_missing_images,
            subject_buckets=args.mimic_buckets,
        )
        _write_domain_outputs(mimic, data_dir, "mimic", selection_only=args.allow_missing_images)
        frames.append(mimic)
        print(f"Wrote {len(mimic)} MIMIC rows")
    if args.domain in ("chexpert", "both"):
        chexpert = build_chexpert(
            chexpert_root,
            int(data_cfg["target_patients"]),
            int(data_cfg["source_selection_seed"]),
            require_images=not args.allow_missing_images,
        )
        _write_domain_outputs(chexpert, data_dir, "chexpert", selection_only=args.allow_missing_images)
        frames.append(chexpert)
        print(f"Wrote {len(chexpert)} CheXpert rows")
    if args.allow_missing_images:
        print("Wrote metadata-only *_selection.csv files. Extract JPGs and rebuild without --allow-missing-images before image tests.")
    elif len(frames) == 2:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(data_dir / "all_manifest.csv", index=False)
        write_class_frequencies(combined, data_dir / "class_frequencies.csv")
        print(f"Wrote combined manifest with {len(combined)} rows")


if __name__ == "__main__":
    main()

