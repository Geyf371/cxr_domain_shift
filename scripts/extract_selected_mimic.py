"""Extract only the JPGs selected in mimic_selection.csv from p10-p19 archives.

Supports ZIP and tar archives (including compressed tar variants). It never
extracts an archive wholesale. The expected MIMIC layout is preserved under
<mimic_root>/files/ so the manifest builder can resolve the images.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _archive_bucket(path: Path) -> str | None:
    match = re.search(r"(?:^|[._-])(p1[0-9])(?:$|[._-])", path.name.lower())
    return match.group(1) if match else None


def _normalise_member(name: str) -> str:
    return "/".join(part for part in name.replace("\\", "/").split("/") if part not in ("", "."))


def _matching_request(member_name: str, requested: dict[str, Path], bucket: str) -> str | None:
    parts = _normalise_member(member_name).split("/")
    for index, part in enumerate(parts):
        if part == bucket:
            relative_path = "/".join(parts[index:])
            if relative_path in requested:
                return relative_path
    return None


def _copy_member(source, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # An interrupted copy must not leave a truncated JPG that appears complete.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as output:
            temporary_path = Path(output.name)
            shutil.copyfileobj(source, output)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _extract_zip(archive: Path, requested: dict[str, Path], bucket: str) -> set[str]:
    extracted: set[str] = set()
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            if info.is_dir():
                continue
            relative_path = _matching_request(info.filename, requested, bucket)
            if relative_path is not None and relative_path not in extracted:
                with handle.open(info) as source:
                    _copy_member(source, requested[relative_path])
                extracted.add(relative_path)
                if len(extracted) == len(requested):
                    break
    return extracted


def _extract_tar(archive: Path, requested: dict[str, Path], bucket: str) -> set[str]:
    extracted: set[str] = set()
    # Streaming mode avoids loading a large tar index or expanding the archive.
    with tarfile.open(archive, mode="r|*") as handle:
        for member in handle:
            if not member.isfile():
                continue
            relative_path = _matching_request(member.name, requested, bucket)
            if relative_path is not None and relative_path not in extracted:
                source = handle.extractfile(member)
                if source is None:
                    continue
                with source:
                    _copy_member(source, requested[relative_path])
                extracted.add(relative_path)
                if len(extracted) == len(requested):
                    return extracted
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract selected MIMIC JPGs without expanding all p10-p19 archives")
    parser.add_argument("--config", default="config/step0.local.yaml")
    parser.add_argument("--manifest", default=None, help="Selection CSV; defaults to data/mimic_selection.csv")
    parser.add_argument("--archive-dir", default=None, help="Defaults to <mimic_root>/file")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_cfg = config["data"]
    if not data_cfg.get("mimic_root"):
        raise ValueError("Set data.mimic_root in the local config first")
    mimic_root = Path(data_cfg["mimic_root"]).resolve()
    manifest_path = Path(args.manifest) if args.manifest else Path(data_cfg.get("data_dir", "data")) / "mimic_selection.csv"
    archive_dir = Path(args.archive_dir) if args.archive_dir else mimic_root / "file"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Build a metadata selection first: {manifest_path}")
    if not archive_dir.exists():
        raise FileNotFoundError(f"Archive directory does not exist: {archive_dir}")

    frame = pd.read_csv(manifest_path)
    if frame.empty or not frame["domain"].eq("mimic").all():
        raise ValueError(f"Expected a non-empty MIMIC-only manifest: {manifest_path}")

    image_root = mimic_root / "files"
    requests: dict[str, dict[str, Path]] = {}
    already_present = 0
    for path_value in frame["image_path"].astype(str):
        image_path = Path(path_value)
        if image_path.is_file():
            already_present += 1
            continue
        try:
            relative_path = image_path.resolve().relative_to(image_root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Image path is outside the expected directory {image_root}: {image_path}"
            ) from exc
        parts = relative_path.split("/")
        if len(parts) < 4 or not re.fullmatch(r"p1[0-9]", parts[0]):
            raise ValueError(f"Unexpected MIMIC image path in manifest: {relative_path}")
        requests.setdefault(parts[0], {})[relative_path] = image_path

    if not requests:
        print(f"All {already_present} selected MIMIC images are already present.")
        return

    archives: dict[str, list[Path]] = {}
    for path in archive_dir.rglob("*"):
        if path.is_file() and (path.suffix.lower() in {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz"} or not path.suffix):
            bucket = _archive_bucket(path)
            if bucket:
                archives.setdefault(bucket, []).append(path)

    missing: set[str] = set()
    for bucket, wanted in sorted(requests.items()):
        candidates = sorted(archives.get(bucket, []), key=lambda item: (len(item.parts), str(item).lower()))
        if not candidates:
            missing.update(wanted)
            print(f"No archive found for {bucket}; {len(wanted)} images remain missing.")
            continue
        remaining = set(wanted)
        for archive in candidates:
            if not remaining:
                break
            try:
                active = {path: wanted[path] for path in remaining}
                if zipfile.is_zipfile(archive):
                    extracted = _extract_zip(archive, active, bucket)
                else:
                    extracted = _extract_tar(archive, active, bucket)
            except (tarfile.TarError, zipfile.BadZipFile, OSError) as exc:
                raise RuntimeError(f"Could not read archive {archive}: {exc}") from exc
            remaining.difference_update(extracted)
            print(f"{archive.name}: extracted {len(extracted)} selected JPGs")
        missing.update(remaining)

    extracted_count = sum(len(paths) for paths in requests.values()) - len(missing)
    print(f"Extracted {extracted_count} selected JPGs; {already_present} were already present.")
    if missing:
        examples = sorted(missing)[:5]
        raise FileNotFoundError(
            f"Could not find {len(missing)} selected JPGs in the matching archives. "
            f"Example archive member paths: {examples}"
        )


if __name__ == "__main__":
    main()
