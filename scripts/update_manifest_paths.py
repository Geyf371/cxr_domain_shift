"""把两份 manifest 的 image_path 改到本机解压目录。

只改 image_path 一列；患者名单、标签、掩码、划分、六份嵌套子集名单原样保留。

与 docs/step1_handoff.md 第 71 行的 PowerShell 版相比，这一版先在内存里
检查完 5000/1000 张图全部存在才写回，任一缺失就整体中止、不留下改了一半的文件。

用法（项目根目录）:
    python scripts/update_manifest_paths.py
    python scripts/update_manifest_paths.py --config config/step1_linear_probe.local.yaml
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path, PureWindowsPath

import yaml

ROOT = Path(__file__).resolve().parents[1]

# 交接包里的原始路径前缀
OLD_ROOTS = {
    "mimic": PureWindowsPath("E:/data"),
    "chexpert": PureWindowsPath("D:/AI_Projects/CheXpert_Small"),
}
EXPECTED = {"mimic": 5000, "chexpert": 1000}


def resolve(row_path: str, new_root: Path, old_root: PureWindowsPath) -> Path:
    current = Path(row_path)
    if current.is_absolute() and current.is_relative_to(new_root):
        return current
    relative = PureWindowsPath(row_path).relative_to(old_root)
    return new_root.joinpath(*relative.parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/step1_linear_probe.local.yaml")
    args = parser.parse_args()
    config_path = ROOT / args.config
    if not config_path.is_file():
        sys.exit(f"缺少 {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_dir = ROOT / config["data"].get("data_dir", "data")

    pending = []
    for domain in ("mimic", "chexpert"):
        root_value = config["data"].get(f"{domain}_root")
        if not root_value:
            sys.exit(f"配置里 data.{domain}_root 为空")
        new_root = Path(root_value)
        manifest = data_dir / f"{domain}_manifest.csv"
        if not manifest.is_file():
            sys.exit(f"缺少 {manifest}")
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        if len(rows) != EXPECTED[domain]:
            sys.exit(f"{domain}: 期望 {EXPECTED[domain]} 行，实际 {len(rows)}")

        missing, changed = [], 0
        for row in rows:
            try:
                updated = resolve(row["image_path"], new_root, OLD_ROOTS[domain])
            except ValueError:
                missing.append(f"无法解析: {row['image_path']}")
                continue
            if not updated.is_file():
                missing.append(str(updated))
            if str(updated) != row["image_path"]:
                changed += 1
            row["image_path"] = str(updated)

        if missing:
            print(f"\n{domain}: {len(missing)} 张图缺失，前 5 个：", file=sys.stderr)
            for item in missing[:5]:
                print("  ", item, file=sys.stderr)
            sys.exit("中止：未写回任何文件")
        print(f"{domain}: {len(rows)} 张图全部存在，待改 {changed} 行")
        pending.append((manifest, fields, rows, changed))

    for manifest, fields, rows, changed in pending:
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"OK: {manifest.name} 已更新 {changed} 行")


if __name__ == "__main__":
    main()
