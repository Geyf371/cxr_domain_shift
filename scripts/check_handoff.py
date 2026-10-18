"""Validate the two independently produced, non-patient-level Step 0 reports."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT, read_config, output_dir, write_json
from scripts.run_step0 import code_fingerprint, data_protocol

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/step0.local.yaml")
    args = parser.parse_args()
    out = output_dir(read_config(args.config))
    handoff = out / "step0_handoff.json"
    if handoff.exists():
        write_json(handoff, {"step0_ready": False, "status": "checking"})
    expected_protocol = data_protocol(read_config(ROOT / "config/step0.yaml"))
    current_code = code_fingerprint()
    reports = []
    for domain in ("mimic", "chexpert"):
        path = out / f"step0_{domain}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path.name}; the owner of {domain} must run run_step0.py and commit its report")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("domain") != domain or not report.get("domain_ready") or report.get("status") != "passed":
            raise ValueError(f"{domain} acceptance has not passed")
        if report["code_sha256"] != current_code:
            raise ValueError(f"{domain} report is from a different code version; rerun the model check")
        if report.get("data_protocol") != expected_protocol:
            raise ValueError(f"{domain} used different sampling settings from the shared config/step0.yaml")
        if not (out / f"class_frequencies_{domain}.csv").is_file():
            raise FileNotFoundError(f"Missing aggregate class frequencies for {domain}")
        reports.append(report)
    a, b = reports
    for key in ("classes", "model_feature_dim", "joint_feature_dim"):
        if a[key] != b[key]:
            raise ValueError(f"Domain protocol mismatch: {key}")
    if a["model"]["sha256"] != b["model"]["sha256"]:
        raise ValueError("Source and target used different model assets")
    write_json(handoff, {"step0_ready": True, "domains": ["mimic", "chexpert"],
               "code_sha256": a["code_sha256"], "model_weights_sha256": a["model"]["weights_sha256"],
               "data_protocol": expected_protocol,
               "scope": "Step 0 shared infrastructure and per-domain 16-image acceptance; later experiments remain with Step 1+"})
    print("PASS: Step 0 handoff ready. Commit outputs/step0_handoff.json.")

if __name__ == "__main__":
    main()
