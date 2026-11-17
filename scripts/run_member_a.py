"""成员A 全部实验：零样本 + Linear Probe（2 预算 × 3 种子）。

用法（项目根目录）:
    python scripts/run_member_a.py              # 全部
    python scripts/run_member_a.py --dry-run    # 只打印将执行的命令
    python scripts/run_member_a.py --only zeroshot
    python scripts/run_member_a.py --only linearprobe
    python scripts/run_member_a.py --only linearprobe --budgets 10pct \
        --out outputs/step1_b64 --tag b64       # 只跑 10pct，产物单独放

前置: 已按 docs/step1_handoff.md 准备好
    config/step1_linear_probe.local.yaml   （本机路径，已 gitignore）
    data/*_manifest.csv 的 image_path 已指向本机

产物命名由 evaluate_model / fit_source 决定：
    checkpoint_linear_probe_{budget}_seed{seed}.pt
    predictions_experiment_{method}_{domain}_{split}_{budget}_seed{seed}.csv
    metrics_per_class_*.csv / metrics_summary_*.csv
    features_*.npz / metadata_*.json
    training_*.json / measurement_*.json / inference_*.json

零样本与预算无关，因此只跑一次（budget 记为 100pct）；表格里 10%/100% 两列
复用同一结果，请在报告里注明。

batch_size 按实验要求取 64（"AdamW，lr=1e-3，weight_decay=1e-4，batch=64"）。
主干全冻结，前向不建 autograd 图，64 张一批在 4GB 显存上可以跑。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config/step1_linear_probe.local.yaml"
SEEDS = (42, 43, 44)
ALL_BUDGETS = ("10pct", "100pct")
EVAL_SPLITS = (("mimic", "val"), ("mimic", "test"), ("chexpert", "test"))


def write_variant(path: Path, experiment: dict, output_dir: str | None) -> None:
    config = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    config["experiment"] = experiment
    if output_dir:
        config.setdefault("data", {})["output_dir"] = output_dir
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run(command: list[str], dry: bool, attempts: int = 3) -> None:
    """执行一条命令；失败时重试。

    这台机器 16GB 内存，长时间跑矩阵时会因瞬时内存峰值让子进程以
    0xC0000409（栈缓冲区溢出/快速失败）退出。实测单条命令重跑就能成功，
    所以默认重试 3 次，每次失败后等一会再试。
    """
    text = " ".join(command)
    print(f"\n$ {text}", flush=True)
    if dry:
        return
    for attempt in range(1, attempts + 1):
        start = time.time()
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode == 0:
            print(f"  完成，用时 {time.time() - start:.0f}s", flush=True)
            return
        if attempt == attempts:
            sys.exit(f"命令失败（退出码 {result.returncode}，已重试 {attempts} 次）: {text}")
        print(f"  失败（退出码 {result.returncode}），{30 * attempt}s 后重试 "
              f"{attempt}/{attempts - 1}", flush=True)
        time.sleep(30 * attempt)


def already_done(path: Path) -> bool:
    """断点续跑：产物存在且非空就跳过。"""
    return path.is_file() and path.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", choices=("zeroshot", "linearprobe"))
    parser.add_argument("--no-resume", action="store_true", help="忽略已有产物，全部重跑")
    parser.add_argument("--budgets", nargs="+", default=list(ALL_BUDGETS), choices=ALL_BUDGETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--batch-size", type=int, default=64, help="实验要求 batch=64")
    parser.add_argument("--out", help="覆盖 data.output_dir，便于把不同设置分开存放")
    args = parser.parse_args()
    if not args.dry_run and not BASE.is_file():
        sys.exit(f"缺少 {BASE}；先按 docs/step1_handoff.md 生成本机配置")

    python = sys.executable
    tmp = ROOT / "config/.tmp"
    out = ROOT / (args.out or "outputs/step1")
    if not args.dry_run:
        tmp.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume
    out_arg = args.out

    if args.only in (None, "zeroshot"):
        # 零样本：主干全冻结，Step 0 的 MedCLIPAdapter 直接可用，无需训练
        zero = tmp / "zeroshot.yaml"
        if not args.dry_run:
            write_variant(zero, {"method": "pretrained", "factory": None, "prompt_templates": 1,
                                 "seed": 42, "budget": "100pct",
                                 "batch_size": args.batch_size, "threshold": 0.5}, out_arg)
        for domain, split in EVAL_SPLITS:
            done = out / f"metrics_summary_experiment_pretrained_{domain}_{split}_100pct_seed42.csv"
            if resume and already_done(done):
                print(f"跳过（已完成）: {done.name}", flush=True)
                continue
            run([python, "scripts/step1_run.py", "--config", str(zero),
                 "--action", "eval", "--domain", domain, "--split", split,
                 "--measure-inference"], args.dry_run)

    if args.only in (None, "linearprobe"):
        for budget in args.budgets:
            for seed in args.seeds:
                variant = tmp / f"lp_{budget}_seed{seed}.yaml"
                if not args.dry_run:
                    write_variant(variant, {"method": "linear_probe",
                                            "factory": "src.step1_methods.linear_probe:build_model",
                                            "prompt_templates": 1, "seed": seed, "budget": budget,
                                            "batch_size": args.batch_size, "threshold": 0.5}, out_arg)
                # 训练：源域 train 子集，源域 val 早停选模型
                train_log = out / f"training_linear_probe_{budget}_seed{seed}.json"
                checkpoint = out / f"checkpoint_linear_probe_{budget}_seed{seed}.pt"
                if resume and already_done(train_log) and already_done(checkpoint):
                    print(f"跳过训练（已完成）: {train_log.name}", flush=True)
                else:
                    run([python, "scripts/step1_run.py", "--config", str(variant),
                         "--action", "train", "--domain", "mimic"], args.dry_run)
                # 评测：源域 val / 源域 test / 目标域 test，全部用同一份最佳 checkpoint
                for domain, split in EVAL_SPLITS:
                    done = out / f"metrics_summary_experiment_linear_probe_{domain}_{split}_{budget}_seed{seed}.csv"
                    if resume and already_done(done):
                        print(f"跳过（已完成）: {done.name}", flush=True)
                        continue
                    # 推理计时只在目标域 test 上测一次，避免重复占时间
                    extra = ["--measure-inference"] if (domain, split) == ("chexpert", "test") else []
                    run([python, "scripts/step1_run.py", "--config", str(variant),
                         "--action", "eval", "--domain", domain, "--split", split,
                         "--checkpoint", str(checkpoint)] + extra, args.dry_run)

    if not args.dry_run:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n全部完成" if not args.dry_run else "\n（dry-run，未执行）")


if __name__ == "__main__":
    main()
