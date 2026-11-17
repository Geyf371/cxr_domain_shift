"""Export all 12 completed LoRA runs without publishing private assets to Git."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    source, dest = args.source.resolve(), args.destination.resolve()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'step1-member-b':
        raise ValueError('Export from the committed step1-member-b branch')
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=ROOT, check=True)
    subprocess.run(['git', 'cat-file', '-e', 'HEAD:scripts/export_lora_delivery.py'], cwd=ROOT, check=True)
    if dest.exists() or source == dest or source in dest.parents:
        raise ValueError('Use a new destination outside the formal run tree')
    expected = [f'lora_r{rank}_{budget}_seed{seed}' for rank in (8, 16)
                for budget in ('10pct', '100pct') for seed in (42, 43, 44)]
    for name in expected:
        record = json.loads((source / name / 'completed.json').read_text())
        if record['status'] != 'passed' or set(record['final_test_domains']) != {'mimic', 'chexpert'}:
            raise ValueError(f'Incomplete run: {name}')
    dest.mkdir(parents=True)
    rows, runs = [], []

    def portable(value, target):
        if isinstance(value, dict):
            return {key: portable(item, target) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item, target) for item in value]
        if isinstance(value, str):
            if value.startswith(str(source) + '/'):
                mapped = dest / 'runs' / Path(value).relative_to(source)
                return os.path.relpath(mapped, target.parent)
            project = str(ROOT.parent)
            if value.startswith(project + '/'):
                return '<PROJECT_ROOT>/' + value[len(project) + 1:]
        return value

    for name in expected:
        src = source / name
        run = dest / 'runs' / name
        out = run / 'outputs'
        out.mkdir(parents=True)
        completed = json.loads((src / 'completed.json').read_text())
        integrity = json.loads((src / 'integrity.json').read_text())
        if not integrity['frozen_state_unchanged']:
            raise ValueError(f'Frozen state changed: {name}')
        rank, budget, seed = int(name.split('_')[1][1:]), name.split('_')[2], int(name.split('seed')[1])
        config = yaml.safe_load((ROOT / f'config/lora_r{rank}.yaml').read_text())
        actual = yaml.safe_load((src / 'config.yaml').read_text())
        # Keep actual experiment/training fields, omit only machine paths.
        config['experiment'] = actual['experiment']
        config['training'] = actual['training']
        config['lora'] = actual['lora']
        config['data']['output_dir'] = f'outputs/{name}'
        config['outputs']['output_dir'] = f'outputs/{name}'
        (run / 'config.portable.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        for domain in ('mimic', 'chexpert'):
            tag = f'experiment_lora_r{rank}_{domain}_test_{budget}_seed{seed}'
            prediction = pd.read_csv(src / 'outputs' / f'predictions_{tag}.csv')
            summary = pd.read_csv(src / 'outputs' / f'metrics_summary_{tag}.csv')
            with np.load(src / 'outputs' / f'features_{tag}.npz', allow_pickle=False) as cache:
                if list(cache['sample_id']) != list(prediction.sample_id):
                    raise ValueError(f'Feature/prediction identities differ: {name}/{domain}')
                for key, width in [('v', 512), ('q', 5), ('z', 517)]:
                    if cache[key].shape != (len(prediction), width) or not np.isfinite(cache[key]).all():
                        raise ValueError(f'Invalid features: {name}/{domain}/{key}')
            if len(summary) != 1 or summary.iloc[0]['scope'] != 'experiment':
                raise ValueError(f'Invalid summary: {name}/{domain}')
            row = summary.iloc[0].to_dict()
            rows.append({key: row[key] for key in ['method', 'budget', 'seed', 'domain', 'split', 'scope',
                         'macro_auc', 'macro_f1', 'macro_auprc', 'macro_auc_valid_classes', 'num_samples']})
        # Include outputs and evidence; never include source imagery or local config.
        files = list((src / 'outputs').iterdir())
        files += [src / n for n in ('completed.json', 'integrity.json', 'preflight.json', 'run.log')]
        for original in files:
            target = out / original.name if original.parent.name == 'outputs' else run / original.name
            if original.suffix == '.json':
                target.write_text(json.dumps(portable(json.loads(original.read_text()), target),
                                             ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            elif original.suffix == '.log':
                target.write_text(original.read_text().replace(str(ROOT.parent), '<PROJECT_ROOT>'), encoding='utf-8')
            else:
                shutil.copy2(original, target)
        checkpoint = Path(completed['training']['checkpoint'])
        if not checkpoint.is_file() or not (out / checkpoint.name).is_file():
            raise ValueError(f'Missing checkpoint: {name}')
        training = completed['training']
        best = max(training['history'], key=lambda item: item['val_macro_auc'])
        runs.append(dict(name=name, rank=rank, budget=budget, seed=seed,
                         checkpoint=f'runs/{name}/outputs/{checkpoint.name}',
                         epochs=len(training['history']), best_epoch=best['epoch'],
                         training_and_validation_seconds=training['elapsed_seconds'],
                         train_only_seconds=None, inference_1000_seconds=None,
                         peak_cuda_allocated_bytes=training['peak_cuda_memory_bytes']))
        print(f'Exported {name}', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(dest / 'test_metrics.csv', index=False)
    summary = frame.groupby(['method', 'budget', 'domain'])[['macro_auc', 'macro_f1', 'macro_auprc']].agg(['mean', 'std'])
    summary.to_csv(dest / 'test_metrics_mean_std.csv')
    validation = dest / 'validation'
    validation.mkdir()
    resources = source.parent
    evidence = [resources / 'validation/lora_forward_backward.json', resources / 'validation/lora_tests.xml',
                resources / 'validation/swin_attention_structure.json']
    diagnostics = resources / 'diagnostics'
    for directory in sorted(diagnostics.glob('lora_r8_128_seed42_*')):
        if (directory / 'completed.json').exists():
            evidence.extend([directory / 'selection.json', directory / 'selected_patients.csv',
                             Path(json.loads((directory / 'completed.json').read_text())['result'])])
            break
    for original in evidence:
        target = validation / original.name
        if original.suffix == '.json':
            data = portable(json.loads(original.read_text()), target)
            if original.name == 'result.json':
                data['checkpoint'] = 'DIAGNOSTIC_CHECKPOINT_NOT_INCLUDED'
            target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        else:
            shutil.copy2(original, target)
    shutil.copy2(source / 'lora_r8_10pct_seed42/pip-freeze.txt', dest / 'training_environment.freeze.txt')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'step1-member-b':
        raise ValueError('Export from the committed step1-member-b branch')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'bundle', 'create', str(dest / 'code.bundle'), 'step1-member-b'], cwd=ROOT, check=True)
    (dest / 'manifest.json').write_text(json.dumps(dict(schema='lora_delivery_v1', code_commit=commit,
        prompt_protocol='original Step0 single template; no clinical_v1 substitution', runs=runs,
        note='code_commit packages the implementation; training source hashes remain in per-run preflight.json'),
        indent=2) + '\n')
    shutil.copy2(ROOT / 'docs/lora_delivery.md', dest / 'README.md')
    checks = []
    for path in sorted(dest.rglob('*')):
        if path.is_file():
            checks.append(f'{digest(path)}  {path.relative_to(dest).as_posix()}')
    (dest / 'SHA256SUMS').write_text('\n'.join(checks) + '\n')
    print(f'PASS: 12 runs, 24 test artifacts, {len(checks)} checksummed files -> {dest}', flush=True)


if __name__ == '__main__':
    main()
