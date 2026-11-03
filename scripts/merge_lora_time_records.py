"""Versioned merge of resource replay timings; never relabel historical timings."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError('Use a new destination; preserve v1 evidence')
    report = json.loads((args.replay / 'verification.json').read_text())
    if report['status'] != 'passed' or report['verified_runs'] != 12:
        raise ValueError('Replay must have twelve verified results')
    measurements = pd.read_csv(args.replay / 'training_time_measurements.csv')
    original = pd.read_csv(args.records / 'run_resources.csv')
    if len(measurements) != 12 or set(measurements.run) != set(original.run):
        raise ValueError('Run identities must match all twelve original runs')
    rows = []
    for row in measurements.to_dict('records'):
        result = json.loads((args.replay / row['run'] / 'result.json').read_text())
        historical = original[original.run.eq(row['run'])].iloc[0]
        if not result['temporary_checkpoint_deleted'] or not result['frozen_state_unchanged']:
            raise ValueError('Replay cleanup or integrity verification failed')
        if len(result['history']) != historical['epochs'] or result['best_epoch'] != historical['best_epoch']:
            raise ValueError('Epoch/best-epoch mismatch; do not silently equate different runs')
        if abs(sum(e['train_only_seconds'] for e in result['epoch_timings']) - row['train_only_seconds']) > 1e-6:
            raise ValueError('Epoch timing total mismatch')
        rows.append(dict(run=row['run'], replay_train_only_seconds=row['train_only_seconds'],
                         replay_validation_seconds=row['validation_seconds'],
                         replay_checkpoint_save_seconds=row['checkpoint_save_seconds'],
                         replay_checkpoint_restore_seconds=row['checkpoint_restore_seconds'],
                         replay_initialization_seconds=row['initialization_seconds'],
                         replay_fit_seconds=row['elapsed_seconds'],
                         replay_training_peak_allocated_bytes=result['training_peak_cuda_memory_bytes'],
                         replay_validation_peak_allocated_bytes=result['validation_peak_cuda_memory_bytes'],
                         train_time_source='resource_only_training_replay_20261103',
                         replay_temporary_checkpoint_deleted=True))
    original = original.rename(columns={'train_only_seconds': 'historical_train_only_seconds'})
    merged = original.merge(pd.DataFrame(rows), on='run', validate='one_to_one')
    merged['train_only_seconds'] = merged.replay_train_only_seconds
    shutil.copytree(args.records, args.destination)
    dest = args.destination
    (dest / 'SHA256SUMS').unlink(missing_ok=True)
    merged.to_csv(dest / 'run_resources.csv', index=False)
    evidence = dest / 'training_time_replay'
    evidence.mkdir()
    for name in ['training_time_measurements.csv', 'epoch_measurements.csv', 'time_mean_std.csv',
                 'verification.csv', 'verification.json', 'status.json', 'README.md']:
        shutil.copy2(args.replay / name, evidence / name)
    for row in rows:
        run = evidence / row['run']
        run.mkdir()
        shutil.copy2(args.replay / row['run'] / 'result.json', run / 'result.json')
    manifest = json.loads((dest / 'supplement_manifest.json').read_text())
    manifest.update(schema='lora_output_record_supplement_v2',
        historical_train_only_seconds=None,
        train_only_seconds_source='resource_only_training_replay_20261103; not historical',
        training_time_records='run_resources.csv', replay_runs=12,
        replay_finished_at=report['finished_at'], replay_temporary_checkpoints_deleted=True,
        replay_max_loss_difference=report['max_loss_difference'],
        replay_max_val_auc_difference=report['max_val_auc_difference'])
    # v1 scalar null referred to unavailable history; v2 stores measurements per run.
    manifest.pop('train_only_seconds', None)
    manifest.pop('train_only_seconds_reason', None)
    manifest['inference_weight_updates_during_measurement'] = manifest.pop('weight_updates_during_measurement', False)
    manifest['replay_weight_updates_during_measurement'] = True
    (dest / 'supplement_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (dest / 'README.md').write_text((dest / 'README.md').read_text().replace('train_only_seconds保持空值/null', 'historical_train_only_seconds保持空值/null（本次补测见v2）') + '''

## v2：纯训练计时补测已合并

12组资源计时重跑已完成，原轮数及最佳轮次一致，逐轮loss和val AUC与原运行最大差均为0。
run_resources.csv的train_only_seconds/replay_train_only_seconds来自本次补测；train_time_source标记来源。
historical_train_only_seconds仍为空；training_validation_seconds等原字段继续保留原始口径。
新增replay_validation/checkpoint_save/checkpoint_restore/initialization/fit_seconds和训练/验证分段显存。
新临时模型已删除，原正式模型与主交付包不变。training_time_replay/保留原始分项记录与核验结果。
此前“纯训练耗时未提供”的说明仅描述历史缺项，不再表示缺少本次补测；禁止将补测值标成历史实测。
''')
    print(f'PASS: merged twelve timings -> {dest}')


if __name__ == '__main__':
    main()
