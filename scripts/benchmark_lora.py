"""Time every completed LoRA checkpoint on all 1000 fixed target images."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import read_config, seed_everything, write_json
from src.data import load_manifest, make_loader
from src.model_loader import load_medclip
from src.model_adapter import build_model
from src.trainer import load_checkpoint


def state_digest(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-name', help='Optional exact run name for isolated-process measurement')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; never overwrite benchmark evidence')
    args.output.mkdir(parents=True)
    runs = sorted(p for p in args.formal_root.glob('lora_r*') if (p / 'completed.json').is_file())
    if args.run_name:
        runs = [run for run in runs if run.name == args.run_name]
    if len(runs) != (1 if args.run_name else 12):
        raise ValueError('Expected all twelve completed formal runs')
    raw = []
    for run in runs:
        config = read_config(run / 'config.yaml')
        config['model']['device'] = 'cuda'
        seed_everything(42)
        torch.set_num_threads(4)
        record = json.loads((run / 'completed.json').read_text())
        checkpoint = Path(record['training']['checkpoint'])
        frame = load_manifest(Path(config['data']['data_dir']) / 'chexpert_manifest.csv')
        assert len(frame) == frame.patient_id.nunique() == 1000
        assert frame.split.eq('test').all() and frame.domain.eq('chexpert').all()
        frame = frame.sort_values(['patient_id', 'study_id', 'image_id'])
        expected = frame.apply(lambda r: '|'.join(str(r[k]) for k in ('domain', 'patient_id', 'study_id', 'image_id')), axis=1).tolist()
        bundle = load_medclip(config)
        model = build_model(bundle, config).eval()
        load_checkpoint(checkpoint, model, 'cuda')
        gc.collect()
        torch.cuda.empty_cache()
        before = state_digest(model)
        loader = lambda: make_loader(frame, bundle.processor, batch_size=16, shuffle=False, seed=42)
        # Warm exactly five full batches. Labels never control timing selection.
        with torch.no_grad():
            for index, batch in enumerate(loader()):
                batch['pixel_values'] = batch['pixel_values'].cuda()
                model.forward_batch(batch)
                if index == 4:
                    break
        del batch
        torch.cuda.synchronize()
        run_rows = []
        reference = pd.read_csv(run / 'outputs' / f"predictions_experiment_{config['experiment']['method']}_chexpert_test_{config['experiment']['budget']}_seed{config['experiment']['seed']}.csv")
        columns = [c for c in reference.columns if c.startswith('prob_')]
        for repeat in range(1, 4):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            model_seconds = 0.0
            identities, probabilities = [], []
            with torch.no_grad():
                for batch in loader():
                    batch['pixel_values'] = batch['pixel_values'].cuda()
                    torch.cuda.synchronize()
                    forward_started = time.perf_counter()
                    output = model.forward_batch(batch)
                    torch.cuda.synchronize()
                    model_seconds += time.perf_counter() - forward_started
                    assert output.logits.shape == (len(batch['sample_id']), 5)
                    assert torch.isfinite(output.logits).all()
                    probabilities.append(output.logits.sigmoid().cpu().numpy())
                    identities.extend(batch['sample_id'])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert identities == expected and len(set(identities)) == 1000
            values = np.concatenate(probabilities)
            lookup = {key: i for i, key in enumerate(identities)}
            ordered = values[[lookup[key] for key in reference.sample_id]]
            difference = float(np.max(np.abs(ordered - reference[columns].to_numpy())))
            assert difference <= 1e-5, 'Benchmark predictions changed from the formal checkpoint evaluation'
            row = dict(run=run.name, method=config['experiment']['method'], budget=config['experiment']['budget'],
                       seed=config['experiment']['seed'], repeat=repeat, images=1000, batches=63,
                       batch_size=16, precision='FP32', model_seconds=model_seconds,
                       end_to_end_seconds=elapsed, model_images_per_second=1000/model_seconds,
                       end_to_end_images_per_second=1000/elapsed,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                       formal_valid_predictions_max_abs=difference)
            run_rows.append(row)
            raw.append(row)
            pd.DataFrame(raw).to_csv(args.output / 'inference_repeats.csv', index=False)
            print(json.dumps(row), flush=True)
        assert state_digest(model) == before, 'Inference changed model weights or buffers'
        write_json(args.output / f'{run.name}.json', dict(status='passed', checkpoint=checkpoint.name,
                   model_state_unchanged=True, state_sha256=before, repeats=run_rows,
                   cuda_runtime=torch.version.cuda, torch_version=str(torch.__version__),
                   gpu=torch.cuda.get_device_name(), dtype=str(next(model.parameters()).dtype),
                   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
        del model, bundle, output, batch, probabilities, values, reference
        gc.collect()
        torch.cuda.empty_cache()
    result = pd.DataFrame(raw)
    result.groupby(['run', 'method', 'budget', 'seed'])[['model_seconds', 'end_to_end_seconds',
        'model_images_per_second', 'end_to_end_images_per_second', 'peak_cuda_allocated_bytes']].agg(['mean', 'std']).to_csv(args.output / 'inference_mean_std.csv')
    write_json(args.output / 'completed.json', dict(status='passed', checkpoints=len(runs), repeats=3*len(runs),
        images_per_repeat=1000, warmup_images_per_checkpoint=80, labels_used_for_filtering=False,
        scope='timing only; target labels not used for model selection or tuning',
        model_timing='forward_batch after synchronous host-to-device transfer; excludes sigmoid and collection',
        end_to_end_timing='loading/preprocessing/H2D/forward/sigmoid/CPU collection; excludes model loading and disk export',
        cache_policy='no OS cache flushing; repeated warm measurements'))


if __name__ == '__main__':
    main()

