"""Verify real source-image LoRA forward/backward, without optimizer updates."""
import argparse
import gc
from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import read_config, project_path, seed_everything, write_json
from src.data import load_manifest, make_loader, CLASS_NAMES
from src.model_loader import load_medclip
from src.model_adapter import MedCLIPAdapter, build_model
from src.trainer import masked_bce_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/step0.local.yaml')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    torch.set_num_threads(4)
    data_dir = project_path(config['data']['data_dir'])
    frame = load_manifest(data_dir / 'mimic_manifest.csv')
    ids = pd.read_csv(data_dir / 'source_train_seed42_10pct.csv').patient_id
    valid = frame[[f'mask_{c}' for c in CLASS_NAMES]].sum(axis=1).gt(0)
    sample = frame[frame.patient_id.isin(ids) & frame.split.eq('train') & valid].sort_values('patient_id').head(2)
    assert len(sample) == 2
    results = []
    for rank in (8, 16):
        seed_everything(42)
        config['experiment'].update(method=f'lora_r{rank}', factory='src.lora:build_lora', prompt_templates=1)
        config['lora'] = dict(rank=rank, alpha=2 * rank, dropout=.05)
        bundle = load_medclip(config)
        batch = next(iter(make_loader(sample, bundle.processor, batch_size=2)))
        with torch.no_grad():
            baseline = MedCLIPAdapter(bundle).eval().forward_batch(batch).image_features_v.clone()
        model = build_model(bundle, config).eval()
        with torch.no_grad():
            output = model.forward_batch(batch)
            difference = float((output.image_features_v - baseline).abs().max())
            assert difference <= 1e-6 and output.logits.shape == (2, 5)
            assert torch.isfinite(output.logits).all()
        model.train()
        output = model.forward_batch(batch)
        device = output.logits.device
        loss = masked_bce_loss(output.logits, batch['labels'].to(device), batch['mask'].to(device))
        loss.backward()
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert all(p.grad.abs().sum() > 0 for n, p in trainable.items() if n.endswith('lora_b'))
        count = sum(p.numel() for p in trainable.values())
        assert count == 17664 * rank + 2590 and len(model.injected_paths) == 24
        results.append(dict(rank=rank, trainable_parameters=count, injected_paths=model.injected_paths,
                            initial_image_feature_max_abs=difference, loss=float(loss.detach()),
                            feature_shapes={k: list(getattr(output, k).shape) for k in
                                            ('logits', 'image_features_v', 'prompt_scores_q', 'joint_features_z')},
                            device=str(device), finite_gradients=True, frozen_parameters_without_grad=True))
        del output, model, bundle, trainable, baseline, loss, batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(args.output, dict(scope='two real source train images; backward only; no optimizer steps or epochs',
                                results=results))
    print(f'PASS: {args.output}')


if __name__ == '__main__':
    main()
