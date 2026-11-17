"""Behavior checks independent of patient data and downloaded model assets."""
from types import SimpleNamespace
import copy

import pytest
import torch
from torch import nn

from src.lora import LoRALinear, LoRAMedCLIPAdapter, inject_qv
from src.model_adapter import build_model
from src.model_interface import ModelBatchOutput
from src.trainer import train_source_epoch, save_checkpoint, load_checkpoint, fit_source, masked_bce_loss


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        self.block.attention = nn.Module()
        attention = nn.Module()
        self.block.attention.add_module('self', attention)
        for key in ('query', 'key', 'value'):
            setattr(attention, key, nn.Linear(3, 3))
        self.projection_head = nn.Linear(3, 4)

    def forward(self, x):
        a = self.block.attention.self
        return self.projection_head(torch.tanh(a.query(x) * a.key(x) + a.value(x)))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = Vision()
        self.embedding = nn.Embedding(10, 4)
        self.logit_scale = nn.Parameter(torch.tensor(2.))

    def text_model(self, ids, attention):
        return self.embedding(ids[:, 0])


def bundle():
    torch.manual_seed(123)
    tokenizer = lambda texts, **kwargs: {'input_ids': torch.arange(len(texts)).view(-1, 1),
                                       'attention_mask': torch.ones(len(texts), 1)}
    return SimpleNamespace(model=Backbone(), vision_feature_dim=4,
                           processor=SimpleNamespace(tokenizer=tokenizer))


def config(rank=8):
    return {'experiment': {'method': f'lora_r{rank}', 'factory': 'src.lora:build_lora',
                            'seed': 42, 'budget': '10pct'},
            'lora': {'rank': rank, 'alpha': 2 * rank, 'dropout': .05},
            'training': {'epochs': 8, 'patience': 2}}


def batch(n=3):
    return {'pixel_values': torch.arange(n*3).float().reshape(n, 3) / 7,
            'labels': torch.ones(n, 5), 'mask': torch.ones(n, 5), 'domain': ['mimic'] * n}


@pytest.mark.parametrize('rank', [8, 16])
def test_adapter_freezing_updates_groups_and_checkpoint(rank, tmp_path):
    model = build_model(bundle(), config(rank))
    assert isinstance(model, LoRAMedCLIPAdapter)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 12 * rank + 50
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    buffers = {n: p.clone() for n, p in model.named_buffers()}
    low_rank_before = {n: p.clone() for n, p in model.named_parameters() if 'lora_' in n}
    groups = model.optimizer_groups(config(rank))
    assert [g['lr'] for g in groups] == [1e-4, 1e-3]
    grouped = [id(p) for g in groups for p in g['params']]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(p) for p in model.parameters() if p.requires_grad}
    model.train()
    assert not model.backbone.training and model.head.training
    assert all(m.dropout.training for m in model.modules() if isinstance(m, LoRALinear))
    optimizer = torch.optim.AdamW(groups, weight_decay=0)
    for _ in range(3):
        train_source_epoch(model, [batch()], optimizer, 'cpu')
    for n, p in model.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), n
        if n in low_rank_before:
            assert not torch.equal(p, low_rank_before[n]), n
    assert all(torch.equal(b, buffers[n]) for n, b in model.named_buffers())
    data = batch()
    loss = masked_bce_loss(model.forward_batch(data).logits, data['labels'], data['mask'])
    loss.backward()
    for name, parameter in model.named_parameters():
        if 'lora_' in name:
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0, name
    model.eval()
    output = model.forward_batch(batch())
    assert output.logits.shape == (3, 5) and torch.isfinite(output.logits).all()
    assert output.joint_features_z.shape == (3, 9)
    assert torch.equal(output.joint_features_z[:, :4], output.image_features_v)
    path = tmp_path / 'model.pt'
    save_checkpoint(path, model, {'rank': rank})
    restored = build_model(bundle(), config(rank)).eval()
    load_checkpoint(path, restored)
    assert torch.equal(output.logits, restored.forward_batch(batch()).logits)


def test_zero_delta_and_reject_fused_without_partial_injection():
    layer = nn.Linear(3, 6)
    x = torch.randn(2, 3)
    wrapper = LoRALinear(layer, 8, 16, .05)
    assert torch.equal(wrapper(x), layer(x))
    vision = Vision()
    del vision.block.attention.self.value
    with pytest.raises(ValueError, match='Unsupported'):
        inject_qv(vision, 8, 16, .05)
    assert isinstance(vision.block.attention.self.query, nn.Linear)
    with pytest.raises(ValueError, match='fused'):
        inject_qv(nn.Linear(3, 9), 8, 16, .05)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 5)

    def forward_batch(self, data):
        return ModelBatchOutput(logits=self.head(data['pixel_values']))


def concatenate(items):
    return {key: torch.cat([b[key] for b in items]) if torch.is_tensor(items[0][key])
            else sum([b[key] for b in items], []) for key in items[0]}


def test_accumulation_matches_large_batches_with_masking_skips_and_tail():
    torch.manual_seed(5)
    model = Head()
    reference = copy.deepcopy(model)
    a, b, tail, invalid = batch(2), batch(1), batch(2), batch(1)
    a['mask'][0, :4] = 0
    a['labels'][0, :4] = float('nan')
    invalid['mask'].zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=.02)
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=.02)
    result = train_source_epoch(model, [a, invalid, b, tail], optimizer, 'cpu', gradient_accumulation_steps=2)
    train_source_epoch(reference, [concatenate([a, b]), tail], expected_optimizer, 'cpu')
    assert result['optimizer_steps'] == 2 and result['skipped_no_valid_samples'] == 1
    for actual, expected in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(actual, expected, atol=1e-7)
    for value in [0, -1, 1.5, True]:
        with pytest.raises(ValueError):
            train_source_epoch(model, [a], optimizer, 'cpu', gradient_accumulation_steps=value)


def test_source_early_stopping_restores_best_and_records_metadata(tmp_path, monkeypatch):
    from src import trainer
    scores = iter([.8, .7, .6])
    states = []
    def evaluate(model, *args, **kwargs):
        assert kwargs['domain'] == 'mimic' and kwargs['split'] == 'val'
        states.append(copy.deepcopy(model.state_dict()))
        return {'metrics': {'macro_auc': next(scores)}}
    monkeypatch.setattr(trainer, 'evaluate_model', evaluate)
    model = Head()
    result = fit_source(model, [batch()], [batch()], config(), tmp_path, {'weights_sha256': 'test'})
    assert len(result['history']) == 3 and result['best_val_macro_auc'] == .8
    assert all(torch.equal(v, states[0][n]) for n, v in model.state_dict().items())
    metadata = load_checkpoint(result['checkpoint'], model)
    assert metadata['lora']['rank'] == 8 and metadata['provenance']['weights_sha256'] == 'test'
    assert result['elapsed_seconds'] > 0 and result['peak_cuda_memory_bytes'] == 0
