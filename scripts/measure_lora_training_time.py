"""Resource-only replay: preserve original checkpoints, delete successful temporary ones."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import pandas as pd
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.runtime import read_config,seed_everything,write_json
from src.data import load_manifest,make_loader,CLASS_NAMES
from src.model_loader import load_medclip
from src.model_adapter import build_model
from src import trainer

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal-run',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    source=args.formal_run.resolve();BASE=args.output_root.resolve();BASE.mkdir(parents=True,exist_ok=True)
    run=BASE/source.name
    assert not run.exists(),'Refusing to overwrite replay'
    run.mkdir();out=run/'outputs';out.mkdir()
    config=read_config(source/'config.yaml');config['data']['output_dir']=str(out);config['outputs']['output_dir']=str(out)
    (run/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    seed_everything(config['experiment']['seed']);torch.set_num_threads(4)
    frame=load_manifest(Path(config['data']['data_dir'])/'mimic_manifest.csv')
    ids=pd.read_csv(Path(config['data']['data_dir'])/f"source_train_seed{config['experiment']['seed']}_{config['experiment']['budget']}.csv").patient_id
    train=frame[frame.patient_id.isin(ids)&frame.split.eq('train')].sort_values(['patient_id','study_id','image_id'])
    val=frame[frame.split.eq('val')].sort_values(['patient_id','study_id','image_id'])
    assert len(train)==(400 if config['experiment']['budget']=='10pct' else 4000) and len(val)==500
    started=time.perf_counter();bundle=load_medclip(config);model=build_model(bundle,config)
    torch.cuda.synchronize();load_seconds=time.perf_counter()-started
    
    def frozen_hash():
     h=hashlib.sha256()
     for name,t in [(n,p) for n,p in model.named_parameters() if not p.requires_grad]+list(model.named_buffers()):
      h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
     return h.hexdigest()
    before=frozen_hash()
    trainable_before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    records=[];durations={'checkpoint_save_seconds':0.,'checkpoint_restore_seconds':0.}
    original_train=trainer.train_source_epoch;original_eval=trainer.evaluate_model
    original_save=trainer.save_checkpoint;original_load=trainer.load_checkpoint
    
    def training(*args,**kwargs):
     torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
     result=original_train(*args,**kwargs);torch.cuda.synchronize()
     records.append(dict(epoch=len(records)+1,train_only_seconds=time.perf_counter()-start,
                         training_peak_allocated_bytes=torch.cuda.max_memory_allocated(),**result))
     print('TRAIN',json.dumps(records[-1]),flush=True)
     write_json(run/'epoch_timing_progress.json',records)
     return result
    
    def evaluation(*args,**kwargs):
     assert kwargs['domain']=='mimic' and kwargs['split']=='val'
     kwargs['scope']='resource_replay'
     torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();result=original_eval(*args,**kwargs);torch.cuda.synchronize()
     records[-1].update(validation_seconds=time.perf_counter()-start,validation_peak_allocated_bytes=torch.cuda.max_memory_allocated(),val_macro_auc=result['metrics']['macro_auc'])
     print('VAL',json.dumps(records[-1]),flush=True);write_json(run/'epoch_timing_progress.json',records)
     return result
    
    def save(*args,**kwargs):
     torch.cuda.synchronize();start=time.perf_counter();result=original_save(*args,**kwargs);torch.cuda.synchronize()
     durations['checkpoint_save_seconds']+=time.perf_counter()-start
     return result
    
    def restore(*args,**kwargs):
     torch.cuda.synchronize();start=time.perf_counter();result=original_load(*args,**kwargs);torch.cuda.synchronize()
     durations['checkpoint_restore_seconds']+=time.perf_counter()-start
     return result
    trainer.train_source_epoch=training;trainer.evaluate_model=evaluation;trainer.save_checkpoint=save;trainer.load_checkpoint=restore
    loader=lambda rows,shuffle=False:make_loader(rows,bundle.processor,batch_size=config['experiment']['batch_size'],shuffle=shuffle,seed=config['experiment']['seed'])
    result=trainer.fit_source(model,loader(train,True),loader(val),config,out,bundle.provenance)
    assert frozen_hash()==before
    assert all(not torch.equal(p.detach().cpu(),trainable_before[n]) for n,p in model.named_parameters() if p.requires_grad)
    checkpoint=Path(result.pop('checkpoint'));assert checkpoint.parent==out
    best=max(result['history'],key=lambda h:h['val_macro_auc'])
    valid=train[train[[f'mask_{c}' for c in CLASS_NAMES]].sum(axis=1).gt(0)].head(2)
    probe=next(iter(loader(valid)));model.eval()
    with torch.no_grad():pred=model.forward_batch(probe).logits.clone()
    metadata=original_load(checkpoint,model,'cuda')
    with torch.no_grad():difference=float((pred-model.forward_batch(probe).logits).abs().max())
    assert metadata['epoch']==best['epoch'] and difference<=1e-6
    original=json.loads((source/'completed.json').read_text())['training']
    result['fit_final_phase_peak_bytes']=result.pop('peak_cuda_memory_bytes')
    result['training_peak_cuda_memory_bytes']=max(r['training_peak_allocated_bytes'] for r in records)
    result['validation_peak_cuda_memory_bytes']=max(r['validation_peak_allocated_bytes'] for r in records)
    result.update(scope='resource_only_training_replay',run=source.name,epoch_timings=records,
     train_only_seconds=sum(r['train_only_seconds'] for r in records),validation_seconds=sum(r['validation_seconds'] for r in records),
     initialization_seconds=load_seconds,**durations,original_formal_checkpoint_untouched=True,
     frozen_state_unchanged=True,checkpoint_restore_max_abs=difference,temporary_checkpoint_bytes=checkpoint.stat().st_size,
     best_epoch=best['epoch'],original_epochs=len(original['history']),original_best_val_macro_auc=original['best_val_macro_auc'],
     same_epoch_count_as_original=len(result['history'])==len(original['history']),
     timing_definition='synchronized sum of source training epochs, including data loading, preprocessing, backward and optimizer steps; excludes validation/checkpoint I/O/initialization/audit',
     note='new replay timing, not recovered historical timing; no target or source test evaluated')
    write_json(run/'result_before_cleanup.json',result)
    checkpoint.unlink();result['temporary_checkpoint_deleted']=True
    write_json(run/'result.json',result);(run/'result_before_cleanup.json').unlink()
    print('COMPLETE',json.dumps({k:result[k] for k in ['run','train_only_seconds','validation_seconds','temporary_checkpoint_deleted']}),flush=True)

if __name__ == '__main__':
    main()
