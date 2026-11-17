"""Reconstruct data from preserved source and frozen rules. No optimizer or training."""
import pathlib,json,shutil,hashlib,numpy as np,pyarrow as pa,pyarrow.parquet as pq
import argparse
p=argparse.ArgumentParser()
p.add_argument('--source',type=pathlib.Path,required=True)
p.add_argument('--out',type=pathlib.Path,required=True)
a=p.parse_args()
ref=pathlib.Path(__file__).resolve().parent;source=a.source.resolve();out=a.out.resolve()
assert not out.exists(),'Do not overwrite an existing dataset'
spec=json.loads((ref/'poses.json').read_text());saved=json.loads((ref/'stats.json').read_text());cfg=json.loads((ref/'train_config.json').read_text());train=cfg['dataset']['episodes'];assert len(train)==160
df=pq.read_table(source/'data/chunk-000/file-000.parquet').to_pandas();episodes=pq.read_table(source/'meta/episodes/chunk-000/file-000.parquet').to_pylist();frames=[];rows=[];mapping=[];offset=0;train_action=[];train_state=[]
for ep in episodes:
 eid=ep['episode_index'];g=df[df.episode_index==eid].copy();actions=np.stack(g.action);valid=np.flatnonzero(actions[:,spec['signal_action_index']]>=spec['label_rule']['drop_leading_frames_while_signal_below']);assert len(valid)>0
 start=int(valid[0]);g=g.iloc[start:].copy();actions=actions[start:];labels=(actions[:,spec['signal_action_index']]<spec['label_rule']['threshold_rad']).astype(np.float32)
 new=np.concatenate([actions[:,:7],labels[:,None],actions[:,18:21]],axis=1).astype(np.float32);g['action']=list(new);n=len(g);g['frame_index']=np.arange(n,dtype=np.int64);g['timestamp']=(np.arange(n)/10).astype(np.float32);g['index']=np.arange(offset,offset+n,dtype=np.int64)
 if eid in train:train_action.append(new.astype(np.float64));train_state.append(np.stack(g['observation.state']).astype(np.float64))
 row={k:v for k,v in ep.items() if not k.startswith('stats/')};row.update(length=n,dataset_from_index=offset,dataset_to_index=offset+n)
 for camera in ('observation.images.head_cam_h','observation.images.wrist_cam_r'):
  row[f'videos/{camera}/from_timestamp']=ep[f'videos/{camera}/from_timestamp']+start/10
  row[f'videos/{camera}/to_timestamp']=row[f'videos/{camera}/from_timestamp']+n/10
 mapping.append({'episode_index':eid,'dropped_leading_frames':start,'retained_frames':n,'new_from_index':offset,'old_from_index':ep['dataset_from_index'],'label_switches':int(np.count_nonzero(np.diff(labels)))})
 frames.append(g);rows.append(row);offset+=n
assert len(rows)==500
comparisons={}
for key,x in [('action',np.concatenate(train_action)),('observation.state',np.concatenate(train_state))]:
 comparisons[key]={}
 for stat,actual in [('mean',x.mean(0)),('std',x.std(0)),('min',x.min(0)),('max',x.max(0))]:
  expected=np.asarray(saved[key][stat]);diff=float(np.max(np.abs(actual-expected)));comparisons[key][stat]=diff
  assert np.allclose(actual,expected,atol=2e-6,rtol=2e-5),(key,stat,diff)
assert sum(r['retained_frames'] for r in mapping if r['episode_index'] in train)==57212
for folder in ('data/chunk-000','meta/episodes/chunk-000'): (out/folder).mkdir(parents=True,exist_ok=True)
import pandas as pd
table=pa.Table.from_pandas(pd.concat(frames,ignore_index=True),preserve_index=False)
# Keep float32 list types used by the source dataset.
for key in ('action','observation.state'):table=table.set_column(table.schema.get_field_index(key),key,pa.array(table[key].to_pylist(),type=pa.list_(pa.float32())))
pq.write_table(table,out/'data/chunk-000/file-000.parquet');pq.write_table(pa.Table.from_pylist(rows),out/'meta/episodes/chunk-000/file-000.parquet')
info=json.loads((source/'meta/info.json').read_text());info['total_frames']=offset;original_names=info['features']['action']['names']['action_names'];info['features']['action']['shape']=[11];info['features']['action']['names']={'action_names':original_names[:7]+['hand_mode']+original_names[18:21]}
(out/'meta/info.json').write_text(json.dumps(info,indent=2));shutil.copy2(source/'meta/tasks.parquet',out/'meta/tasks.parquet');shutil.copy2(ref/'stats.json',out/'meta/stats.json');shutil.copy2(ref/'poses.json',out/'poses.json')
(out/'videos').symlink_to(source/'videos',target_is_directory=True)
(out/'frame_mapping.json').write_text(json.dumps(mapping,indent=2))
selection={'train_episodes':train,'train_frames':57212,'validation_episodes':None,'original_validation_frames_reported':14301,'held_out_pool_episodes':sorted(set(range(500))-set(train)),'validation_status':'Original 40-episode list not recovered. Remaining 340 episodes are a held-out candidate pool, not the original validation split.'}
(out/'selection_recovered.json').write_text(json.dumps(selection,indent=2))
report={'source_commit':'490bba041cfac362a74ef06febbc5e78d126c9bd','total_episodes':500,'total_retained_frames':offset,'training_episodes':160,'training_frames':57212,'frozen_statistics_comparison':comparisons,'statistics':'Copied preserved training-only statistics after independently checking state/action against reconstructed training frames; image statistics preserved from original artifact','episode_image_statistics':'Removed old per-episode statistics because initialization frames were trimmed; training normalization uses preserved global training-only stats','short_checkpoint':'not recovered and not needed for fresh training','original_validation_list':'not recovered','training_started':False}
(out/'RECONSTRUCTION.json').write_text(json.dumps(report,indent=2))

# Original validation40 was subsequently recovered and verified.
selected=json.loads((ref/'selection.json').read_text(encoding='utf-8-sig'))
val=selected['validation_episodes']
assert len(val)==40 and not set(train)&set(val)
assert sum(r['retained_frames'] for r in mapping if r['episode_index'] in val)==14301
(out/'selection_recovered.json').write_text(json.dumps(selected,indent=2))
report.update(original_validation_list='recovered and verified',validation_episodes=40,validation_frames=14301,selected_frames=71513)
(out/'RECONSTRUCTION.json').write_text(json.dumps(report,indent=2))

print(json.dumps(report,indent=2))
