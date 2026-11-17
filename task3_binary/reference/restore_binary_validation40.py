"""Validate the supplied original split, then persist it. No training."""
import pathlib,json,hashlib,shutil
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
b=pathlib.Path('/root/gpufree-data/openlet');audit=b/'migration_20261111/validation_recovery';candidate=audit/'split_val_candidate.txt';root=b/'datasets/task3_binary_490bba0'
raw=candidate.read_bytes();tokens=raw.decode('utf-8-sig').split();assert len(tokens)==40,'Expected exactly 40 IDs'
val=[int(x) for x in tokens];assert len(set(val))==40 and all(0<=i<500 for i in val)
selection=json.loads((root/'selection_recovered.json').read_text());train=selection['train_episodes'];assert len(train)==160;assert not set(train)&set(val),'Training and validation overlap'
rows=[]
for path in (root/'meta/episodes').rglob('*.parquet'):rows.extend(pq.read_table(path,columns=['episode_index','length']).to_pylist())
counts={int(r['episode_index']):int(r['length']) for r in rows};assert sum(counts[i] for i in train)==57212
assert sum(counts[i] for i in val)==14301,'Does not match the preserved binary validation frame count'
stats_sha=hashlib.sha256((root/'meta/stats.json').read_bytes()).hexdigest()
ds=LeRobotDataset('OpenLET/task3_binary_490bba0',root=root,episodes=val,video_backend='pyav',delta_timestamps={'action':[i/10 for i in range(100)]});assert len(ds)==14301
for index in (0,len(ds)//2,len(ds)-1):
 sample=ds[index];assert tuple(sample['action'].shape)==(100,11);assert tuple(sample['observation.state'].shape)==(18,)
 for camera in ('observation.images.head_cam_h','observation.images.wrist_cam_r'):assert tuple(sample[camera].shape)==(3,480,640)
assert hashlib.sha256((root/'meta/stats.json').read_bytes()).hexdigest()==stats_sha
report={'status':'original binary validation split supplied by user and verified','dataset_commit':'490bba041cfac362a74ef06febbc5e78d126c9bd','validation_episodes':val,'validation_episode_count':40,'validation_frames':14301,'training_episode_count':160,'training_frames':57212,'selected_frames':71513,'overlap':[],'source_split_sha256':hashlib.sha256(raw).hexdigest(),'video_decode_samples':[0,7150,14300],'state_shape':[18],'action_chunk_shape':[100,11],'normalization_unchanged_sha256':stats_sha,'training_started':False}
# Persist only after all membership, frame-count and loader checks pass.
selection.update(validation_episodes=val,validation_frames=14301,selected_episodes=sorted(train+val),selected_frames=71513,unused_episodes=sorted(set(range(500))-set(train)-set(val)),validation_status='Original validation40 supplied by user; 14301 retained frames verified; disjoint from restored train160',validation_split_sha256=report['source_split_sha256'])
(root/'selection_recovered.json').write_text(json.dumps(selection,indent=2));(root/'split_val.txt').write_bytes(raw);(root/'split_train.txt').write_text(''.join(f'{i}\n' for i in train))
reconstruction=json.loads((root/'RECONSTRUCTION.json').read_text());reconstruction.update(original_validation_list='recovered and verified',validation_episodes=40,validation_frames=14301,selected_frames=71513);(root/'RECONSTRUCTION.json').write_text(json.dumps(reconstruction,indent=2))
work=b/'runs/task3_binary_l40';shutil.copy2(root/'selection_recovered.json',work/'selection.json')
(work/'validation_dataset.json').write_text(json.dumps({'repo_id':'OpenLET/task3_binary_490bba0','root':str(root),'episodes':val,'video_backend':'pyav','frames':14301,'stats':'Use unchanged training-only stats from dataset root'},indent=2))
prep=work/'configs/preparation.json';p=json.loads(prep.read_text());p.update(validation_episodes=val,validation_frames=14301,split_verified=True,normalization_verified=True);p['checks_still_required']=['benchmark batch/workers and select a configuration before full training','document any change in optimizer update count when changing batch size'];prep.write_text(json.dumps(p,indent=2))
(audit/'binary_validation40_verified.json').write_text(json.dumps(report,indent=2));(audit/'binary_split_val_original.txt').write_bytes(raw)
old=audit/'binary_validation_recovery_status.json'
if old.exists():
 try:status=json.loads(old.read_text(encoding='utf-8'))
 except UnicodeDecodeError:status=json.loads(old.read_text(encoding='gb18030'))
 status.update(binary_original_validation40_recovered=True,status='Resolved: user supplied original split; see binary_validation40_verified.json',required_artifact=None);old.write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2))
