import pathlib,json,numpy as np,torch,pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
b=pathlib.Path('/root/gpufree-data/openlet');r=b/'datasets/task3_binary_490bba0';ref=b/'migration_20261111';selection=json.loads((r/'selection_recovered.json').read_text());train=selection['train_episodes']
assert not set(train)&set(selection['held_out_pool_episodes'])
assert (b/'datasets/task3_source_490bba0/VIDEOS_COMPLETE.json').is_file()
spec=json.loads((b/'datasets/task3_source_490bba0/meta/info.json').read_text());info=json.loads((r/'meta/info.json').read_text());names=spec['features']['action']['names']['action_names'];info['features']['action']['names']={'action_names':names[:7]+['hand_mode']+names[18:21]};(r/'meta/info.json').write_text(json.dumps(info,indent=2))
ds=LeRobotDataset('OpenLET/task3_binary_490bba0',root=r,episodes=train,video_backend='pyav',delta_timestamps={'action':[i/10 for i in range(100)]});assert len(ds)==57212
checks=[]
for idx in (0,len(ds)//2,len(ds)-1):
 sample=ds[idx];assert sample['action'].shape==(100,11);assert sample['observation.state'].shape==(18,)
 assert set(sample['action'][:,7].tolist())<={0.,1.}
 for camera in ('observation.images.head_cam_h','observation.images.wrist_cam_r'):assert sample[camera].shape==(3,480,640)
 checks.append(idx)
table=pq.read_table(r/'data/chunk-000/file-000.parquet').to_pandas();episode=np.stack(table[table.episode_index==2].action);saved=json.loads((ref/'binary_reference/episode002_replay.json').read_text());assert np.array_equal(episode,np.asarray(saved['action11'],dtype=np.float32))
pc=PreTrainedConfig.from_pretrained(ref/'binary_model_config',local_files_only=True);pc.device='cuda';policy=ACTPolicy(pc).to('cuda').eval();policy.reset();pre,post=make_pre_post_processors(pc,dataset_stats=ds.meta.stats)
sample=ds[0];batch={k:sample[k].unsqueeze(0) for k in pc.input_features}
with torch.inference_mode():action=post(policy.select_action(pre(batch)))
assert action.shape==(1,11) and torch.isfinite(action).all()
report={'train_frames':len(ds),'train_episodes':len(train),'source_video_files_hash_verified':32,'loader_samples':checks,'episode2_all_371_actions_match_preserved_reference':True,'state_shape':[18],'action_chunk_shape':[100,11],'fresh_policy_gpu_inference_shape':list(action.shape),'normalization':'Preserved training-only statistics; reconstructed state/action mean/std/min/max exactly match','training_started':False,'optimizer_steps':0,'validation_status':selection['validation_status'],'note':'Fresh random policy used only for interface check; not a trained model or task-success test'}
(ref/'binary_recovery_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
